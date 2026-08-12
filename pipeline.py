import json
import logging
import os
import re
import shutil
import time

import chromadb
import fitz
import google.generativeai as genai
from groq import Groq, RateLimitError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from unstructured.documents.elements import ListItem, NarrativeText, Table, Title
from unstructured.partition.pdf import partition_pdf

import config
import database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline")

genai.configure(api_key=config.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(config.GEMINI_MODEL)

groq_client = Groq(api_key=config.GROQ_API_KEY, max_retries=0)


class RateLimitedError(Exception):
    """Raised when Groq returns a 429. Carries the provider's own reported wait time."""

    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limited by Groq — retry after {retry_after}s")


def _extract_retry_after(exc: RateLimitError) -> int:
    response = getattr(exc, "response", None)
    if response is not None:
        header_value = response.headers.get("retry-after")
        if header_value:
            try:
                return max(1, int(float(header_value)) + 1)
            except ValueError:
                pass
    message = getattr(exc, "message", None) or str(exc)
    match = re.search(r"try again in ([\d.]+)s", message, re.IGNORECASE)
    if match:
        return max(1, int(float(match.group(1))) + 1)
    return 30

# device="cpu" is forced, not inferred: on ZeroGPU a GPU is visible at startup
# but not reliably usable outside an @spaces.GPU-wrapped call, so letting these
# auto-detect cuda silently corrupts every embedding/rerank at query time.
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
reranker_model = CrossEncoder(config.RERANKER_MODEL, device="cpu")

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, filename), encoding="utf-8") as f:
        return f.read()


_DOCUMENT_TITLES_PATH = os.path.join(os.path.dirname(__file__), "document_titles.json")


def _load_document_titles() -> dict:
    if not os.path.isfile(_DOCUMENT_TITLES_PATH):
        return {}
    with open(_DOCUMENT_TITLES_PATH, encoding="utf-8") as f:
        return json.load(f)


_DOCUMENT_TITLES = _load_document_titles()


def get_document_title(filename: str) -> str:
    """Display-only complete title for a registered filename — falls back to the
    filename (minus extension) for any document not in document_titles.json,
    e.g. one added after the mapping was last regenerated."""
    return _DOCUMENT_TITLES.get(filename, os.path.splitext(filename)[0])


IMAGE_DESCRIPTION_PROMPT = _load_prompt("image_description.md")
RESOLVE_DOCUMENTS_PROMPT = _load_prompt("resolve_documents.md")
REWRITE_PROMPT = _load_prompt("rewrite.md")
CLASSIFY_PROMPT = _load_prompt("classify.md")
QA_ANSWER_PROMPT = _load_prompt("qa_answer.md")
SUMMARIZE_PROMPT = _load_prompt("summarize.md")
RECAP_PROMPT = _load_prompt("recap.md")
DIFF_PROMPT = _load_prompt("diff.md")

FALLBACK_MESSAGE = "I couldn't find relevant information in the uploaded documents to answer this question."

QA_FALLBACK_MESSAGE = (
    "I couldn't find relevant information in the uploaded documents to answer this question.\n\n"
    "A couple of things that might help:\n"
    "- Adding more detail or context to your question.\n"
    "- Mentioning a specific document by name — it won't filter the search to just that "
    "document, but it does help match the right passages.\n\n"
    "Or use one of the options below."
)


def _extract_json(text: str):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _call_groq_json(system_prompt: str, user_content: str):
    try:
        response = groq_client.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return _extract_json(response.choices[0].message.content)
    except RateLimitError as exc:
        raise RateLimitedError(_extract_retry_after(exc)) from exc
    except Exception as exc:
        logger.error("Groq call failed: %s", exc)
        return None


def _call_groq_text(system_prompt: str, user_content: str) -> str:
    try:
        response = groq_client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content.strip()
    except RateLimitError as exc:
        raise RateLimitedError(_extract_retry_after(exc)) from exc


def _call_groq_text_resilient(system_prompt: str, user_content: str, max_retries: int = 3) -> str:
    """Same as _call_groq_text, but waits out Groq's reported cooldown and retries
    instead of aborting immediately — map-reduce summarization makes several
    sequential calls for one document and is far more likely to hit the
    per-minute token limit mid-run than a single QA call is."""
    for attempt in range(1, max_retries + 1):
        try:
            return _call_groq_text(system_prompt, user_content)
        except RateLimitedError as exc:
            if attempt == max_retries:
                raise
            logger.warning(
                "Rate limited during summarization — waiting %ss (attempt %d/%d)",
                exc.retry_after, attempt, max_retries,
            )
            time.sleep(exc.retry_after)


# ---------------------------------------------------------------------------
# Ingestion — element processing
# ---------------------------------------------------------------------------

def _html_table_to_markdown(html: str) -> str:
    rows = re.findall(r"<tr.*?>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL)
    table_rows = []
    for row in rows:
        cells = re.findall(r"<t[dh].*?>(.*?)</t[dh]>", row, flags=re.IGNORECASE | re.DOTALL)
        cleaned = [re.sub(r"<.*?>", "", cell).strip() for cell in cells]
        if cleaned:
            table_rows.append(cleaned)
    if not table_rows:
        return re.sub(r"<.*?>", "", html).strip()
    col_count = len(table_rows[0])
    lines = ["| " + " | ".join(table_rows[0]) + " |", "| " + " | ".join(["---"] * col_count) + " |"]
    for row in table_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _table_to_markdown(element) -> str:
    if hasattr(element, "to_markdown"):
        try:
            return element.to_markdown()
        except Exception:
            pass
    html = getattr(getattr(element, "metadata", None), "text_as_html", None)
    if html:
        return _html_table_to_markdown(html)
    return str(element)


def _process_pdf_elements(elements) -> list[dict]:
    raw_blocks = []
    pending_title = None
    narrative_buffer = []
    narrative_page = None
    list_buffer = []
    list_page = None

    def attach_title(text):
        nonlocal pending_title
        if pending_title:
            text = f"{pending_title}\n\n{text}"
            pending_title = None
        return text

    def flush_narrative():
        nonlocal narrative_buffer, narrative_page
        if narrative_buffer:
            text = attach_title("\n".join(narrative_buffer))
            raw_blocks.append({"type": "narrative", "text": text, "page": narrative_page})
            narrative_buffer = []
            narrative_page = None

    def flush_list():
        nonlocal list_buffer, list_page
        if list_buffer:
            text = attach_title("\n".join(f"- {item}" for item in list_buffer))
            raw_blocks.append({"type": "list", "text": text, "page": list_page})
            list_buffer = []
            list_page = None

    for element in elements:
        page = getattr(element.metadata, "page_number", None) or 1

        if isinstance(element, Title):
            flush_narrative()
            flush_list()
            title_text = str(element).strip()
            pending_title = f"{pending_title}\n{title_text}" if pending_title else title_text
            continue

        if isinstance(element, Table):
            flush_narrative()
            flush_list()
            raw_blocks.append(
                {"type": "table", "text": attach_title(_table_to_markdown(element)), "page": page}
            )
            continue

        if isinstance(element, ListItem):
            flush_narrative()
            if list_buffer and list_page != page:
                flush_list()
            list_buffer.append(str(element).strip())
            list_page = page
            continue

        if isinstance(element, NarrativeText):
            flush_list()
            if narrative_buffer and narrative_page != page:
                flush_narrative()
            narrative_buffer.append(str(element).strip())
            narrative_page = page
            continue

    flush_narrative()
    flush_list()
    return raw_blocks


# ---------------------------------------------------------------------------
# Ingestion — image extraction
# ---------------------------------------------------------------------------

def _describe_image(image_bytes: bytes, mime_type: str) -> str | None:
    for attempt in range(1, 4):
        try:
            response = gemini_model.generate_content(
                [IMAGE_DESCRIPTION_PROMPT, {"mime_type": mime_type, "data": image_bytes}]
            )
            text = (response.text or "").strip()
            if not text or text.upper() == "SKIP":
                return None
            return text
        except Exception as exc:
            message = str(exc)
            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                logger.warning("Gemini Vision 429 — skipping image")
                return None
            if attempt < 3:
                logger.warning("Gemini Vision error (attempt %d/3): %s — retrying in 2s", attempt, exc)
                time.sleep(2)
                continue
            logger.warning("Gemini Vision failed after 3 attempts: %s — skipping image", exc)
            return None
    return None


def _extract_page_images(doc, page_number: int) -> list[str]:
    descriptions = []
    page = doc[page_number - 1]
    for img in page.get_images(full=True):
        xref = img[0]
        base_image = doc.extract_image(xref)
        width = base_image.get("width", 0)
        height = base_image.get("height", 0)
        if width < config.MIN_IMAGE_WIDTH or height < config.MIN_IMAGE_HEIGHT:
            continue

        # doc.extract_image() pulls the raw embedded stream, which can come back
        # corrupted (valid-but-garbled pixels) for some colorspace/mask encodings.
        # Rendering the page region instead goes through the same compositing a
        # PDF viewer uses, so it's authoritative. extract_image() is still used
        # above for its (reliable) width/height metadata.
        rects = page.get_image_rects(xref)
        if rects:
            pixmap = page.get_pixmap(dpi=200, clip=rects[0])
            image_bytes = pixmap.tobytes("png")
            mime_type = "image/png"
        else:
            image_bytes = base_image["image"]
            mime_type = f"image/{base_image.get('ext', 'png')}"

        description = _describe_image(image_bytes, mime_type)
        if description:
            descriptions.append(description)
    return descriptions


# ---------------------------------------------------------------------------
# Ingestion — main entry points
# ---------------------------------------------------------------------------

def ingest_pdf(filepath: str, collection) -> int:
    filename = os.path.basename(filepath)

    try:
        elements = partition_pdf(filename=filepath, strategy="hi_res", infer_table_structure=True)
    except Exception as exc:
        logger.warning("hi_res parsing failed for %s (%s) — falling back to fast", filename, exc)
        elements = partition_pdf(filename=filepath, strategy="fast", infer_table_structure=True)

    raw_blocks = _process_pdf_elements(elements)
    blocks_by_page: dict[int, list[dict]] = {}
    for block in raw_blocks:
        blocks_by_page.setdefault(block["page"], []).append(block)

    fitz_doc = fitz.open(filepath)
    total_pages = len(fitz_doc)

    chunks = []
    chunk_index = 0

    for page_number in range(1, total_pages + 1):
        for block in blocks_by_page.get(page_number, []):
            if len(block["text"]) < 1000 and block["type"] != "narrative":
                pieces = [block["text"]]
            else:
                pieces = _text_splitter.split_text(block["text"])

            for piece in pieces:
                chunks.append(
                    {
                        "text": piece,
                        "document_name": filename,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "has_image": False,
                        "element_type": block["type"],
                    }
                )
                chunk_index += 1

        for description in _extract_page_images(fitz_doc, page_number):
            chunks.append(
                {
                    "text": f"[Image on page {page_number}: {description}]",
                    "document_name": filename,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "has_image": True,
                    "element_type": "image",
                }
            )
            chunk_index += 1

    fitz_doc.close()

    if not chunks:
        raise ValueError(f"No content extracted from {filename}")

    embeddings = embedding_model.encode([c["text"] for c in chunks]).tolist()
    ids = [f"{filename}_chunk_{c['chunk_index']}" for c in chunks]
    metadatas = [
        {
            "document_name": c["document_name"],
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "has_image": c["has_image"],
            "element_type": c["element_type"],
        }
        for c in chunks
    ]
    documents_text = [c["text"] for c in chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=documents_text, metadatas=metadatas)

    return len(chunks)


def ingest_document(filepath: str, collection) -> None:
    filename = os.path.basename(filepath)
    try:
        chunk_count = ingest_pdf(filepath, collection)
        database.register_document(filename, chunk_count, status="ready")
        logger.info("Ingested %s (%d chunks)", filename, chunk_count)
    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", filename, exc)
        database.register_document(filename, 0, status="error")


# ---------------------------------------------------------------------------
# Startup — snapshot seeding, ChromaDB client, diff check against /docs/
# ---------------------------------------------------------------------------

_client = None
_collection = None

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshot")


def _seed_from_snapshot() -> None:
    """On ephemeral storage (e.g. HF Spaces free tier), /data is wiped on every
    restart. If no SQLite DB exists yet, restore the committed snapshot so
    startup() sees existing documents and skips re-ingestion."""
    if os.path.exists(config.SQLITE_PATH):
        return

    snapshot_chromadb = os.path.join(SNAPSHOT_DIR, "chromadb")
    snapshot_documents = os.path.join(SNAPSHOT_DIR, "documents.json")
    if not os.path.isdir(snapshot_chromadb) or not os.path.isfile(snapshot_documents):
        return

    logger.info("No existing data found — seeding from committed snapshot")

    os.makedirs(os.path.dirname(config.SQLITE_PATH) or ".", exist_ok=True)
    shutil.copytree(snapshot_chromadb, config.CHROMADB_PATH, dirs_exist_ok=True)

    database.init_db()
    with open(snapshot_documents, encoding="utf-8") as f:
        for row in json.load(f):
            database.register_document(row["filename"], row["chunk_count"], row["status"])
            if row.get("summary"):
                database.update_document_summary(row["filename"], row["summary"])


def get_collection():
    global _client, _collection
    if _collection is None:
        os.makedirs(config.CHROMADB_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(path=config.CHROMADB_PATH)
        _collection = _client.get_or_create_collection(
            name="mindrag_docs",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def startup() -> None:
    _seed_from_snapshot()
    database.init_db()
    collection = get_collection()

    docs_on_disk = sorted(f for f in os.listdir(config.DOCS_FOLDER) if f.lower().endswith(".pdf"))
    if len(docs_on_disk) > config.MAX_DOCUMENTS:
        skipped = docs_on_disk[config.MAX_DOCUMENTS :]
        docs_on_disk = docs_on_disk[: config.MAX_DOCUMENTS]
        logger.warning(
            "More than %d PDFs found in %s — skipping: %s",
            config.MAX_DOCUMENTS,
            config.DOCS_FOLDER,
            skipped,
        )

    registered = set(database.get_registered_documents())
    on_disk = set(docs_on_disk)

    if registered == on_disk:
        logger.info("Knowledge base loaded from cache")
        return

    for filename in registered - on_disk:
        collection.delete(where={"document_name": filename})
        database.delete_document(filename)
        logger.info("Removed %s from knowledge base", filename)

    for filename in sorted(on_disk - registered):
        filepath = os.path.join(config.DOCS_FOLDER, filename)
        ingest_document(filepath, collection)


# ---------------------------------------------------------------------------
# Query pipeline — rewrite check
# ---------------------------------------------------------------------------

def check_and_rewrite(raw_query: str, session_id: str) -> dict:
    last_message = database.get_last_message(session_id)
    if last_message is None:
        return {"needs_rewrite": False, "rewritten_query": raw_query}

    user_content = f"Previous message: {last_message}\n\nCurrent query: {raw_query}"
    result = _call_groq_json(REWRITE_PROMPT, user_content)

    if not result or "needs_rewrite" not in result or "rewritten_query" not in result:
        return {"needs_rewrite": False, "rewritten_query": raw_query}

    return {
        "needs_rewrite": bool(result["needs_rewrite"]),
        "rewritten_query": result["rewritten_query"],
    }


# ---------------------------------------------------------------------------
# Query pipeline — intent classification
# ---------------------------------------------------------------------------

_VALID_INTENTS = {"qa", "summarization", "diff", "recap", "meta", "smalltalk"}


def classify_intent(final_query: str) -> str:
    result = _call_groq_json(CLASSIFY_PROMPT, final_query)
    if not result or result.get("intent") not in _VALID_INTENTS:
        return "qa"
    return result["intent"]


# ---------------------------------------------------------------------------
# Query pipeline — document QA
# ---------------------------------------------------------------------------

def _format_chunks(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Chunk {i} — Source: {chunk['document_name']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def _format_history(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {msg['content']}")
    return "\n".join(lines)


def handle_qa(final_query: str, session_id: str) -> dict:
    num_documents = len(database.get_registered_documents())
    retrieval_k = config.get_retrieval_k(num_documents)
    top_k = config.get_top_k(num_documents)

    query_embedding = embedding_model.encode([final_query])[0].tolist()
    collection = get_collection()

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieval_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("ChromaDB query failed: %s", exc)
        return {"answer": FALLBACK_MESSAGE, "sources": [], "chunks_used": 0}

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    candidates = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        confidence = 1 - distance
        if confidence >= config.CONFIDENCE_THRESHOLD:
            candidates.append({"text": text, **metadata, "confidence": confidence})

    if not candidates:
        return {"answer": QA_FALLBACK_MESSAGE, "sources": [], "chunks_used": 0, "is_fallback": True}

    pairs = [(final_query, c["text"]) for c in candidates]
    scores = reranker_model.predict(pairs)
    for c, score in zip(candidates, scores):
        c["reranker_score"] = float(score)

    survivors = [c for c in candidates if c["reranker_score"] > config.RERANKER_SCORE_THRESHOLD]

    if not survivors:
        return {"answer": QA_FALLBACK_MESSAGE, "sources": [], "chunks_used": 0, "is_fallback": True}

    survivors.sort(key=lambda c: c["reranker_score"], reverse=True)
    selected = survivors[:top_k]

    history = database.get_last_n_messages(session_id, n=config.HISTORY_MESSAGES)

    user_content = (
        f"Document chunks:\n{_format_chunks(selected)}\n\n"
        f"Conversation history:\n{_format_history(history)}\n\n"
        f"Question: {final_query}"
    )

    try:
        answer = _call_groq_text(QA_ANSWER_PROMPT, user_content)
    except RateLimitedError:
        raise
    except Exception as exc:
        logger.error("Groq call failed: %s", exc)
        answer = "I encountered an error processing your request. Please try again."

    sources = [
        {
            "document": c["document_name"],
            "page": c["page"],
            "confidence": round(c["confidence"], 4),
            "reranker_score": round(c["reranker_score"], 4),
        }
        for c in selected
    ]

    return {"answer": answer, "sources": sources, "chunks_used": len(selected)}


# ---------------------------------------------------------------------------
# Query pipeline — summarization (map-reduce)
# ---------------------------------------------------------------------------

_title_embedding_cache = {"filenames": None, "embeddings": None}


def _get_title_embeddings(registered_filenames: list[str]):
    """Encodes the registered document titles once, cached until the
    registered set changes (a new document ingested/removed)."""
    if _title_embedding_cache["filenames"] != registered_filenames:
        titles = [os.path.splitext(f)[0] for f in registered_filenames]
        _title_embedding_cache["embeddings"] = embedding_model.encode(titles)
        _title_embedding_cache["filenames"] = list(registered_filenames)
    return _title_embedding_cache["embeddings"]


def _resolve_documents_embedding_only(
    query: str, registered_filenames: list[str], scored: list, require_conjunction: bool = True
) -> dict:
    """Pure embedding+keyword resolution — used as the fallback when the LLM
    resolver call fails/is unparseable. `scored` is the pre-computed
    (filename, cosine_score) list, sorted descending."""
    top_filename, top_score = scored[0]
    if top_score < config.DOC_RESOLUTION_LOW_FLOOR:
        return {"type": "none"}

    above_floor = [f for f, s in scored if s >= config.DOC_RESOLUTION_LOW_FLOOR]
    query_lower = query.lower()
    has_conjunction = any(kw in query_lower for kw in config.MULTI_DOCUMENT_KEYWORDS)

    if len(above_floor) >= 2 and (has_conjunction or not require_conjunction):
        return {"type": "multi", "filenames": above_floor[: config.MAX_MULTI_DOCUMENTS]}

    second_score = scored[1][1] if len(scored) > 1 else 0.0
    if top_score - second_score >= config.DOC_RESOLUTION_AUTO_MARGIN:
        return {"type": "single", "filename": top_filename}

    return {"type": "ambiguous", "candidates": above_floor[: config.MAX_MULTI_DOCUMENTS]}


def _resolve_documents_llm(query: str, registered_filenames: list[str]) -> list[str] | None:
    """Asks Groq which document(s) the query refers to, grounded in the actual
    registered titles so it can't hallucinate one. Returns a list of exactly-
    matching registered filenames (possibly empty — "no document referenced"),
    or None if the call/parse failed or returned nothing matchable (caller
    falls back to pure embedding resolution)."""
    titles = [get_document_title(f) for f in registered_filenames]
    title_to_filename = dict(zip(titles, registered_filenames))
    doc_list = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    user_content = f"DOCUMENT LIST:\n{doc_list}\n\nQUERY: {query}"

    result = _call_groq_json(RESOLVE_DOCUMENTS_PROMPT, user_content)
    if not result or not isinstance(result.get("documents"), list):
        return None

    resolved = []
    for item in result["documents"]:
        title = item.get("title") if isinstance(item, dict) else item
        if not isinstance(title, str):
            continue
        title = re.sub(r"^\d+[.)]\s*", "", title).strip()
        filename = title_to_filename.get(title)
        if filename and filename not in resolved:
            resolved.append(filename)
    return resolved


def _resolve_documents(query: str, registered_filenames: list[str], require_conjunction: bool = True) -> dict:
    """Resolves which document(s), if any, a query refers to. Hybrid: an LLM
    call (grounded in the actual titles) identifies WHICH document(s) are
    referenced — it handles arbitrary phrasing ("differ from", a bare comma
    list, etc.) that keyword gating misses. Embedding cosine similarity
    supplies a calibrated ambiguity check for the single-document case — the
    LLM tends to confidently pick one document even when several are
    genuinely plausible, where an explicit numeric margin correctly flags
    that as uncertain. Falls back to pure embedding+keyword resolution if the
    LLM call fails.

    `require_conjunction` gates whether 2+ LLM-identified documents count as
    a genuine multi-document request (needs "and"/"compare"/etc. in the
    query) or just ambiguity between candidates for a singular request (e.g.
    "the deep learning paper" when two documents plausibly match). Callers
    for `diff` intent should pass False — classify_intent() already requires
    diff queries to reference 2+ documents, so the gate would only risk
    false "ambiguous" results for phrasings like "how does X differ from Y"
    that don't contain a listed keyword.

    Returns one of:
      {"type": "none"}                          — no document referenced
      {"type": "single", "filename": ...}       — one confident match
      {"type": "multi", "filenames": [...]}     — multiple confident matches
      {"type": "ambiguous", "candidates": [...]} — top-scoring but unclear which one
    """
    if not registered_filenames:
        return {"type": "none"}

    title_embeddings = _get_title_embeddings(registered_filenames)
    query_embedding = embedding_model.encode([query])[0]
    scores = util.cos_sim(query_embedding, title_embeddings)[0].tolist()
    scored = sorted(zip(registered_filenames, scores), key=lambda x: x[1], reverse=True)
    score_by_filename = dict(zip(registered_filenames, scores))

    llm_docs = _resolve_documents_llm(query, registered_filenames)
    if llm_docs is None:
        return _resolve_documents_embedding_only(query, registered_filenames, scored, require_conjunction)

    if len(llm_docs) == 0:
        return {"type": "none"}

    if len(llm_docs) >= 2:
        has_conjunction = any(kw in query.lower() for kw in config.MULTI_DOCUMENT_KEYWORDS)
        if has_conjunction or not require_conjunction:
            return {"type": "multi", "filenames": llm_docs[: config.MAX_MULTI_DOCUMENTS]}
        return {"type": "ambiguous", "candidates": llm_docs[: config.MAX_MULTI_DOCUMENTS]}

    chosen = llm_docs[0]
    chosen_score = score_by_filename.get(chosen, 0.0)
    others = [(f, s) for f, s in scored if f != chosen]
    runner_up_score = others[0][1] if others else 0.0

    if chosen_score - runner_up_score >= config.DOC_RESOLUTION_AUTO_MARGIN:
        return {"type": "single", "filename": chosen}

    candidates = [chosen] + [f for f, s in others if s >= config.DOC_RESOLUTION_LOW_FLOOR]
    return {"type": "ambiguous", "candidates": candidates[: config.MAX_MULTI_DOCUMENTS]}


def _fetch_document_chunks(collection, filename=None) -> list[dict]:
    result = collection.get(where={"document_name": filename}) if filename else collection.get()
    chunks = [
        {"text": doc, **meta}
        for doc, meta in zip(result.get("documents", []), result.get("metadatas", []))
    ]
    chunks.sort(key=lambda c: c["chunk_index"])
    return chunks


def _summarize_chunks(chunks: list[dict]) -> str:
    batches = [
        chunks[i : i + config.SUMMARY_BATCH_SIZE] for i in range(0, len(chunks), config.SUMMARY_BATCH_SIZE)
    ]
    partial_summaries = []
    for batch in batches:
        batch_text = "\n\n".join(c["text"] for c in batch)
        try:
            partial_summaries.append(_call_groq_text_resilient(SUMMARIZE_PROMPT, batch_text))
        except RateLimitedError:
            raise
        except Exception as exc:
            logger.error("Groq call failed during summarization batch: %s", exc)

    if not partial_summaries:
        return "I encountered an error processing your request. Please try again."

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    combined_input = "\n\n".join(f"Partial summary {i + 1}:\n{s}" for i, s in enumerate(partial_summaries))
    try:
        return _call_groq_text_resilient(
            SUMMARIZE_PROMPT, f"Combine these partial summaries into one cohesive summary:\n\n{combined_input}"
        )
    except RateLimitedError:
        raise
    except Exception as exc:
        logger.error("Groq call failed combining summaries: %s", exc)
        return "I encountered an error processing your request. Please try again."


_DOCS_PER_ANSWER_BATCH = config.MAX_MULTI_DOCUMENTS


def _answer_from_documents(content_blocks: list[str], question: str, prompt: str = SUMMARIZE_PROMPT) -> str:
    """Answers `question` from a list of per-document content blocks (pre-baked
    or fallback-generated summaries), using `prompt` as the system prompt.
    Single-call for a small number of documents; batches otherwise — combining
    all documents' summaries in one request (the "summarize everything" case)
    would otherwise exceed the free-tier TPM limit."""
    if len(content_blocks) <= _DOCS_PER_ANSWER_BATCH:
        combined = "\n\n".join(content_blocks)
        return _call_groq_text_resilient(prompt, f"{combined}\n\nQuestion: {question}")

    batches = [
        content_blocks[i : i + _DOCS_PER_ANSWER_BATCH]
        for i in range(0, len(content_blocks), _DOCS_PER_ANSWER_BATCH)
    ]
    partial_answers = []
    for batch in batches:
        combined = "\n\n".join(batch)
        partial_answers.append(
            _call_groq_text_resilient(prompt, f"{combined}\n\nQuestion: {question}")
        )
    combined_partials = "\n\n".join(f"Partial answer {i + 1}:\n{a}" for i, a in enumerate(partial_answers))
    return _call_groq_text_resilient(prompt, f"{combined_partials}\n\nQuestion: {question}")


def _gather_document_content(collection, filenames: list[str]) -> tuple[list[str], list[str], int]:
    """Fetches each document's content to answer from — its pre-baked summary,
    or a live map-reduce fallback if it has none. Returns (content_blocks,
    documents_used, chunks_used_total); shared by handle_summarization and
    handle_diff so both stay consistent about the NULL-summary fallback."""
    content_blocks = []
    documents_used = []
    chunks_used_total = 0

    for filename in filenames:
        summary = database.get_document_summary(filename)
        if not summary:
            chunks = _fetch_document_chunks(collection, filename)
            if not chunks:
                continue
            summary = _summarize_chunks(chunks)
            chunks_used_total += len(chunks)
        content_blocks.append(f"## {filename}\n{summary}")
        documents_used.append(filename)

    return content_blocks, documents_used, chunks_used_total


def handle_summarization(final_query: str, session_id: str, resolution: dict) -> dict:
    """resolution is the output of _resolve_documents() — computed by the caller
    (app.py) so it can intercept an "ambiguous" result with a disambiguation
    popup before ever reaching this function. A "single"/"multi" resolution
    answers from the named document(s); anything else (including "ambiguous",
    which app.py should not be passing through) falls back to all-documents mode."""
    collection = get_collection()
    resolution_type = resolution.get("type", "none")

    if resolution_type == "single":
        filenames = [resolution["filename"]]
    elif resolution_type == "multi":
        filenames = resolution["filenames"]
    else:
        filenames = database.get_registered_documents()

    content_blocks, documents_used, chunks_used_total = _gather_document_content(collection, filenames)

    if not content_blocks:
        return {"answer": FALLBACK_MESSAGE, "sources": [], "chunks_used": 0, "documents_used": []}

    try:
        answer = _answer_from_documents(content_blocks, final_query)
    except RateLimitedError:
        raise
    except Exception as exc:
        logger.error("Groq call failed during summarization: %s", exc)
        answer = "I encountered an error processing your request. Please try again."

    sources = [{"document": f, "page": None} for f in documents_used]
    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": chunks_used_total,
        "documents_used": documents_used,
    }


_overview_cache = {"filenames": None, "content": None}


def get_documents_overview() -> str:
    """Cached 'summarize everything' answer for the recovery popup's 'Get an
    overview' button — identical on every request since it isn't tied to a
    specific user question and the knowledge base only changes at startup, so
    it's generated once (real Groq call, may raise RateLimitedError) and
    reused until the registered document set changes."""
    registered = database.get_registered_documents()
    if _overview_cache["filenames"] == registered:
        return _overview_cache["content"]

    result = handle_summarization(
        "Give me a general overview of these documents.",
        session_id=None,
        resolution={"type": "none"},
    )
    _overview_cache["content"] = result["answer"]
    _overview_cache["filenames"] = list(registered)
    return _overview_cache["content"]


# ---------------------------------------------------------------------------
# Onboarding — domain-hint generation
# ---------------------------------------------------------------------------

_DOMAIN_HINT_PROMPT = (
    "You are summarizing the subject domain of a document collection for a chat "
    "assistant's onboarding message. Given summaries of one or more documents (or "
    "partial domain descriptions from earlier batches), respond with ONE short "
    'sentence (max ~20 words) describing the overall subject domain(s) covered — '
    'e.g. "machine learning research, healthcare AI, and human-robot interaction". '
    "No preamble, no markdown, just the sentence."
)

_domain_hint_cache = {"filenames": None, "hint": None}


def get_domain_hint(registered_filenames: list[str]) -> str:
    """One-time Groq call summarizing the corpus's subject domain, cached and
    regenerated only when the registered document set changes — same
    invalidation trigger as _get_title_embeddings. Used by the shared
    onboarding content (Step 5); never blocks startup on failure."""
    if _domain_hint_cache["filenames"] == registered_filenames:
        return _domain_hint_cache["hint"]

    content_blocks = []
    for filename in registered_filenames:
        summary = database.get_document_summary(filename)
        if summary:
            content_blocks.append(f"## {get_document_title(filename)}\n{summary}")

    hint = ""
    if content_blocks:
        try:
            hint = _answer_from_documents(
                content_blocks,
                "In one short sentence (max ~20 words), describe the overall "
                "subject domain(s) this collection of documents covers.",
                prompt=_DOMAIN_HINT_PROMPT,
            )
        except Exception as exc:
            logger.error("Domain-hint generation failed: %s", exc)
            hint = ""

    _domain_hint_cache["hint"] = hint
    _domain_hint_cache["filenames"] = list(registered_filenames)
    return hint


# ---------------------------------------------------------------------------
# Onboarding — shared capabilities + example-queries content
# ---------------------------------------------------------------------------

_onboarding_content_cache = {"filenames": None, "content": None}


def _build_example_queries(registered_filenames: list[str]) -> dict:
    """One grounded example query per intent, built from real registered
    document titles rather than generic placeholders."""
    if not registered_filenames:
        return {}
    titles = [get_document_title(f) for f in registered_filenames]
    diff_b = titles[1] if len(titles) > 1 else titles[0]
    return {
        "qa": f'"What are the key findings in {titles[0]}?"',
        "summarization": f'"Summarize {titles[0]}"',
        "diff": f'"Compare {titles[0]} and {diff_b}"',
        "recap": '"What have we discussed so far?"',
    }


def _build_onboarding_content(registered_filenames: list[str]) -> str:
    examples = _build_example_queries(registered_filenames)
    domain_hint = get_domain_hint(registered_filenames)

    lines = ["**What I can help with:**"]
    lines.append("- Answer questions about the documents (Document QA)")
    lines.append("- Summarize one or more documents (Summarization)")
    lines.append("- Compare two or more documents (Diff/Comparison)")
    lines.append("- Recap our conversation so far (Conversation Recap)")

    if domain_hint:
        lines.append(f"\nThis knowledge base covers: {domain_hint}")

    if examples:
        lines.append("\n**Example queries:**")
        lines.append(f"- {examples['qa']}")
        lines.append(f"- {examples['summarization']}")
        lines.append(f"- {examples['diff']}")
        lines.append(f"- {examples['recap']}")

    return "\n".join(lines)


def get_onboarding_content(registered_filenames: list[str]) -> str:
    """Shared capabilities + example-queries + domain-hint content, built once
    and cached — reused by the permanent chat header block (app.py), the
    meta-intent handler, and the recovery popup's 'show me what I can ask'
    button. Regenerated only when the registered document set changes."""
    if _onboarding_content_cache["filenames"] != registered_filenames:
        _onboarding_content_cache["content"] = _build_onboarding_content(registered_filenames)
        _onboarding_content_cache["filenames"] = list(registered_filenames)
    return _onboarding_content_cache["content"]


def handle_meta() -> dict:
    """Capability/onboarding queries ('what can I ask?') — zero additional
    LLM call, just returns the shared onboarding content."""
    answer = get_onboarding_content(database.get_registered_documents())
    return {"answer": answer, "sources": [], "chunks_used": 0}


SMALLTALK_REPLY = (
    "Hi there! I'm here to help with the documents — ask me a question, "
    "request a summary or comparison, or ask for a recap of our chat "
    'whenever you\'re ready. Type "what can I ask?" if you\'d like a few examples.'
)


def handle_smalltalk() -> dict:
    """Pure greetings/thanks/chit-chat with no actionable request — zero LLM
    call, a fixed friendly reply distinct from the fuller onboarding content
    (that's reserved for an explicit 'meta' capability question)."""
    return {"answer": SMALLTALK_REPLY, "sources": [], "chunks_used": 0}


# ---------------------------------------------------------------------------
# Query pipeline — document comparison (diff)
# ---------------------------------------------------------------------------

def handle_diff(final_query: str, session_id: str, resolution: dict) -> dict:
    """A diff needs two or more identified documents — unlike summarization,
    there's no sensible "compare everything" fallback, so a single/ambiguous/
    none resolution returns a clarifying response instead of guessing or
    reusing the disambiguation popup (that stays summarization-only; its
    "summarize everything" bypass button has no diff equivalent)."""
    resolution_type = resolution.get("type", "none")

    if resolution_type != "multi":
        if resolution_type == "single":
            answer = (
                f"I can only compare two or more documents, but this only matched one: "
                f"{resolution['filename']}. Try naming a second document to compare it against."
            )
        elif resolution_type == "ambiguous":
            names = ", ".join(resolution["candidates"])
            answer = (
                f"I'm not sure which documents you want to compare — possible matches: {names}. "
                f"Try naming them more specifically."
            )
        else:
            answer = "I couldn't identify which documents you want to compare. Try naming them directly."
        return {"answer": answer, "sources": [], "chunks_used": 0, "documents_used": []}

    collection = get_collection()
    content_blocks, documents_used, chunks_used_total = _gather_document_content(
        collection, resolution["filenames"]
    )

    if len(documents_used) < 2:
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": [],
            "chunks_used": chunks_used_total,
            "documents_used": documents_used,
        }

    try:
        answer = _answer_from_documents(content_blocks, final_query, prompt=DIFF_PROMPT)
    except RateLimitedError:
        raise
    except Exception as exc:
        logger.error("Groq call failed during diff: %s", exc)
        answer = "I encountered an error processing your request. Please try again."

    sources = [{"document": f, "page": None} for f in documents_used]
    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": chunks_used_total,
        "documents_used": documents_used,
    }


# ---------------------------------------------------------------------------
# Query pipeline — conversation recap
# ---------------------------------------------------------------------------

def handle_recap(session_id: str) -> dict:
    messages = database.get_all_messages(session_id)
    if not messages:
        return {
            "answer": (
                "It looks like we haven't chatted yet, so there's nothing to recap. "
                "Ask me something about the documents first, and I'll be able to "
                "summarize our conversation from there."
            ),
            "sources": [],
            "chunks_used": 0,
        }

    history_text = _format_history(messages)
    try:
        answer = _call_groq_text(RECAP_PROMPT, history_text)
    except RateLimitedError:
        raise
    except Exception as exc:
        logger.error("Groq call failed during recap: %s", exc)
        answer = "I encountered an error processing your request. Please try again."

    return {"answer": answer, "sources": [], "chunks_used": 0}
