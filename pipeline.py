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
from sentence_transformers import CrossEncoder, SentenceTransformer
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

embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
reranker_model = CrossEncoder(config.RERANKER_MODEL)

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(filename: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, filename), encoding="utf-8") as f:
        return f.read()


IMAGE_DESCRIPTION_PROMPT = _load_prompt("image_description.md")
REWRITE_PROMPT = _load_prompt("rewrite.md")
CLASSIFY_PROMPT = _load_prompt("classify.md")
QA_ANSWER_PROMPT = _load_prompt("qa_answer.md")
SUMMARIZE_PROMPT = _load_prompt("summarize.md")
RECAP_PROMPT = _load_prompt("recap.md")

FALLBACK_MESSAGE = "I couldn't find relevant information in the uploaded documents to answer this question."


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
        mime_type = f"image/{base_image.get('ext', 'png')}"
        description = _describe_image(base_image["image"], mime_type)
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

_VALID_INTENTS = {"qa", "summarization", "recap"}


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
        return {"answer": FALLBACK_MESSAGE, "sources": [], "chunks_used": 0}

    pairs = [(final_query, c["text"]) for c in candidates]
    scores = reranker_model.predict(pairs)
    for c, score in zip(candidates, scores):
        c["reranker_score"] = float(score)

    survivors = [c for c in candidates if c["reranker_score"] > config.RERANKER_SCORE_THRESHOLD]

    if not survivors:
        return {"answer": FALLBACK_MESSAGE, "sources": [], "chunks_used": 0}

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

    sources = []
    seen = set()
    for c in selected:
        key = (c["document_name"], c["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"document": c["document_name"], "page": c["page"]})

    return {"answer": answer, "sources": sources, "chunks_used": len(selected)}


# ---------------------------------------------------------------------------
# Query pipeline — summarization (map-reduce)
# ---------------------------------------------------------------------------

def _detect_named_document(query: str, registered_filenames: list[str]):
    query_lower = query.lower()
    for filename in registered_filenames:
        stem = os.path.splitext(filename)[0].lower()
        if stem in query_lower:
            return filename
    return None


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
            partial_summaries.append(_call_groq_text(SUMMARIZE_PROMPT, batch_text))
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
        return _call_groq_text(
            SUMMARIZE_PROMPT, f"Combine these partial summaries into one cohesive summary:\n\n{combined_input}"
        )
    except RateLimitedError:
        raise
    except Exception as exc:
        logger.error("Groq call failed combining summaries: %s", exc)
        return "I encountered an error processing your request. Please try again."


def handle_summarization(final_query: str, session_id: str) -> dict:
    collection = get_collection()
    registered = database.get_registered_documents()
    matched_filename = _detect_named_document(final_query, registered)

    if matched_filename:
        chunks = _fetch_document_chunks(collection, matched_filename)
        summary = _summarize_chunks(chunks)
        return {
            "answer": summary,
            "sources": [{"document": matched_filename, "page": None}],
            "chunks_used": len(chunks),
        }

    total_chunks = 0
    doc_summaries = []
    for filename in registered:
        chunks = _fetch_document_chunks(collection, filename)
        if not chunks:
            continue
        total_chunks += len(chunks)
        doc_summary = _summarize_chunks(chunks)
        doc_summaries.append(f"## {filename}\n{doc_summary}")

    if not doc_summaries:
        return {"answer": FALLBACK_MESSAGE, "sources": [], "chunks_used": 0}

    combined_text = "\n\n".join(doc_summaries)
    try:
        overview = _call_groq_text(
            SUMMARIZE_PROMPT, f"Combine these per-document summaries into one overview:\n\n{combined_text}"
        )
    except RateLimitedError:
        raise
    except Exception as exc:
        logger.error("Groq call failed combining document summaries: %s", exc)
        overview = "I encountered an error processing your request. Please try again."

    sources = [{"document": f, "page": None} for f in registered]
    return {"answer": overview, "sources": sources, "chunks_used": total_chunks}


# ---------------------------------------------------------------------------
# Query pipeline — conversation recap
# ---------------------------------------------------------------------------

def handle_recap(session_id: str) -> dict:
    messages = database.get_all_messages(session_id)
    if not messages:
        return {"answer": "There is no conversation history yet to recap.", "sources": [], "chunks_used": 0}

    history_text = _format_history(messages)
    try:
        answer = _call_groq_text(RECAP_PROMPT, history_text)
    except RateLimitedError:
        raise
    except Exception as exc:
        logger.error("Groq call failed during recap: %s", exc)
        answer = "I encountered an error processing your request. Please try again."

    return {"answer": answer, "sources": [], "chunks_used": 0}
