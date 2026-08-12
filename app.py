import time

import gradio as gr
import spaces

import database
import pipeline


@spaces.GPU
def _zerogpu_startup_probe():
    """Never called — ZeroGPU (the free-tier Gradio Space hardware) requires at
    least one @spaces.GPU function to exist at startup, even though this app
    is intentionally CPU-only (see CLAUDE.md)."""
    pass


pipeline.startup()

_registered_docs = database.get_registered_documents()
_doc_list_md = (
    "\n".join(f"- {pipeline.get_document_title(name)}" for name in _registered_docs)
    if _registered_docs
    else "_No documents loaded_"
)
_doc_count_text = f"{len(_registered_docs)} documents loaded"

SESSION_JS = """
() => {
    console.log('[MindRAG DEBUG] SESSION_JS running');
    let id = sessionStorage.getItem('mindrag_session_id');
    if (!id) {
        id = crypto.randomUUID();
        sessionStorage.setItem('mindrag_session_id', id);
    }
    console.log('[MindRAG DEBUG] SESSION_JS returning id:', id);
    return id;
}
"""

NEW_SESSION_JS = """
() => {
    const id = crypto.randomUUID();
    sessionStorage.setItem('mindrag_session_id', id);
    return id;
}
"""

SCROLL_TO_QUERY_JS = """
() => {
    const container = document.querySelector('#chatbot .bubble-wrap');
    if (!container) return;
    const userRows = container.querySelectorAll('.message-row.user-row');
    if (!userRows.length) return;
    userRows[userRows.length - 1].scrollIntoView({ behavior: 'smooth', block: 'start' });
}
"""

COOLDOWN_TIMER_JS = """
() => {
    const numberInput = document.querySelector('#cooldown-seconds-box input');
    const seconds = numberInput ? parseFloat(numberInput.value) : 0;
    if (!seconds || seconds <= 0) {
        return;
    }
    const queryInput = document.querySelector('#query-box textarea');
    let remaining = Math.ceil(seconds);
    const tick = () => {
        if (queryInput) {
            queryInput.placeholder = `Rate limited — retry in ${remaining}s...`;
        }
        if (remaining <= 0) {
            clearInterval(intervalId);
            return;
        }
        remaining -= 1;
    };
    tick();
    const intervalId = setInterval(tick, 1000);
    setTimeout(() => {
        const doneBox = document.querySelector('#cooldown-done-box textarea');
        if (doneBox) {
            doneBox.value = Date.now().toString();
            doneBox.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }, seconds * 1000);
}
"""


def _format_sources_display(sources):
    if not sources:
        return ""
    parts = []
    for s in sources:
        label = f"{s['document']} (p.{s['page']})" if s.get("page") is not None else s["document"]
        confidence = s.get("confidence")
        reranker_score = s.get("reranker_score")
        if confidence is not None and reranker_score is not None:
            label += f" [conf {confidence:.2f}, rerank {reranker_score:.2f}]"
        parts.append(label)
    return "\n".join(parts)


def _format_usage_metric(intent, chunks_used, sources):
    """Chunks Used" for QA responses (a count), "Documents Used" (names) for
    summarization/diff responses — same textbox, different label depending on intent."""
    if intent in ("summarization", "diff"):
        names = [s["document"] for s in (sources or [])]
        value = ", ".join(names) if names else "none"
        return gr.update(label="Documents Used", value=value)
    return gr.update(label="Chunks Used", value=f"{chunks_used} chunks")


def _onboarding_entry():
    """Permanent first chat entry — capabilities + example queries + domain
    hint, styled as a markdown blockquote so it reads as system content
    rather than a normal message, without needing custom CSS."""
    content = pipeline.get_onboarding_content(database.get_registered_documents())
    quoted = "\n".join(f"> {line}" if line.strip() else ">" for line in content.split("\n"))
    return [None, quoted]


def _messages_to_chatbot_history(messages):
    history = []
    pending_user = None
    for msg in messages:
        if msg["role"] == "user":
            pending_user = msg["content"]
        elif msg["role"] == "assistant":
            history.append([pending_user, msg["content"]])
            pending_user = None
    return history


_READY_QUERY_BOX = gr.update(interactive=True, placeholder="Ask a question about the documents...")
_READY_SEND_BTN = gr.update(interactive=True)

# Shared "no-op" updates for the tail of the unified output tuple (disambiguation
# popup fields) when a code path has nothing to say about them.
_DISAMBIG_NOOP = (gr.update(), gr.update(), gr.update(), gr.update())
_DISAMBIG_HIDDEN = (gr.update(visible=False), gr.update(), gr.update(), gr.update())


def _rate_limit_updates(retry_after: int):
    gr.Warning(f"Rate limit reached — please try again in {retry_after} seconds.")
    return (
        gr.update(interactive=False, placeholder=f"Rate limited — retry in {retry_after}s..."),
        gr.update(interactive=False),
        retry_after,
    )


def rehydrate_session(session_id):
    print(f"[DEBUG] rehydrate_session called with session_id={session_id!r}", flush=True)
    if not session_id:
        # session_id should always be set by the JS load step; if it somehow isn't,
        # leave the input disabled rather than allow queries with no session to log against.
        return session_id, [], "", "", "", gr.update(), gr.update()
    messages = database.get_all_messages(session_id)
    history = [_onboarding_entry()] + _messages_to_chatbot_history(messages)
    last = database.get_last_assistant_message(session_id)
    if last is None:
        return session_id, history, "", gr.update(label="Chunks Used", value=""), "", _READY_QUERY_BOX, _READY_SEND_BTN
    latency_text = f"{last['latency_ms']} ms" if last["latency_ms"] is not None else ""
    chunks_update = _format_usage_metric(last["intent"], last["chunks_used"], last["sources"])
    sources_text = _format_sources_display(last["sources"])
    return session_id, history, latency_text, chunks_update, sources_text, _READY_QUERY_BOX, _READY_SEND_BTN


def _classify_and_maybe_resolve(final_query):
    """Classifies intent, and for summarization/diff queries also resolves which
    document(s) it refers to. Resolution is computed here (once) so an
    "ambiguous" result can be intercepted before handle_summarization runs —
    diff handles its own ambiguous/single/none cases internally instead of
    popping the UI (see _run_or_pause_for_disambiguation)."""
    intent = pipeline.classify_intent(final_query)
    resolution = None
    if intent in ("summarization", "diff"):
        resolution = pipeline._resolve_documents(final_query, database.get_registered_documents())
    return intent, resolution


def _run_query(raw_query, final_query, was_rewritten, history, session_id, intent, resolution):
    start = time.time()

    if intent == "recap":
        result = pipeline.handle_recap(session_id)
    elif intent == "summarization":
        result = pipeline.handle_summarization(final_query, session_id, resolution)
    elif intent == "diff":
        result = pipeline.handle_diff(final_query, session_id, resolution)
    elif intent == "meta":
        result = pipeline.handle_meta()
    else:
        result = pipeline.handle_qa(final_query, session_id)

    latency_ms = int((time.time() - start) * 1000)

    database.add_message(
        session_id=session_id,
        role="user",
        raw_query=raw_query,
        rewritten_query=final_query if was_rewritten else None,
        content=raw_query,
        intent=intent,
        chunks_used=0,
        sources=None,
        latency_ms=None,
    )
    database.add_message(
        session_id=session_id,
        role="assistant",
        raw_query=None,
        rewritten_query=None,
        content=result["answer"],
        intent=intent,
        chunks_used=result["chunks_used"],
        sources=result["sources"],
        latency_ms=latency_ms,
    )

    updated_history = history + [[raw_query, result["answer"]]]
    latency_text = f"{latency_ms} ms"
    chunks_update = _format_usage_metric(intent, result["chunks_used"], result["sources"])
    sources_text = _format_sources_display(result["sources"])
    show_recovery = intent == "qa" and bool(result.get("is_fallback"))

    return updated_history, latency_text, chunks_update, sources_text, show_recovery


def _disambiguation_pause_tuple(history, raw_query, final_query, was_rewritten, candidates):
    btn_updates = []
    for i in range(3):
        if i < len(candidates):
            btn_updates.append(gr.update(visible=True, value=candidates[i]))
        else:
            btn_updates.append(gr.update(visible=False))
    return (
        history,
        gr.update(visible=False), "", "", "",
        gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(),
        gr.update(visible=True), *btn_updates,
        raw_query, final_query, was_rewritten, candidates,
        gr.update(visible=False),
    )


def _run_or_pause_for_disambiguation(raw_query, final_query, was_rewritten, history, session_id):
    try:
        intent, resolution = _classify_and_maybe_resolve(final_query)

        if intent == "summarization" and resolution and resolution["type"] == "ambiguous":
            return _disambiguation_pause_tuple(
                history, raw_query, final_query, was_rewritten, resolution["candidates"]
            )

        updated_history, latency_text, chunks_text, sources_text, show_recovery = _run_query(
            raw_query, final_query, was_rewritten, history, session_id, intent, resolution
        )
        return (
            updated_history,
            gr.update(visible=False), "", "", "",
            latency_text, chunks_text, sources_text,
            gr.update(), gr.update(), gr.update(),
            *_DISAMBIG_HIDDEN, "", "", False, [],
            gr.update(visible=show_recovery),
        )
    except pipeline.RateLimitedError as exc:
        query_update, send_update, cooldown_value = _rate_limit_updates(exc.retry_after)
        return (
            history,
            gr.update(visible=False), "", "", "",
            gr.update(), gr.update(), gr.update(),
            query_update, send_update, cooldown_value,
            *_DISAMBIG_HIDDEN, "", "", False, [],
            gr.update(visible=False),
        )


def _finish_after_disambiguation(raw_query, final_query, was_rewritten, history, session_id, resolution):
    try:
        updated_history, latency_text, chunks_text, sources_text, _show_recovery = _run_query(
            raw_query, final_query, was_rewritten, history, session_id, "summarization", resolution
        )
        return (
            updated_history,
            gr.update(visible=False), "", "", "",
            latency_text, chunks_text, sources_text,
            gr.update(), gr.update(), gr.update(),
            *_DISAMBIG_HIDDEN, "", "", False, [],
            gr.update(visible=False),
        )
    except pipeline.RateLimitedError as exc:
        query_update, send_update, cooldown_value = _rate_limit_updates(exc.retry_after)
        return (
            history,
            gr.update(visible=False), "", "", "",
            gr.update(), gr.update(), gr.update(),
            query_update, send_update, cooldown_value,
            *_DISAMBIG_HIDDEN, "", "", False, [],
            gr.update(visible=False),
        )


def resolve_disambiguation_choice(chosen_filename, raw_query, final_query, was_rewritten, history, session_id):
    resolution = {"type": "single", "filename": chosen_filename}
    return _finish_after_disambiguation(raw_query, final_query, was_rewritten, history, session_id, resolution)


def resolve_disambiguation_everything(raw_query, final_query, was_rewritten, history, session_id):
    resolution = {"type": "none"}
    return _finish_after_disambiguation(raw_query, final_query, was_rewritten, history, session_id, resolution)


def _noop_tuple(history):
    return (
        history,
        gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(),
    )


def check_rewrite_step(raw_query, history, session_id):
    if not raw_query or not raw_query.strip() or not session_id:
        return _noop_tuple(history)

    try:
        result = pipeline.check_and_rewrite(raw_query, session_id)

        if result["needs_rewrite"]:
            return (
                history,
                gr.update(visible=True), result["rewritten_query"], raw_query, result["rewritten_query"],
                gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                *_DISAMBIG_NOOP, gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(visible=False),
            )

        return _run_or_pause_for_disambiguation(raw_query, raw_query, False, history, session_id)

    except pipeline.RateLimitedError as exc:
        query_update, send_update, cooldown_value = _rate_limit_updates(exc.retry_after)
        return (
            history,
            gr.update(visible=False), "", "", "",
            gr.update(), gr.update(), gr.update(),
            query_update, send_update, cooldown_value,
            *_DISAMBIG_HIDDEN, "", "", False, [],
            gr.update(visible=False),
        )


def confirm_rewrite(rewritten_query, raw_query, history, session_id):
    return _run_or_pause_for_disambiguation(raw_query, rewritten_query, True, history, session_id)


def use_original(raw_query, history, session_id):
    return _run_or_pause_for_disambiguation(raw_query, raw_query, False, history, session_id)


def recovery_show_capabilities(history):
    """'Show me what I can ask' — zero LLM call, just renders the shared
    onboarding content as a new chat entry."""
    content = pipeline.get_onboarding_content(database.get_registered_documents())
    updated_history = history + [[None, content]]
    return updated_history, gr.update(visible=False), gr.update(), gr.update(), gr.update()


def recovery_get_overview(history):
    """'Get an overview of the documents' — a real all-documents summarization
    call, cached after first generation (pipeline.get_documents_overview)."""
    try:
        content = pipeline.get_documents_overview()
        updated_history = history + [[None, content]]
        return updated_history, gr.update(visible=False), gr.update(), gr.update(), gr.update()
    except pipeline.RateLimitedError as exc:
        query_update, send_update, cooldown_value = _rate_limit_updates(exc.retry_after)
        return history, gr.update(visible=False), query_update, send_update, cooldown_value


def recovery_rephrase(history):
    """'Let me rephrase' — closes the popup with no backend call; the next
    message goes through the normal flow from scratch."""
    return history, gr.update(visible=False), gr.update(), gr.update(), gr.update()


with gr.Blocks(theme=gr.themes.Default()) as demo:
    session_state = gr.State(None)
    raw_state = gr.State("")
    rewritten_state = gr.State("")
    session_id_box = gr.Textbox(visible=False)
    cooldown_seconds_box = gr.Number(visible=False, value=0, elem_id="cooldown-seconds-box")
    cooldown_done_box = gr.Textbox(visible=False, elem_id="cooldown-done-box")

    disambig_raw_query_state = gr.State("")
    disambig_final_query_state = gr.State("")
    disambig_was_rewritten_state = gr.State(False)
    disambig_candidates_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# MindRAG")
            gr.Markdown("### Knowledge Base")
            gr.Markdown(_doc_list_md)
            gr.Markdown(_doc_count_text)
            gr.Markdown("Supports: Document QA · Summarization · Diff/Comparison · Conversation Recap")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(bubble_full_width=False, height=500, elem_id="chatbot")

            with gr.Group(visible=False) as rewrite_group:
                rewrite_box = gr.Textbox(label="Suggested rewrite", interactive=False)
                with gr.Row():
                    confirm_btn = gr.Button("✓ Confirm", variant="primary")
                    use_original_btn = gr.Button("Use original", variant="secondary")

            with gr.Group(visible=False) as disambig_group:
                gr.Markdown("Which document are you asking about?")
                disambig_btn_1 = gr.Button(visible=False)
                disambig_btn_2 = gr.Button(visible=False)
                disambig_btn_3 = gr.Button(visible=False)
                disambig_everything_btn = gr.Button("Summarize everything", variant="secondary")

            with gr.Group(visible=False) as recovery_group:
                recovery_show_btn = gr.Button("Show me what I can ask", variant="secondary")
                recovery_overview_btn = gr.Button("Get an overview of the documents", variant="secondary")
                recovery_rephrase_btn = gr.Button("Let me rephrase", variant="secondary")

            with gr.Row():
                query_box = gr.Textbox(
                    placeholder="Loading session...",
                    show_label=False,
                    scale=4,
                    interactive=False,
                    elem_id="query-box",
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, interactive=False)

            with gr.Accordion("Last Response Metrics", open=False):
                latency_box = gr.Textbox(label="Latency", interactive=False)
                chunks_box = gr.Textbox(label="Chunks Used", interactive=False)
                sources_box = gr.Textbox(label="Sources", interactive=False, lines=4)

            clear_btn = gr.Button("Clear Conversation")

    # Unified output tuple shared by every handler that can advance the query
    # flow (send, confirm/cancel rewrite, resolve/bypass disambiguation) — each
    # handler fills in only the fields relevant to its own transition and uses
    # gr.update() no-ops for the rest.
    ALL_OUTPUTS = [
        chatbot,
        rewrite_group, rewrite_box, raw_state, rewritten_state,
        latency_box, chunks_box, sources_box,
        query_box, send_btn, cooldown_seconds_box,
        disambig_group, disambig_btn_1, disambig_btn_2, disambig_btn_3,
        disambig_raw_query_state, disambig_final_query_state,
        disambig_was_rewritten_state, disambig_candidates_state,
        recovery_group,
    ]

    send_btn.click(fn=check_rewrite_step, inputs=[query_box, chatbot, session_state], outputs=ALL_OUTPUTS).then(
        fn=lambda: "", inputs=None, outputs=[query_box]
    ).then(fn=None, inputs=None, outputs=None, js=SCROLL_TO_QUERY_JS)
    query_box.submit(fn=check_rewrite_step, inputs=[query_box, chatbot, session_state], outputs=ALL_OUTPUTS).then(
        fn=lambda: "", inputs=None, outputs=[query_box]
    ).then(fn=None, inputs=None, outputs=None, js=SCROLL_TO_QUERY_JS)

    confirm_btn.click(
        fn=confirm_rewrite,
        inputs=[rewritten_state, raw_state, chatbot, session_state],
        outputs=ALL_OUTPUTS,
    ).then(fn=None, inputs=None, outputs=None, js=SCROLL_TO_QUERY_JS)

    use_original_btn.click(
        fn=use_original,
        inputs=[raw_state, chatbot, session_state],
        outputs=ALL_OUTPUTS,
    ).then(fn=None, inputs=None, outputs=None, js=SCROLL_TO_QUERY_JS)

    disambig_btn_1.click(
        fn=resolve_disambiguation_choice,
        inputs=[
            disambig_btn_1, disambig_raw_query_state, disambig_final_query_state,
            disambig_was_rewritten_state, chatbot, session_state,
        ],
        outputs=ALL_OUTPUTS,
    ).then(fn=None, inputs=None, outputs=None, js=SCROLL_TO_QUERY_JS)
    disambig_btn_2.click(
        fn=resolve_disambiguation_choice,
        inputs=[
            disambig_btn_2, disambig_raw_query_state, disambig_final_query_state,
            disambig_was_rewritten_state, chatbot, session_state,
        ],
        outputs=ALL_OUTPUTS,
    ).then(fn=None, inputs=None, outputs=None, js=SCROLL_TO_QUERY_JS)
    disambig_btn_3.click(
        fn=resolve_disambiguation_choice,
        inputs=[
            disambig_btn_3, disambig_raw_query_state, disambig_final_query_state,
            disambig_was_rewritten_state, chatbot, session_state,
        ],
        outputs=ALL_OUTPUTS,
    ).then(fn=None, inputs=None, outputs=None, js=SCROLL_TO_QUERY_JS)
    disambig_everything_btn.click(
        fn=resolve_disambiguation_everything,
        inputs=[
            disambig_raw_query_state, disambig_final_query_state,
            disambig_was_rewritten_state, chatbot, session_state,
        ],
        outputs=ALL_OUTPUTS,
    ).then(fn=None, inputs=None, outputs=None, js=SCROLL_TO_QUERY_JS)

    recovery_show_btn.click(
        fn=recovery_show_capabilities,
        inputs=[chatbot],
        outputs=[chatbot, recovery_group, query_box, send_btn, cooldown_seconds_box],
    )
    recovery_overview_btn.click(
        fn=recovery_get_overview,
        inputs=[chatbot],
        outputs=[chatbot, recovery_group, query_box, send_btn, cooldown_seconds_box],
    )
    recovery_rephrase_btn.click(
        fn=recovery_rephrase,
        inputs=[chatbot],
        outputs=[chatbot, recovery_group, query_box, send_btn, cooldown_seconds_box],
    )

    clear_btn.click(fn=None, inputs=None, outputs=[session_id_box], js=NEW_SESSION_JS)

    session_id_box.change(
        fn=rehydrate_session,
        inputs=[session_id_box],
        outputs=[session_state, chatbot, latency_box, chunks_box, sources_box, query_box, send_btn],
    )

    cooldown_seconds_box.change(fn=None, inputs=None, outputs=[cooldown_done_box], js=COOLDOWN_TIMER_JS)

    cooldown_done_box.change(
        fn=lambda: (_READY_QUERY_BOX, _READY_SEND_BTN),
        inputs=None,
        outputs=[query_box, send_btn],
    )

    demo.load(fn=None, inputs=None, outputs=[session_id_box], js=SESSION_JS)


if __name__ == "__main__":
    demo.launch()
