import time

import gradio as gr

import database
import pipeline

pipeline.startup()

_registered_docs = database.get_registered_documents()
_doc_list_md = "\n".join(f"- {name}" for name in _registered_docs) if _registered_docs else "_No documents loaded_"
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

COOLDOWN_TIMER_JS = """
() => {
    const numberInput = document.querySelector('#cooldown-seconds-box input');
    const seconds = numberInput ? parseFloat(numberInput.value) : 0;
    if (!seconds || seconds <= 0) {
        return;
    }
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
        if s.get("page") is not None:
            parts.append(f"{s['document']} (p.{s['page']})")
        else:
            parts.append(s["document"])
    return ", ".join(parts)


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
    history = _messages_to_chatbot_history(messages)
    last = database.get_last_assistant_message(session_id)
    if last is None:
        return session_id, history, "", "", "", _READY_QUERY_BOX, _READY_SEND_BTN
    latency_text = f"{last['latency_ms']} ms" if last["latency_ms"] is not None else ""
    chunks_text = f"{last['chunks_used']} chunks" if last["chunks_used"] is not None else ""
    sources_text = _format_sources_display(last["sources"])
    return session_id, history, latency_text, chunks_text, sources_text, _READY_QUERY_BOX, _READY_SEND_BTN


def _run_query(raw_query, final_query, was_rewritten, history, session_id):
    start = time.time()
    intent = pipeline.classify_intent(final_query)

    if intent == "recap":
        result = pipeline.handle_recap(session_id)
    elif intent == "summarization":
        result = pipeline.handle_summarization(final_query, session_id)
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
    chunks_text = f"{result['chunks_used']} chunks"
    sources_text = _format_sources_display(result["sources"])

    return updated_history, latency_text, chunks_text, sources_text


def check_rewrite_step(raw_query, history, session_id):
    if not raw_query or not raw_query.strip() or not session_id:
        return (
            history,
            gr.update(visible=False),
            "",
            "",
            "",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    try:
        result = pipeline.check_and_rewrite(raw_query, session_id)

        if result["needs_rewrite"]:
            return (
                history,
                gr.update(visible=True),
                result["rewritten_query"],
                raw_query,
                result["rewritten_query"],
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )

        updated_history, latency_text, chunks_text, sources_text = _run_query(
            raw_query, raw_query, False, history, session_id
        )
        return (
            updated_history,
            gr.update(visible=False),
            "",
            "",
            "",
            latency_text,
            chunks_text,
            sources_text,
            gr.update(),
            gr.update(),
            gr.update(),
        )
    except pipeline.RateLimitedError as exc:
        query_update, send_update, cooldown_value = _rate_limit_updates(exc.retry_after)
        return (
            history,
            gr.update(visible=False),
            "",
            "",
            "",
            gr.update(),
            gr.update(),
            gr.update(),
            query_update,
            send_update,
            cooldown_value,
        )


def confirm_rewrite(rewritten_query, raw_query, history, session_id):
    try:
        updated_history, latency_text, chunks_text, sources_text = _run_query(
            raw_query, rewritten_query, True, history, session_id
        )
        return (
            updated_history,
            gr.update(visible=False),
            latency_text,
            chunks_text,
            sources_text,
            gr.update(),
            gr.update(),
            gr.update(),
        )
    except pipeline.RateLimitedError as exc:
        query_update, send_update, cooldown_value = _rate_limit_updates(exc.retry_after)
        return (
            history,
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(),
            query_update,
            send_update,
            cooldown_value,
        )


def use_original(raw_query, history, session_id):
    try:
        updated_history, latency_text, chunks_text, sources_text = _run_query(
            raw_query, raw_query, False, history, session_id
        )
        return (
            updated_history,
            gr.update(visible=False),
            latency_text,
            chunks_text,
            sources_text,
            gr.update(),
            gr.update(),
            gr.update(),
        )
    except pipeline.RateLimitedError as exc:
        query_update, send_update, cooldown_value = _rate_limit_updates(exc.retry_after)
        return (
            history,
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(),
            query_update,
            send_update,
            cooldown_value,
        )


with gr.Blocks(theme=gr.themes.Default()) as demo:
    session_state = gr.State(None)
    raw_state = gr.State("")
    rewritten_state = gr.State("")
    session_id_box = gr.Textbox(visible=False)
    cooldown_seconds_box = gr.Number(visible=False, value=0, elem_id="cooldown-seconds-box")
    cooldown_done_box = gr.Textbox(visible=False, elem_id="cooldown-done-box")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# MindRAG")
            gr.Markdown("### Knowledge Base")
            gr.Markdown(_doc_list_md)
            gr.Markdown(_doc_count_text)
            gr.Markdown("Supports: Document QA · Summarization · Conversation Recap")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(bubble_full_width=False, height=500)

            with gr.Group(visible=False) as rewrite_group:
                rewrite_box = gr.Textbox(label="Suggested rewrite", interactive=False)
                with gr.Row():
                    confirm_btn = gr.Button("✓ Confirm", variant="primary")
                    use_original_btn = gr.Button("Use original", variant="secondary")

            with gr.Row():
                query_box = gr.Textbox(
                    placeholder="Loading session...",
                    show_label=False,
                    scale=4,
                    interactive=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, interactive=False)

            with gr.Accordion("Last Response Metrics", open=False):
                latency_box = gr.Textbox(label="Latency", interactive=False)
                chunks_box = gr.Textbox(label="Chunks Used", interactive=False)
                sources_box = gr.Textbox(label="Sources", interactive=False)

            clear_btn = gr.Button("Clear Conversation")

    send_outputs = [
        chatbot,
        rewrite_group,
        rewrite_box,
        raw_state,
        rewritten_state,
        latency_box,
        chunks_box,
        sources_box,
        query_box,
        send_btn,
        cooldown_seconds_box,
    ]

    send_btn.click(fn=check_rewrite_step, inputs=[query_box, chatbot, session_state], outputs=send_outputs).then(
        fn=lambda: "", inputs=None, outputs=[query_box]
    )
    query_box.submit(fn=check_rewrite_step, inputs=[query_box, chatbot, session_state], outputs=send_outputs).then(
        fn=lambda: "", inputs=None, outputs=[query_box]
    )

    rewrite_outputs = [chatbot, rewrite_group, latency_box, chunks_box, sources_box, query_box, send_btn, cooldown_seconds_box]

    confirm_btn.click(
        fn=confirm_rewrite,
        inputs=[rewritten_state, raw_state, chatbot, session_state],
        outputs=rewrite_outputs,
    )

    use_original_btn.click(
        fn=use_original,
        inputs=[raw_state, chatbot, session_state],
        outputs=rewrite_outputs,
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
