"""Flask application for LLM Chat UI."""

import atexit
import json
from flask import Flask, render_template, request, Response, jsonify

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
from model_scanner import get_all_models
from chat_storage import (
    create_chat, save_chat, load_chat, delete_chat,
    list_chats, add_message
)
from backend_manager import backend_manager
from llm_client import stream_chat

app = Flask(__name__)


# Cleanup on shutdown
atexit.register(backend_manager.cleanup)


# ============== Page Routes ==============

@app.route("/")
def index():
    """Render the main chat page."""
    return render_template("index.html")


# ============== Backend API ==============

@app.route("/api/backends")
def api_backends():
    """Get list of available backends."""
    return jsonify({
        "backends": ["vllm", "llamacpp"],
        "status": backend_manager.get_status()
    })


@app.route("/api/backends/status")
def api_backend_status():
    """Get current backend status."""
    return jsonify(backend_manager.get_status())


@app.route("/api/backends/start", methods=["POST"])
def api_backend_start():
    """Start a backend with SSE progress updates."""
    data = request.get_json()
    backend = data.get("backend")
    model_path = data.get("model_path")

    if not backend or not model_path:
        return jsonify({"error": "backend and model_path required"}), 400

    def generate():
        success = False
        for msg in backend_manager.start_backend(backend, model_path):
            yield f"data: {json.dumps({'message': msg})}\n\n"
            if "is ready" in msg:
                success = True

        yield f"data: {json.dumps({'done': True, 'success': success})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/backends/stop", methods=["POST"])
def api_backend_stop():
    """Stop the current backend."""
    messages = list(backend_manager.stop_current())
    return jsonify({"messages": messages, "status": backend_manager.get_status()})


# ============== Models API ==============

@app.route("/api/models")
def api_models():
    """Get all available models for all backends."""
    return jsonify(get_all_models())


@app.route("/api/models/<backend>")
def api_models_backend(backend):
    """Get available models for a specific backend."""
    all_models = get_all_models()
    if backend not in all_models:
        return jsonify({"error": f"Unknown backend: {backend}"}), 400
    return jsonify({"models": all_models[backend]})


# ============== Chat API ==============

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Send a message and stream the response."""
    data = request.get_json()

    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 2048)
    top_p = data.get("top_p", 0.9)

    # Check backend status
    status = backend_manager.get_status()
    if status["status"] != "running":
        return jsonify({"error": "No backend running"}), 400

    backend = status["backend"]

    def generate():
        content_buffer = ""
        reasoning_buffer = ""

        for chunk in stream_chat(backend, messages, temperature, max_tokens, top_p):
            if chunk["type"] == "content":
                content_buffer += chunk["text"]
                yield f"data: {json.dumps(chunk)}\n\n"
            elif chunk["type"] == "reasoning":
                reasoning_buffer += chunk["text"]
                yield f"data: {json.dumps(chunk)}\n\n"
            elif chunk["type"] == "error":
                yield f"data: {json.dumps(chunk)}\n\n"
            elif chunk["type"] == "done":
                yield f"data: {json.dumps({'type': 'done', 'full_content': content_buffer, 'full_reasoning': reasoning_buffer})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


# ============== Chat History API ==============

@app.route("/api/chats")
def api_chats_list():
    """List all chats."""
    return jsonify({"chats": list_chats()})


@app.route("/api/chats", methods=["POST"])
def api_chats_create():
    """Create a new chat."""
    data = request.get_json() or {}
    backend = data.get("backend", "")
    model = data.get("model", "")
    chat = create_chat(backend, model)
    return jsonify(chat)


@app.route("/api/chats/<chat_id>")
def api_chats_get(chat_id):
    """Get a specific chat."""
    chat = load_chat(chat_id)
    if chat is None:
        return jsonify({"error": "Chat not found"}), 404
    return jsonify(chat)


@app.route("/api/chats/<chat_id>", methods=["PUT"])
def api_chats_update(chat_id):
    """Update a chat."""
    data = request.get_json()
    chat = load_chat(chat_id)

    if chat is None:
        return jsonify({"error": "Chat not found"}), 404

    # Update fields
    if "title" in data:
        chat["title"] = data["title"]
    if "messages" in data:
        chat["messages"] = data["messages"]
    if "backend" in data:
        chat["backend"] = data["backend"]
    if "model" in data:
        chat["model"] = data["model"]

    save_chat(chat_id, chat)
    return jsonify(chat)


@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def api_chats_delete(chat_id):
    """Delete a chat."""
    if delete_chat(chat_id):
        return jsonify({"success": True})
    return jsonify({"error": "Chat not found"}), 404


@app.route("/api/chats/<chat_id>/messages", methods=["POST"])
def api_chats_add_message(chat_id):
    """Add a message to a chat."""
    data = request.get_json()
    role = data.get("role")
    content = data.get("content")
    reasoning = data.get("reasoning", "")

    if not role or content is None:
        return jsonify({"error": "role and content required"}), 400

    if add_message(chat_id, role, content, reasoning):
        return jsonify({"success": True})
    return jsonify({"error": "Chat not found"}), 404


if __name__ == "__main__":
    print(f"Starting LLM Chat UI on http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG, threaded=True)
