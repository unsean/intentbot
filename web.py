"""Flask web interface for the chatbot.

Run: python web.py  -> http://localhost:5000
"""

import json
import queue
import threading

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from main import load_or_train_assistant

app = Flask(__name__)
CORS(app)

chatbot = None


def initialize_chatbot(force_train: bool = False) -> bool:
    """Load or train the assistant. Returns True when ready."""
    global chatbot
    try:
        print("Loading chatbot...")
        chatbot = load_or_train_assistant(force_train=force_train)
        print(f"Chatbot ready ({chatbot.ai_name}).")
        return True
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")
        return False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    user_id = data.get("user_id", "web_user")
    if not message:
        return jsonify({"error": "Empty message"}), 400
    if chatbot is None or not chatbot.is_ready:
        return jsonify({"error": "Chatbot not initialized"}), 503
    try:
        response = chatbot.handle_message(message, user_id)
        return jsonify({
            "response": response.text,
            "intent": response.intent,
            "confidence": round(response.confidence, 3),
            "entities": response.entities,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat/stream")
def chat_stream():
    """SSE endpoint — streams generative replies token-by-token.

    Emits {"token": "..."} events, then a final
    {"done": true, "response", "intent", "confidence", "entities"} event.
    Non-generative replies arrive whole in "done".
    """
    message = (request.args.get("message") or "").strip()
    user_id = request.args.get("user_id", "web_user")
    if not message:
        return jsonify({"error": "Empty message"}), 400
    if chatbot is None or not chatbot.is_ready:
        return jsonify({"error": "Chatbot not initialized"}), 503

    def events():
        q: "queue.Queue" = queue.Queue()
        result = {}
        done = object()

        def run():
            try:
                result["r"] = chatbot.handle_message(
                    message, user_id, on_token=lambda p: q.put(p)
                )
            except Exception as e:
                result["error"] = str(e)
            finally:
                q.put(done)

        threading.Thread(target=run, daemon=True).start()
        while True:
            item = q.get()
            if item is done:
                break
            yield f"data: {json.dumps({'token': item})}\n\n"

        if "error" in result:
            yield f"data: {json.dumps({'error': result['error']})}\n\n"
        else:
            r = result["r"]
            yield f"data: {json.dumps({'done': True, 'response': r.text, 'intent': r.intent, 'confidence': round(r.confidence, 3), 'entities': r.entities})}\n\n"

    return Response(stream_with_context(events()), mimetype="text/event-stream")


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get("user_id", "web_user")
    if chatbot is None:
        return jsonify({"error": "Chatbot not initialized"}), 503
    text = chatbot.reset_context(user_id)
    return jsonify({"response": text})


@app.route("/status")
def status():
    if chatbot is None:
        return jsonify({"ready": False})
    return jsonify({
        "ready": chatbot.is_ready,
        "generative": chatbot.generative_ready,
        "ai_name": chatbot.ai_name,
        "user_name": chatbot.user_profile.get("name"),
        "intents": len(chatbot.intents),
    })


@app.route("/analytics")
def analytics():
    if chatbot is None:
        return jsonify({"error": "Chatbot not initialized"}), 503
    return jsonify(chatbot.get_analytics())


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    if initialize_chatbot():
        app.run(debug=False, host="0.0.0.0", port=5000)
    else:
        print("Chatbot failed to initialize; exiting.")
