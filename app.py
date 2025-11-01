import os
import threading
import gradio as gr
from server import app, socketio

# Start Flask in a background thread
def run_flask():
    socketio.run(app, host="0.0.0.0", port=7860)

thread = threading.Thread(target=run_flask)
thread.start()

# Simple Gradio frontend just to embed the camera feed
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 AI-Powered ProctorVision")
    gr.HTML('<iframe src="http://localhost:7860/student" width="100%" height="600px"></iframe>')

demo.launch(server_name="0.0.0.0", server_port=7861)
