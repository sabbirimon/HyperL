# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Offline text generation for the self-contained desktop distribution."""
import json
import os
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk

# The desktop build includes the same standalone adapter used by the fleet.
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python/hyperl"))
from llm import LlamaSession, LlamaSettings

MODEL = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_SHA = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"


def bundle_dir():
    return Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent


def generate(directory, prompt, cancelled=lambda: False):
    if not prompt.strip() or len(prompt.encode("utf-8")) > 8192:
        raise ValueError("Enter a prompt of up to 8192 UTF-8 bytes.")
    server = directory / ("llama-server.exe" if os.name == "nt" else "llama-server")
    manifest = json.loads((directory / "local-runtime.json").read_text())
    settings = LlamaSettings(str(server), manifest["server_sha256"], str(directory / MODEL),
                             MODEL_SHA, context=2048, threads=min(4, os.cpu_count() or 1))
    session = LlamaSession(settings, cancelled=cancelled)
    try:
        content = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return session.complete(content, max_tokens=96, cancelled=cancelled)
    finally:
        session.close()


def local_window(parent):
    window = tk.Toplevel(parent)
    window.title("HyperL · Local inference")
    body = ttk.Frame(window, padding=14)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="Qwen2.5 0.5B · Local CPU").pack(anchor="w")
    prompt = tk.Text(body, height=4, width=64, wrap="word")
    prompt.pack(fill="x", pady=8)
    result = tk.StringVar()
    ttk.Label(body, textvariable=result, wraplength=510).pack(fill="x")
    events = queue.Queue(maxsize=1)
    stopped = threading.Event()
    running = False
    closing = False

    def poll():
        nonlocal running
        try:
            message = events.get_nowait()
        except queue.Empty:
            window.after(100, poll)
            return
        running = False
        if closing:
            window.destroy()
            return
        result.set(message)
        run.state(["!disabled"])
        stop.state(["disabled"])

    def start():
        nonlocal running
        if running:
            return
        text = prompt.get("1.0", "end-1c")
        stopped.clear()
        running = True
        def execute():
            try:
                message = generate(bundle_dir(), text, stopped.is_set)["text"]
            except Exception as error:
                message = str(error)
            events.put(message)
        run.state(["disabled"])
        stop.state(["!disabled"])
        result.set("Verifying model and running locally…")
        threading.Thread(target=execute).start()
        window.after(100, poll)

    def close():
        nonlocal closing
        closing = True
        stopped.set()
        if not running:
            window.destroy()

    run = ttk.Button(body, text="Generate", command=start)
    run.pack(side="left")
    stop = ttk.Button(body, text="Stop", command=stopped.set, state="disabled")
    stop.pack(side="left")
    window.protocol("WM_DELETE_WINDOW", close)
    window.bind("<Destroy>", lambda event: stopped.set() if event.widget == window else None)
    return window
