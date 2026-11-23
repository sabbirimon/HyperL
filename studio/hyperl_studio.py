# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Local desktop front end for the native HyperL engine. Requires Python 3.9+ and Tk."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import filedialog, ttk


def engine_path():
    name = "hyperl-engine.exe" if os.name == "nt" else "hyperl-engine"
    here = Path(__file__).resolve().parent
    for directory in (here, here.parent / "engine" / "target" / "release"):
        if (directory / name).is_file():
            return str(directory / name)
    return shutil.which(name) or ""


def preferences_path():
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA") or Path.home() / "AppData/Roaming")
    elif sys.platform == "darwin":
        root = Path.home() / "Library/Application Support"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "HyperL" / "studio.json"


def read_settings(path):
    defaults = {"budget": "512", "device": "auto"}
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(settings, dict):
            defaults.update({k: settings[k] for k in defaults if isinstance(settings.get(k), str)})
    except (OSError, ValueError):
        pass
    return defaults


def save_settings(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8") as tmp:
        name = tmp.name
        try:
            json.dump({k: values[k] for k in ("budget", "device")}, tmp)
            tmp.flush()
        except BaseException:
            tmp.close()
            os.unlink(name)
            raise
    try:
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def command(values, action):
    exe = Path(values["engine"])
    if not exe.is_file():
        raise ValueError("Choose the HyperL engine executable.")
    if action == "processors":
        return [str(exe.resolve()), action]
    if action not in ("plan", "project"):
        raise ValueError("Unknown operation.")
    if not Path(values["model"]).is_file():
        raise ValueError("Choose an immutable SafeTensors file.")
    if not re.fullmatch(r"[A-Fa-f0-9]{64}", values["digest"]):
        raise ValueError("Enter the expected 64-character SHA-256.")
    if not all(values[k] for k in ("input", "weight")):
        raise ValueError("Enter both tensor names.")
    budget = int(values["budget"])
    if budget <= 0:
        raise ValueError("Enter a positive memory budget in MiB.")
    device = values["device"]
    if action == "plan" and device.startswith("gpu:"):
        raise ValueError("GPU admission runs on the selected adapter during projection.")
    args = [str(exe.resolve()), action, values["model"], values["digest"], values["input"],
            values["weight"], str(budget * 1048576), device]
    if action == "project":
        if not values["output"] or Path(values["output"]).exists():
            raise ValueError("Choose a new output file.")
        args.append(values["output"])
    if values["scale"]:
        args.extend(["--int4-scale", values["scale"]])
    return args


class Studio:
    def __init__(self, root):
        self.root = root
        root.title("HyperL Studio")
        pane = ttk.Frame(root, padding=18)
        pane.pack(fill="both", expand=True)
        pane.columnconfigure(1, weight=1)
        self.settings = preferences_path()
        data = dict(engine=engine_path(), model="", digest="", input="x", weight="w",
                    output="", scale="", **read_settings(self.settings))
        self.fields = {k: tk.StringVar(value=v) for k, v in data.items()}
        ttk.Label(pane, text="Tensor projection", font=("TkDefaultFont", 18)).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
        rows = [("engine", "Engine executable"), ("model", "Immutable SafeTensors file"),
                ("digest", "Expected SHA-256"), ("input", "Input tensor"), ("weight", "Weight tensor"),
                ("device", "Processor or worker count"), ("budget", "Tensor budget (MiB)"),
                ("scale", "Packed INT4 scale (optional)"), ("output", "New output file")]
        for row, (key, label) in enumerate(rows, 1):
            ttk.Label(pane, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12))
            if key == "device":
                self.devices = ttk.Combobox(pane, textvariable=self.fields[key], values=["auto"])
                widget = self.devices
            else:
                widget = ttk.Entry(pane, textvariable=self.fields[key], width=48)
            widget.grid(row=row, column=1, sticky="ew", pady=3)
            if key in ("engine", "model", "output"):
                ttk.Button(pane, text="Browse…", command=lambda k=key: self.browse(k)).grid(row=row, column=2)
        tools = ttk.Frame(pane)
        tools.grid(row=10, columnspan=3, sticky="w", pady=12)
        self.actions = []
        for action, label in [("processors", "Find processors"), ("plan", "Preview plan"), ("project", "Run projection")]:
            button = ttk.Button(tools, text=label, command=lambda a=action: self.start(a))
            button.pack(side="left", padx=(0, 4))
            self.actions.append(button)
        self.stop = ttk.Button(tools, text="Cancel", command=self.cancel, state="disabled")
        self.stop.pack(side="left")
        self.status = tk.StringVar(value="Choose a model and enter its verified digest.")
        ttk.Label(pane, textvariable=self.status, wraplength=580, justify="left").grid(
            row=11, columnspan=3, sticky="w")
        self.process = None
        self.closing = False
        root.protocol("WM_DELETE_WINDOW", self.close)

    def browse(self, key):
        path = filedialog.asksaveasfilename(defaultextension=".safetensors") if key == "output" else filedialog.askopenfilename()
        if path:
            self.fields[key].set(path)

    def start(self, action):
        if self.process is not None:
            return
        values = {k: v.get() for k, v in self.fields.items()}
        try:
            args = command(values, action)
            if action != "processors":
                save_settings(self.settings, values)
            # File-backed output keeps device discovery from blocking on full pipes.
            self.log = tempfile.TemporaryFile()
            try:
                self.process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=self.log,
                                                stderr=subprocess.STDOUT, creationflags=(
                                                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0))
            except OSError:
                self.log.close()
                raise
        except (ValueError, OSError) as err:
            self.status.set(str(err))
            return
        self.status.set({"processors": "Discovering processors…", "plan": "Verifying model and planning…",
                         "project": "Running projection…"}[action])
        for button in self.actions:
            button.state(["disabled"])
        self.stop.state(["!disabled"])
        self.poll(action)

    def poll(self, action):
        if self.process.poll() is None:
            self.root.after(80, self.poll, action)
            return
        code = self.process.returncode
        self.process = None
        self.log.seek(0)
        result = self.log.read(32768).decode("utf-8", errors="replace").strip()
        self.log.close()
        if self.closing:
            self.root.destroy()
            return
        if code == 0 and action == "processors":
            self.devices["values"] = ["auto"] + [s for s in result.splitlines() if s.startswith("gpu:")]
        self.status.set(result or ("Finished" if code == 0 else f"Engine exited with code {code}"))
        self.stop.state(["disabled"])
        for button in self.actions:
            button.state(["!disabled"])

    def cancel(self):
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            except OSError:
                process.terminate()
            self.root.after(3000, lambda: process.kill() if process.poll() is None else None)

    def close(self):
        if self.process is None:
            self.root.destroy()
        else:
            self.closing = True
            self.cancel()


if __name__ == "__main__":
    window = tk.Tk()
    Studio(window)
    window.mainloop()
