"""Assemble and verify the desktop archive from pinned local build inputs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studio"))
from local_inference import MODEL, MODEL_SHA
from llm import LLAMA_REVISION, file_digest


def main():
    source = ROOT / "build/llama-src"
    if subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip() != LLAMA_REVISION:
        raise ValueError("Unexpected llama.cpp revision")
    native = ROOT / "build/desktop-llama"
    subprocess.run(["cmake", "-S", source, "-B", native, "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_SHARED_LIBS=OFF", "-DGGML_NATIVE=OFF", "-DGGML_METAL=OFF", "-DGGML_OPENMP=OFF",
                    "-DGGML_AVX=OFF", "-DGGML_AVX2=OFF", "-DGGML_FMA=OFF", "-DGGML_F16C=OFF", "-DGGML_BMI2=OFF",
                    "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_APP=OFF",
                    "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=ON"], check=True)
    subprocess.run(["cmake", "--build", native, "--config", "Release", "--target", "llama-server", "--parallel", "4"], check=True)
    suffix = ".exe" if os.name == "nt" else ""
    [server] = (native / "bin").rglob("llama-server" + suffix)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--clean", "--onefile", "--name", "HyperL-Studio",
                    "--paths", str(ROOT / "python/hyperl"), "--exclude-module", "numpy",
                    "studio/hyperl_studio.py"], check=True)
    with tempfile.TemporaryDirectory() as temp:
        stage = Path(temp)
        model = stage / MODEL
        url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/"
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--max-time", "1800",
                        "--max-filesize", "491400032", "--output", model, url + MODEL], check=True)
        if model.stat().st_size != 491400032 or file_digest(model) != MODEL_SHA:
            raise ValueError("Unexpected starter model")
        subprocess.run(["curl", "--fail", "--location", "--output", stage / "Qwen-LICENSE", url + "LICENSE"], check=True)
        files = [ROOT / "dist" / ("HyperL-Studio" + suffix), ROOT / "engine/target/release" / ("hyperl-engine" + suffix),
                 server, ROOT / "LICENSE", ROOT / "NOTICE", ROOT / "studio/START.txt", ROOT / "studio/hyperl_studio.py"]
        for file in files:
            shutil.copy2(file, stage / file.name)
        shutil.copy2(source / "LICENSE", stage / "llama.cpp-LICENSE")
        (stage / "local-runtime.json").write_text(json.dumps({"revision": LLAMA_REVISION, "server_sha256": file_digest(server)}))
        archive = ROOT / "hyperl-desktop.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as out:
            for file in stage.iterdir():
                out.write(file, file.name)
        with tempfile.TemporaryDirectory() as target, zipfile.ZipFile(archive) as package:
            package.extractall(target)
            for name in ["HyperL-Studio", "llama-server", "hyperl-engine"]:
                item = Path(target) / (name + suffix)
                item.chmod(item.stat().st_mode | stat.S_IXUSR)
            subprocess.run([str(Path(target) / ("HyperL-Studio" + suffix)), "--local-check"], check=True, timeout=300)


if __name__ == "__main__":
    main()
