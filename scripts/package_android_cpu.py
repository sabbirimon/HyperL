"""Build an unsigned, aligned Android native CPU diagnostics APK (API 26+)."""
import argparse
from pathlib import Path
import platform
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdk", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sdk = args.sdk.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error("choose a new output file")
    output.parent.mkdir(parents=True, exist_ok=True)
    ndk = sdk / "ndk/28.2.13676358"
    tools = sdk / "build-tools/37.0.0"
    host = {"Linux": "linux-x86_64", "Darwin": "darwin-x86_64"}[platform.system()]
    app = ROOT / "android/native-studio"
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        work = Path(temp)
        apk = work / "package.apk"
        subprocess.run([tools/"aapt2","link","-I",sdk/"platforms/android-36/android.jar",
                        "--manifest",app/"AndroidManifest.xml","-o",apk],check=True)
        with zipfile.ZipFile(apk,"a") as archive:
            for abi, prefix in [("arm64-v8a","aarch64"),("x86_64","x86_64")]:
                lib = work / "libhyperl-mobile.so"
                cc = ndk/f"toolchains/llvm/prebuilt/{host}/bin/{prefix}-linux-android26-clang"
                subprocess.run([cc,"-std=c11","-D_GNU_SOURCE","-O2","-ffp-contract=off","-fPIC","-shared",
                    "-I",ROOT/"native/include",app/"activity.c",ROOT/"native/engine.c",ROOT/"native/kernels.c",
                    "-Wl,-z,max-page-size=16384","-landroid","-llog","-lm","-o",lib],check=True)
                archive.write(lib,f"lib/{abi}/{lib.name}",compress_type=zipfile.ZIP_STORED)
            for name in ("NOTICE","LICENSE"):
                archive.write(ROOT/name,"assets/"+name)
        subprocess.run([tools/"zipalign","-P","16","4",apk,work/"aligned.apk"], check=True)
        (work/"aligned.apk").rename(output)


if __name__=="__main__":
    main()
