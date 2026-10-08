#!/usr/bin/env python3
"""Build self-contained, unsigned preview installers on their target OS. No downloads."""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HOST_TYPES = {"Darwin": ("dmg",), "Windows": ("msi",), "Linux": ("deb", "rpm")}
UPGRADE_UUID = "a5a487a3-b534-4ed1-9b2c-7b163ef15ea3"


def run(command, *, env=None, timeout=600):
    try:
        return subprocess.run([str(x) for x in command], check=True, capture_output=True,
                              text=True, env=env, timeout=timeout).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{Path(command[0]).name} failed ({exc.returncode}):\n{(exc.stdout + exc.stderr)[-8192:]}") from exc


def version_from(root):
    match = re.search(r'^version = "(\d+\.\d+\.\d+-alpha\.(\d+))"',
                      (root / "build.gradle.kts").read_text(), re.M)
    if not match:
        raise ValueError("Expected an explicit alpha release version")
    # Native installer metadata cannot express SemVer prereleases. Until 1.0,
    # encode the alpha sequence in its patch field; the payload keeps full SemVer.
    major, minor, _ = match[1].split("-", 1)[0].split(".")
    # macOS requires the first integer to be positive. Installer metadata has a
    # separate epoch (language/application major + 1), not a production milestone.
    return match[1], f"{int(major) + 1}.{minor}.{match[2]}"


def launcher(image, host):
    return {"Darwin": image / "Contents/MacOS/hyperl-cli",
            "Windows": image / "hyperl-cli.exe", "Linux": image / "bin/hyperl-cli"}[host]


def build(root, distribution, native, destination, jdk, types):
    host = platform.system()
    if host not in HOST_TYPES or not types or any(t not in HOST_TYPES[host] for t in types):
        raise ValueError(f"Build only native installer types for {host}: {HOST_TYPES.get(host)}")
    if destination.exists():
        raise ValueError("Choose a new output directory; existing packages are never replaced")
    tool = jdk / "bin" / ("jpackage.exe" if host == "Windows" else "jpackage")
    if not tool.is_file():
        raise ValueError("An explicit JDK 21 with jpackage is required")
    jdk_version = run([jdk / "bin" / ("java.exe" if host == "Windows" else "java"), "--version"])
    if not re.search(r'\b21[. ]', jdk_version):
        raise ValueError("Desktop packaging is qualified with JDK 21 only")
    version, installer_version = version_from(root)
    jars = sorted((distribution / "lib").glob("*.jar"))
    main = [p for p in jars if p.name == f"hyperl-{version}.jar"]
    if len(main) != 1:
        raise ValueError("Run Gradle installDist for the current version first")
    libraries = [p for p in native.rglob("*hyperl_cpu_runtime*")
                 if p.is_file() and p.suffix in (".dll", ".dylib", ".so")]
    if len(libraries) != 1:
        raise ValueError("Expected exactly one built native CPU shared library")
    if host == "Linux":
        for tool_name in ("fakeroot", "dpkg-deb") if "deb" in types else ():
            if not shutil.which(tool_name):
                raise ValueError(f"Missing build tool {tool_name}; install it in your build environment")
        if "rpm" in types and not shutil.which("rpmbuild"):
            raise ValueError("Missing build tool rpmbuild")
    destination.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="hyperl-package-") as temporary:
        stage = Path(temporary)
        payload = stage / "input"
        payload.mkdir()
        for jar in jars:
            shutil.copy2(jar, payload / jar.name)
        for name in ("README.md", "LICENSE", "NOTICE", "docs", "examples", "python", "sdk"):
            source = distribution / name
            if source.is_dir():
                shutil.copytree(source, payload / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            elif source.is_file():
                shutil.copy2(source, payload / name)
        (payload / "native").mkdir()
        shutil.copy2(libraries[0], payload / "native" / libraries[0].name)
        if host == "Darwin":
            bridge = native / "hyperl-metal"
            if not bridge.is_file():
                raise ValueError("Build the macOS Metal bridge before packaging")
            shutil.copy2(bridge, payload / "native/hyperl-metal")
        properties = stage / "cli.properties"
        properties.write_text("arguments=\nwin-console=true\nwin-menu=false\nwin-shortcut=false\n")
        image_dir = destination / "app-image"
        command = [tool, "--type", "app-image", "--dest", image_dir,
                   "--input", payload, "--name", "HyperL", "--main-jar", main[0].name,
                   "--main-class", "org.hyperl.MainKt", "--arguments", "gui",
                   "--app-version", installer_version, "--vendor", "HyperL contributors",
                   "--description", f"HyperL {version} experimental developer workbench",
                   "--java-options", "-Xmx512m", "--add-launcher", f"hyperl-cli={properties}"]
        if host == "Darwin":
            command += ["--mac-package-identifier", "org.hyperl.workbench", "--mac-package-name", "HyperL"]
        run(command)
        image = image_dir / ("HyperL.app" if host == "Darwin" else "HyperL")
        cli = launcher(image, host)
        environment = os.environ.copy()
        environment.pop("JAVA_HOME", None)
        # Prove the native launcher uses its bundled runtime, without a PATH Java.
        environment["PATH"] = ""
        capabilities = json.loads(run([cli, "capabilities"], env=environment, timeout=30))
        if capabilities["version"] != version:
            raise ValueError("Packaged CLI version mismatch")
        inventory = run([cli, "library"], env=environment, timeout=30)
        if len(inventory.strip().splitlines()) != 12:
            raise ValueError("Packaged ready-made library inventory mismatch")
        fixture = stage / "inputs.json"
        fixture.write_text('{"x":[-1,2,3],"w":[2,3,4]}')
        result = json.loads(run([cli, "recipe-run", "weighted_relu", fixture], env=environment, timeout=30))
        if result != [0.0, 6.0, 12.0]:
            raise ValueError("Packaged CPU arithmetic failed")
        (destination / "bundled-runtime-smoke.json").write_text(json.dumps({
            "format": "hyperl-desktop-smoke/1", "version": version, "host": host,
            "architecture": platform.machine(), "jdk": jdk_version.strip(),
            "externalJava": False, "libraryRecipes": 12, "cpuResult": result,
            "status": "passed-bundled-runtime", "installerBuild": "pending",
            "gpuExecution": "not-run", "interactiveGui": "not-run"}, indent=2) + "\n")
        artifacts = []
        for package_type in types:
            out = stage / package_type
            options = [tool, "--type", package_type, "--app-image", image, "--dest", out,
                       "--name", "HyperL", "--app-version", installer_version,
                       "--vendor", "HyperL contributors", "--license-file", root / "LICENSE",
                       "--description", f"HyperL {version} unsigned experimental preview",
                       "--about-url", "https://github.com/sabbirimon/HyperL"]
            if host == "Windows":
                options += ["--win-per-user-install", "--win-dir-chooser", "--win-menu",
                            "--win-menu-group", "HyperL", "--win-upgrade-uuid", UPGRADE_UUID]
            elif host == "Linux":
                options += ["--linux-package-name", "hyperl", "--linux-shortcut", "--linux-menu-group", "Development",
                            "--linux-app-release", f"alpha{version.rsplit('.', 1)[1]}"]
                if package_type == "rpm":
                    options += ["--linux-rpm-license-type", "Apache-2.0"]
                else:
                    options += ["--linux-deb-maintainer", "30753145+sabbirimon@users.noreply.github.com"]
            run(options)
            produced = list(out.glob(f"*.{package_type}"))
            if len(produced) != 1:
                raise ValueError(f"Expected exactly one {package_type} artifact")
            filename = f"hyperl-{version}-{host.lower()}-{platform.machine().lower()}.{package_type}"
            target = destination / filename
            shutil.copy2(produced[0], target)
            artifacts.append({"file": filename, "bytes": target.stat().st_size,
                              "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        report = {"format": "hyperl-desktop-package/1", "version": version, "installerVersion": installer_version,
                  "host": host, "architecture": platform.machine(), "jdk": jdk_version.strip(),
                  "signed": False, "notarized": False, "productionQualified": False,
                  "bundledRuntimeCpuSmoke": "passed", "cpuExample": result,
                  "interactiveInstallUninstall": "not-run", "gpuExecution": "not-run",
                  "artifacts": artifacts}
        (destination / "packaging-report.json").write_text(json.dumps(report, indent=2) + "\n")
        (destination / "SHA256SUMS").write_text("".join(f"{a['sha256']}  {a['file']}\n" for a in artifacts))
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jdk", type=Path, required=True)
    parser.add_argument("--types", help="Comma separated native package types; defaults to host types")
    args = parser.parse_args()
    requested = tuple(args.types.split(",")) if args.types else HOST_TYPES.get(platform.system(), ())
    print(json.dumps(build(args.root.resolve(), args.distribution.resolve(), args.native.resolve(),
                           args.output.resolve(), args.jdk.resolve(), requested), indent=2))
