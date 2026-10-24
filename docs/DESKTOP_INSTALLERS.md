# macOS, Windows and Linux desktop installers

The alpha.4 release contains portable ZIP/TAR archives and requires Java 17+.
The alpha.5 packaging increment adds native **unsigned preview** installers with
a bundled Java 21 runtime. Download assets matching your OS **and architecture**
from [Releases](https://github.com/sabbirimon/HyperL/releases). A source ZIP is not
an installer. An installer build does not establish production certification.

| Target | Asset ending | Installation | CLI location |
|---|---|---|---|
| Intel Mac, including A1990 | `darwin-x86_64.dmg` | Mount; copy HyperL.app to Applications or `~/Applications` | `HyperL.app/Contents/MacOS/hyperl-cli` |
| Apple Silicon Mac | `darwin-arm64.dmg` | Same procedure, use the ARM64 asset | Same relative path |
| Windows x64 | `windows-amd64.msi` | Run MSI; per-user installation and Start Menu entry | `hyperl-cli.exe` inside the selected install directory |
| Debian/Ubuntu x64 | `linux-x86_64.deb` | Package manager, normally `sudo apt install ./FILE.deb` | `/opt/hyperl/bin/hyperl-cli` |
| RPM-based Linux x64 | `linux-x86_64.rpm` | Compatible package manager, normally `sudo dnf install ./FILE.rpm` | `/opt/hyperl/bin/hyperl-cli` |

The main **HyperL** launcher opens the graphical workbench. The lowercase
**hyperl-cli** launcher accepts CLI commands. Neither adds a cloud account, model,
GPU driver, background service or global PATH entry. The native CPU library is
included; macOS includes the optional Metal bridge. GPU use still requires the
explicit bridge path, exact device selection and correctness checks. The phone
app is separate: these packages cannot install on Android or iOS.

## Verify, launch and remove

Compare your asset's SHA-256 with the release checksum before installation:

```sh
# macOS
shasum -a 256 hyperl-0.1.0-alpha.5-darwin-x86_64.dmg
"$HOME/Applications/HyperL.app/Contents/MacOS/hyperl-cli" capabilities
"$HOME/Applications/HyperL.app/Contents/MacOS/hyperl-cli" library

# Linux (example for the x64 DEB)
sha256sum hyperl-0.1.0-alpha.5-linux-x86_64.deb
sudo apt install ./hyperl-0.1.0-alpha.5-linux-x86_64.deb
/opt/hyperl/bin/hyperl-cli capabilities
```

```powershell
Get-FileHash .\hyperl-0.1.0-alpha.5-windows-amd64.msi -Algorithm SHA256
# After choosing your installation directory in the MSI:
& 'C:\YOUR_SELECTED_DIRECTORY\hyperl-cli.exe' capabilities
```

These previews lack Apple Developer ID/notarization and Windows Authenticode
signatures. OS trust prompts may prevent normal launch. Review the exact source,
artifact and publisher before approving it; do not disable system security. Use
the verified portable archive with your maintained Java installation if you
prefer to wait for signed installers. Checksums do not authenticate the publisher.

Close HyperL before uninstalling. On macOS, remove only the installed HyperL.app;
on Windows use Settings → Apps → HyperL → Uninstall. On Linux use
`sudo apt remove hyperl` or `sudo dnf remove hyperl`. Keep workspaces, data and
keys outside installation directories. Native installer upgrade/rollback and
interactive install/uninstall remain acceptance gates until separately recorded.

## Build the installers yourself

Use JDK 21 and build **on the target OS/architecture**. macOS cannot cross-package
a Windows MSI. Linux packaging needs `fakeroot`/`dpkg-deb` for DEB and `rpmbuild`
for RPM; Windows packaging needs WiX 3. These tools are build dependencies, not
end-user requirements. See the [JDK packaging overview](https://docs.oracle.com/en/java/javase/21/jpackage/packaging-overview.html)
and [jpackage options](https://docs.oracle.com/en/java/javase/21/docs/specs/man/jpackage.html).

```sh
./gradlew --no-daemon test installDist distZip distTar
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
ctest --test-dir build/native -C Release --output-on-failure
python3 scripts/package_desktop.py --distribution build/install/hyperl \
  --native build/native --output build/desktop --jdk "$JAVA_HOME"
```

Windows: use `gradlew.bat`, `python`, and `--jdk "$env:JAVA_HOME"` in PowerShell.
Choose a **new** output directory for every build. The script never downloads
dependencies, installs drivers, modifies an existing app or signs with user keys.
`--types dmg`, `--types msi` or `--types deb` selects a supported native format.
The default builds all supported formats for that host. CI uses explicit GitHub
[runner labels](https://github.com/actions/runner-images) for both Mac architectures,
Windows x64 and Ubuntu 22.04 x64. Linux binaries require a compatible glibc and
desktop graphics stack; the RPM artifact is not evidence for every RPM distro.

Each package contains the application/dependency JARs, examples, docs, license
notices, Python preview source, C SDK source and host-built CPU library. The
bundled Java runtime retains its `legal/` notices. Native installer metadata uses
`1.1.5` for alpha.5 because native formats do not express SemVer prereleases and
macOS requires a positive first integer. This independent metadata maps the
application major plus one, minor, and alpha sequence; it is not a 1.0 release.
the CLI and release retain `0.1.0-alpha.5`. Preview identity is explicit.

Before packaging, the script executes `capabilities`, checks all twelve library
entries and runs weighted ReLU `[0,6,12]` **without JAVA_HOME or a PATH Java**.
`packaging-report.json` records host, architecture, JDK, hashes, smoke results and
unrun install/GPU gates. Review this alongside CI. A production release also needs
signed provenance, supported OS baselines, dependency review and real lifecycle
tests described in [the production plan](PRODUCTION_AND_MESHLIT_PLAN.md).
