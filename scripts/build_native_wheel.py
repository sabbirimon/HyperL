#!/usr/bin/env python3
"""Build a local platform wheel from an explicitly selected, tested CPU library.

Requires the pinned wheel/setuptools build tools. No downloads or index publication.
Linux output is a runner-local linux wheel, never an unaudited manylinux claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from hyperl import NativeCpu, Program, Step, f32
from hyperl.bundled import machine_name
from hyperl.contract import RUNTIME_REVISION


def platform_tag():
    if sys.platform == 'darwin':
        # Python built with old SDKs can report 10.16 for modern macOS. Select
        # the actual build-host major version, conservatively not an older OS.
        version = subprocess.check_output(['sw_vers', '-productVersion'], text=True,
                                         env={**os.environ, 'SYSTEM_VERSION_COMPAT': '0'}).strip().split('.')
        major, minor = map(int, version[:2])
        return f'macosx_{major}_{0 if major >= 11 else minor}_{machine_name(platform.machine())}'
    return sysconfig.get_platform().replace('-', '_').replace('.', '_')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names = {'darwin': 'libhyperl_cpu_runtime.dylib', 'linux': 'libhyperl_cpu_runtime.so',
             'win32': 'hyperl_cpu_runtime.dll'}
    library = args.library
    if (not library.is_absolute() or not library.is_file() or library.is_symlink() or
            library.name != names.get(sys.platform) or not 1 <= library.stat().st_size <= 32 * 1024 * 1024):
        parser.error('choose the absolute reviewed CPU library for this build host')
    subprocess.run([sys.executable, str(ROOT / 'scripts/generate_contract.py'), '--check'], check=True)
    # Owner-invoked execution checks the actual selected library before bundling.
    cpu = NativeCpu(library)
    program = Program(('x',), (Step('y', 'relu', ('x',)),), 'y')
    assert list(cpu.execute(program, {'x': f32([-1, 2, 3])})) == [0, 2, 3]
    assert list(cpu.precise_sum(f32([16777216, 1, -16777216]))) == [1]
    from wheel.wheelfile import WheelFile
    from wheel import __version__ as wheel_version
    if wheel_version != '0.45.1':
        parser.error('use reviewed wheel==0.45.1')
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hyperl-wheel-') as folder:
        temp = Path(folder)
        source = temp / 'python'
        shutil.copytree(ROOT / 'python', source,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info', 'build', '_native'))
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-index', '--no-build-isolation',
                        '--no-deps', '--wheel-dir', str(temp / 'pure'), str(source)], check=True)
        pure, = (temp / 'pure').glob('*.whl')
        tag = platform_tag()
        output = args.output / pure.name.replace('py3-none-any.whl', f'py3-none-{tag}.whl')
        if output.exists():
            parser.error('wheel destination exists; select a new output directory')
        record = {
            'format': 'hyperl-native-bundle/1', 'system': sys.platform,
            'machine': machine_name(platform.machine()), 'platformTag': tag,
            'library': library.name, 'sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
            'abiVersion': 1, 'runtimeRevision': RUNTIME_REVISION,
            'sourceSha256': {str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                             for path in map(Path, ['native/cpu.c', 'native/include/hyperl.h',
                                                   'native/include/hyperl_contract.h', 'contracts/hyperl-1.json'])},
        }
        staged = temp / output.name
        with zipfile.ZipFile(pure) as source_wheel, WheelFile(staged, 'w') as wheel:
            for name in source_wheel.namelist():
                if name.endswith('/RECORD'):
                    continue
                data = source_wheel.read(name)
                if name.endswith('/WHEEL'):
                    data = (f'Wheel-Version: 1.0\nGenerator: HyperL native wheel builder/1\n'
                            f'Root-Is-Purelib: false\nTag: py3-none-{tag}\n').encode('utf-8')
                wheel.writestr(name, data)
            wheel.writestr('hyperl/_native/' + library.name, library.read_bytes())
            wheel.writestr('hyperl/_native/manifest.json', json.dumps(record, indent=2) + '\n')
        # Atomic no-overwrite publication in the selected output directory.
        private = args.output / ('.' + output.name + '.tmp')
        try:
            with private.open('xb') as file:
                file.write(staged.read_bytes())
            os.link(private, output)
        finally:
            private.unlink(missing_ok=True)
        print(json.dumps({'wheel': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                          'platform': tag, 'uploaded': False}))


if __name__ == '__main__':
    main()
