"""Explicit opt-in to the package's native CPU binary; no discovery or download."""
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from .contract import RUNTIME_REVISION


def machine_name(value):
    return {'amd64': 'x86_64', 'aarch64': 'arm64'}.get(value.lower(), value.lower())


def bundled_native_path(directory=None):
    root = Path(directory) if directory is not None else Path(__file__).parent / '_native'
    manifest = root / 'manifest.json'
    if not manifest.is_file() or manifest.is_symlink() or not 1 <= manifest.stat().st_size <= 16384:
        raise RuntimeError('No valid bundled CPU binary; install a qualified platform wheel or supply an explicit library')
    record = json.loads(manifest.read_text(encoding='utf-8'))
    fields = {'format', 'system', 'machine', 'platformTag', 'library', 'sha256',
              'abiVersion', 'runtimeRevision', 'sourceSha256'}
    if (not isinstance(record, dict) or set(record) != fields or
            record['format'] != 'hyperl-native-bundle/1' or
            type(record['abiVersion']) is not int or record['abiVersion'] != 1 or
            record['runtimeRevision'] != RUNTIME_REVISION or
            not isinstance(record['platformTag'], str) or
            not isinstance(record['sourceSha256'], dict) or
            not all(isinstance(name, str) and isinstance(value, str) and
                    re.fullmatch('[a-f0-9]{64}', value) for name, value in record['sourceSha256'].items())):
        raise RuntimeError('Invalid native bundle metadata')
    if record['system'] != sys.platform or record['machine'] != machine_name(platform.machine()):
        raise RuntimeError('Bundled CPU target does not match this interpreter process')
    names = {'darwin': 'libhyperl_cpu_runtime.dylib', 'linux': 'libhyperl_cpu_runtime.so',
             'win32': 'hyperl_cpu_runtime.dll'}
    if record['library'] != names.get(sys.platform):
        raise RuntimeError('Unexpected bundled CPU library filename')
    library = root / record['library']
    if not library.is_file() or library.is_symlink() or not 1 <= library.stat().st_size <= 32 * 1024 * 1024:
        raise RuntimeError('Missing or oversized bundled CPU library')
    if not isinstance(record['sha256'], str) or not re.fullmatch('[a-f0-9]{64}', record['sha256']):
        raise RuntimeError('Invalid native digest')
    digest = hashlib.sha256()
    with library.open('rb') as input_file:
        for data in iter(lambda: input_file.read(65536), b''):
            digest.update(data)
    if digest.hexdigest() != record['sha256']:
        raise RuntimeError('Bundled CPU digest mismatch; no native code loaded')
    return library.resolve()
