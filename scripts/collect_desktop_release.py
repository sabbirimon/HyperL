#!/usr/bin/env python3
"""Collect verified artifacts from the four target jobs; never download or publish."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def collect(root, inputs, output, version, run):
    assert re.fullmatch(r'\d+\.\d+\.\d+-alpha\.\d+', version)
    assert f'version = "{version}"' in (root / 'build.gradle.kts').read_text()
    folders = list(inputs.glob('desktop-*'))
    assert {p.name for p in folders} == {'desktop-macos-15','desktop-macos-15-intel','desktop-windows-2022','desktop-ubuntu-22.04'}
    output.mkdir()
    reports, names = [], set()
    def copy(path):
        assert path.is_file() and path.name not in names
        names.add(path.name); shutil.copy2(path, output / path.name)
    for folder in folders:
        payload = folder / 'desktop'
        report = json.loads((payload / 'packaging-report.json').read_text())
        assert report['format'] == 'hyperl-desktop-package/1' and report['version'] == version
        assert report['bundledRuntimeCpuSmoke'] == 'passed' and report['cpuExample'] == [0,6,12]
        assert report['signed'] is False and report['productionQualified'] is False
        for asset in report['artifacts']:
            name = asset['file']
            assert Path(name).name == name and name.startswith(f'hyperl-{version}-')
            path = payload / name
            assert sha(path) == asset['sha256'] and path.stat().st_size == asset['bytes']
            copy(path)
        reports.append(report)
    assert len(names) == 5
    assert {p.suffix for p in output.iterdir()} == {'.dmg','.msi','.deb','.rpm'}
    for suffix in ('zip','tar'):
        copy(inputs / 'desktop-ubuntu-22.04/distributions' / f'hyperl-{version}.{suffix}')
    manifest = {'format':'hyperl-preview-release/1','version':version,'run':str(run),
                'reports':reports,'productionQualified':False}
    (output / 'desktop-validation.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for name in ('DESKTOP_INSTALLERS.md','USE_CASES.md'): copy(root / 'docs' / name)
    (output / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(output.iterdir())))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--version',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();collect(Path(__file__).resolve().parents[1],a.inputs,a.output,a.version,a.run)
    print('Verified five native installers and portable distributions; collection complete')
