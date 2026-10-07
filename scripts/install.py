#!/usr/bin/env python3
"""Verify a release archive and install into a new user-selected prefix. No downloads/root."""
import argparse,hashlib,os,re,shutil,stat,tempfile,zipfile
from pathlib import Path,PurePosixPath

def install(archive:Path,prefix:Path,expected_sha:str):
    if not re.fullmatch(r'[a-fA-F0-9]{64}',expected_sha):raise ValueError('Expected SHA-256 required')
    digest=hashlib.sha256()
    with archive.open('rb') as src:
        for chunk in iter(lambda:src.read(1024*1024),b''):digest.update(chunk)
    if digest.hexdigest()!=expected_sha.lower():raise ValueError('Archive SHA-256 mismatch')
    prefix=prefix.absolute()
    if prefix.exists() or prefix.is_symlink():raise ValueError('Choose a new prefix; installer never overwrites')
    with zipfile.ZipFile(archive) as z:
        files=z.infolist()
        if not files or len(files)>2048 or sum(f.file_size for f in files)>256*1024*1024:raise ValueError('Archive size/file limit')
        seen=set();root=None
        for f in files:
            name=f.filename;p=PurePosixPath(name);mode=f.external_attr>>16
            if '\\' in name or ':' in name or p.is_absolute() or not p.parts or '..' in p.parts or any(part in ('.','') for part in p.parts):raise ValueError('Unsafe archive path')
            if any(part.endswith((' ','.')) for part in p.parts):raise ValueError('Ambiguous platform path')
            reserved={'CON','PRN','AUX','NUL'}|{f'{kind}{n}' for kind in ('COM','LPT') for n in range(1,10)}
            if any(part.split('.')[0].upper() in reserved for part in p.parts):raise ValueError('Reserved Windows path')
            if mode and stat.S_IFMT(mode) not in (0,stat.S_IFREG,stat.S_IFDIR):raise ValueError('Special/symlink entry rejected')
            if f.file_size>64*1024*1024:raise ValueError('Entry size limit')
            identity=str(p).casefold()
            if identity in seen:raise ValueError('Duplicate archive entry')
            seen.add(identity)
            if root is None:root=p.parts[0]
            if p.parts[0]!=root or not root.startswith('hyperl-'):raise ValueError('Expected one HyperL release root')
        stage=Path(tempfile.mkdtemp(prefix='.hyperl-install-',dir=prefix.parent))
        try:
            for f in files:
                target=stage.joinpath(*PurePosixPath(f.filename).parts)
                if f.is_dir():target.mkdir(parents=True,exist_ok=True);continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(f) as src,target.open('xb') as dst:
                    copied=0
                    while True:
                        data=src.read(1024*1024)
                        if not data:break
                        copied+=len(data)
                        if copied>f.file_size:raise ValueError('Expanded size mismatch')
                        dst.write(data)
                    if copied!=f.file_size:raise ValueError('Truncated archive entry')
                os.chmod(target,0o755 if target.name=='hyperl' and target.parent.name=='bin' else 0o644)
            package=stage/root
            if not (package/'bin/hyperl').is_file() or not (package/'bin/hyperl.bat').is_file():raise ValueError('Launchers missing')
            if prefix.exists() or prefix.is_symlink():raise ValueError('Prefix changed during install')
            package.rename(prefix)
        finally:shutil.rmtree(stage)
    return prefix

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive',type=Path);parser.add_argument('prefix',type=Path);parser.add_argument('sha256')
    a=parser.parse_args();print('Installed:',install(a.archive,a.prefix,a.sha256));print('No PATH/global settings changed. Uninstall only that prefix; keep keys/datasets separately.')
