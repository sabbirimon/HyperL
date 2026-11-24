# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Private short-lived test certificates. Never a deployment CA or key distributor."""
from pathlib import Path
import subprocess


def certificates(root, identities=("tenant:tenant-a", "cell:cell-a", "cell:unauthorized")):
    root = Path(root)
    def run(*args):
        subprocess.run(["openssl", *map(str, args)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=30)
    ca, ca_key = root / "ca.pem", root / "ca.key"
    run("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", ca_key, "-out", ca,
        "-subj", "/CN=HyperL-private-test-CA", "-days", "1", "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign", "-addext", "subjectKeyIdentifier=hash")
    result = {}
    for index, identity in enumerate(("server", *identities)):
        key, csr, cert, ext = (root / f"{index}.{suffix}" for suffix in ("key", "csr", "pem", "ext"))
        run("req", "-new", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", csr, "-subj", "/CN=" + identity)
        ext.write_text("authorityKeyIdentifier=keyid:always,issuer\nsubjectKeyIdentifier=hash\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=" + ("serverAuth\nsubjectAltName=IP:127.0.0.1\n" if identity == "server" else "clientAuth\n"))
        run("x509", "-req", "-in", csr, "-CA", ca, "-CAkey", ca_key, "-CAcreateserial", "-out", cert, "-days", "1", "-extfile", ext)
        key.chmod(0o600)
        result[identity] = (str(cert), str(key), str(ca))
    ca_key.chmod(0o600)
    return result
