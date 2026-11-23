# SPDX-License-Identifier: LicenseRef-HyperL-Community-1.0
"""Build a fixed-size ONNX graph from a HyperL migration and check its selected provider."""
import argparse
import json
from pathlib import Path
import tempfile


def verify(folder, provider, length, options):
    import numpy as np
    import onnx
    import onnxruntime as ort

    if not 1 <= length <= 262144:
        raise ValueError("Choose a length between 1 and 262144")
    if provider not in ort.get_available_providers():
        raise ValueError(f"{provider} unavailable. Installed providers: {ort.get_available_providers()}")
    report = json.loads((folder / "migration.json").read_text())
    if report.get("format") != "hyperl-migration/1" or report.get("target") != "onnx-text":
        raise ValueError("Expected an ONNX text migration directory")
    spec = report["interface"]
    model = onnx.parser.parse_model((folder / "kernel.onnxtxt").read_text())
    op = spec["operation"]
    ops = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply}
    if len(model.graph.node) != 1 or op not in ops or model.graph.node[0].op_type != op:
        raise ValueError("Expected a migrated single pointwise operation")
    for value in [*model.graph.input, *model.graph.output]:
        value.type.tensor_type.shape.dim[0].dim_value = length
    onnx.checker.check_model(model, full_check=True)
    feeds = {item.name: (np.arange(length, dtype=np.float32) % 79 - n * 7) * (0.25 + n)
             for n, item in enumerate(model.graph.input)}
    expected = ops[op](*(feeds[f"data{i}"] for i in spec["operands"]))
    with tempfile.TemporaryDirectory() as temp:
        config = ort.SessionOptions()
        config.enable_profiling = True
        config.profile_file_prefix = str(Path(temp) / "hyperl-ort")
        if provider != "CPUExecutionProvider":
            config.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(model.SerializeToString(), config, providers=[(provider, options)])
        session.disable_fallback()
        actual, = session.run(None, feeds)
        events = json.loads(Path(session.end_profiling()).read_text())
    providers = {e["args"]["provider"] for e in events if e.get("cat") == "Node" and e.get("args", {}).get("provider")}
    if providers != {provider}:
        raise ValueError(f"Could not verify requested node placement. Observed {sorted(providers)}")
    np.testing.assert_array_equal(actual, expected)
    return {"provider": provider, "runtime": ort.__version__, "operation": op, "verifiedElements": length,
            "deviceIdentityVerified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--length", type=int, default=137)
    parser.add_argument("--provider-options", type=json.loads, default={})
    args = parser.parse_args()
    print(json.dumps(verify(args.folder, args.provider, args.length, args.provider_options)))
