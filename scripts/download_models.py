#!/usr/bin/env python3
"""
Build-time model fetch and hash verification.

Runs inside the Docker build's model stage. Downloads the InsightFace pack once,
records a SHA-256 per .onnx file, and — when a manifest is supplied — FAILS THE
BUILD on any mismatch.

Why this exists rather than letting InsightFace download at runtime:

  * On a disposable Pod, a runtime download happens on EVERY burst, adding minutes
    to GPU_READY_TIME and putting a third-party CDN on the provisioning critical
    path where a hiccup becomes a burst failure.
  * Baking makes the model immutable per image tag. That is precisely the property
    FAISS compatibility requires: given an image digest, the model that produced
    an embedding is knowable.
  * It lets /v1/health publish hashes that India's gate can assert. That is the
    only check which cannot be satisfied by a coincidentally-similar model.

USAGE
    python download_models.py --model buffalo_l --dest /models \
        [--manifest model_manifest.json] \
        [--write-manifest /models/model_manifest.lock.json]

FIRST BUILD BOOTSTRAP
    model_manifest.json ships empty. Run once with --write-manifest, then copy the
    generated hashes into model_manifest.json and commit them. From that point on
    the build verifies rather than trusts. Before doing so, confirm the hashes
    match the .onnx files on the India server — that comparison is what actually
    establishes parity (plan Appendix A / Q1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_onnx(root: str) -> Dict[str, str]:
    """Map filename -> absolute path for every .onnx under root."""
    found: Dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename.endswith(".onnx"):
                found[filename] = os.path.join(dirpath, filename)
    return found


def download(model_name: str, dest: str) -> None:
    """Trigger InsightFace's own download into `dest`.

    Uses the library's downloader rather than hard-coded URLs so the layout is
    exactly what FaceAnalysis expects at runtime.
    """
    os.makedirs(dest, exist_ok=True)
    os.environ["INSIGHTFACE_HOME"] = dest

    print(f"[download_models] fetching model pack {model_name!r} into {dest}")

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name=model_name, root=dest, providers=["CPUExecutionProvider"]
    )
    # ctx_id=-1 forces CPU: there is no GPU in the build stage, and we only need
    # the download plus a load check.
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("[download_models] model loaded successfully on CPU (download verified)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and hash InsightFace models")
    parser.add_argument("--model", required=True, help="e.g. buffalo_l or buffalo_s")
    parser.add_argument("--dest", required=True, help="INSIGHTFACE_HOME to populate")
    parser.add_argument(
        "--manifest",
        help="Expected filename -> sha256 map. Build FAILS on mismatch. An empty "
             "or absent manifest only warns, for first-build bootstrap.",
    )
    parser.add_argument(
        "--write-manifest", help="Write the observed hashes here"
    )
    args = parser.parse_args()

    download(args.model, args.dest)

    onnx_files = find_onnx(args.dest)
    if not onnx_files:
        print(
            f"[download_models] FATAL: no .onnx files found under {args.dest}. "
            f"The model pack did not download; refusing to build an image that "
            f"would try to fetch models at runtime.",
            file=sys.stderr,
        )
        return 1

    observed = {name: sha256_file(path) for name, path in sorted(onnx_files.items())}

    print(f"[download_models] {len(observed)} model file(s):")
    for name, digest in observed.items():
        size_mb = os.path.getsize(onnx_files[name]) / 1_048_576
        print(f"    {name:<28} {size_mb:7.1f} MB  sha256={digest}")

    expected = _load_expected(args.manifest)
    if expected:
        problems = _compare(expected, observed)
        if problems:
            print(
                "[download_models] FATAL: model hash verification failed. The "
                "upstream model files differ from the committed manifest, which "
                "means embeddings from this image would not match the FAISS "
                "index built with the approved model:",
                file=sys.stderr,
            )
            for problem in problems:
                print(f"    - {problem}", file=sys.stderr)
            return 1
        print("[download_models] hash verification PASSED against the manifest")
    else:
        print(
            "[download_models] WARNING: no expected hashes supplied, so the model "
            "identity is UNVERIFIED. Copy the hashes above into "
            "model_manifest.json and confirm they match the .onnx files on the "
            "India server before production use."
        )

    if args.write_manifest:
        payload = {
            "model_name": args.model,
            "files": observed,
        }
        os.makedirs(os.path.dirname(args.write_manifest) or ".", exist_ok=True)
        with open(args.write_manifest, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"[download_models] wrote {args.write_manifest}")

    return 0


def _load_expected(path: str | None) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except ValueError as exc:
            print(f"[download_models] manifest is not valid JSON: {exc}",
                  file=sys.stderr)
            return {}
    if isinstance(data, dict) and "files" in data:
        data = data["files"]
    return data if isinstance(data, dict) else {}


def _compare(expected: Dict[str, str], observed: Dict[str, str]) -> List[str]:
    problems: List[str] = []
    for name, digest in expected.items():
        actual = observed.get(name)
        if actual is None:
            problems.append(f"{name} is missing from the download")
        elif actual.lower() != str(digest).lower():
            problems.append(
                f"{name}: got {actual}, expected {digest}"
            )
    unexpected = set(observed) - set(expected)
    if unexpected:
        problems.append(
            f"unexpected model files present: {sorted(unexpected)}"
        )
    return problems


if __name__ == "__main__":
    sys.exit(main())
