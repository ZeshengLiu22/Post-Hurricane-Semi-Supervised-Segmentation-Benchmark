#!/usr/bin/env python3
"""Check method-local pretrained weights needed by benchmark training.

This catches two common cases:
- the file is absent;
- the file is only a Git LFS pointer, not the real binary weight file.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "UniMatch ResNet-101 backbone": PROJECT_ROOT / "UniMatch" / "pretrained" / "resnet101.pth",
    "S4MC ResNet-101 ImageNet backbone": PROJECT_ROOT / "s4mc" / "resnet101.pth",
    "Dual-Teacher MiT-B1 backbone": PROJECT_ROOT / "dual_teacher" / "pretrained" / "mit_b1.pth",
    "ClassMix COCO ResNet-101 backbone": PROJECT_ROOT
    / "ClassMix"
    / "pretrained"
    / "resnet101COCO-41f33a49.pth",
}

OPTIONAL = {
    "UniMatch Xception-65 backbone, only needed for Xception configs": PROJECT_ROOT
    / "UniMatch"
    / "pretrained"
    / "xception.pth",
}


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    return path.read_bytes()[:80].startswith(b"version https://git-lfs.github.com/spec")


def describe(path: Path) -> str:
    if not path.exists():
        return "missing"
    if is_lfs_pointer(path):
        return f"Git LFS pointer only ({path.stat().st_size} bytes)"
    return f"present ({path.stat().st_size / (1024 * 1024):.1f} MB)"


def main() -> None:
    failed = False
    print("Required method-local pretrained weights:")
    for label, path in REQUIRED.items():
        status = describe(path)
        print(f"- {label}: {status} -> {path.relative_to(PROJECT_ROOT)}")
        if status.startswith("missing") or status.startswith("Git LFS"):
            failed = True

    print("\nOptional pretrained weights:")
    for label, path in OPTIONAL.items():
        print(f"- {label}: {describe(path)} -> {path.relative_to(PROJECT_ROOT)}")

    if failed:
        raise SystemExit(
            "\nMissing required pretrained weights. Run tools/prepare_method_datasets.py "
            "or copy the real binaries into the method-local paths above."
        )


if __name__ == "__main__":
    main()
