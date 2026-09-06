"""Repository configuration discovery, independent of optional dependencies."""
import os
from pathlib import Path


def project_root() -> Path:
    explicit = os.environ.get("OMD_PROJECT_ROOT")
    candidates = [Path(explicit)] if explicit else [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "configs/project.json").is_file():
            return candidate.resolve()
    raise RuntimeError("Run from an Oh-My-Duck checkout, or set OMD_PROJECT_ROOT")
