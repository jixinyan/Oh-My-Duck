import os
from pathlib import Path
import sys

source = str(Path(__file__).resolve().parent / "src")
sys.path.insert(0, source)
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (source, os.environ.get("PYTHONPATH"))))
from oh_my_duck.cli import main

if __name__ == "__main__":
    sys.exit(main())
