#!/usr/bin/env python3
"""Source-checkout entry point. Installed editable projects also expose the omd command."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from oh_my_duck.cli import main

if __name__ == "__main__":
    sys.exit(main())
