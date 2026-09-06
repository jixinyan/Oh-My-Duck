"""Finite PD diagnostic replay. This does not validate a BAM walking policy."""
import sys
from .rollout import main
if __name__ == "__main__":
    sys.exit(main("eval"))
