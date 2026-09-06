"""Finite PD diagnostic replay. This does not validate a BAM walking policy."""
import sys
from oh_my_duck.rl.backends.isaac_newton.rollout import main
if __name__ == "__main__":
    sys.exit(main("eval"))
