"""Check Newton solver, converted asset, finite PD stepping and offscreen image."""
import sys
from oh_my_duck.rl.backends.isaac_newton.rollout import main
if __name__ == "__main__":
    sys.exit(main("probe"))
