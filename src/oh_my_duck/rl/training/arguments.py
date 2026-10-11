import argparse
from collections.abc import Sequence


def task_selection(arguments: Sequence[str]) -> str | None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--task", action="append", default=[])
    selected, _ = parser.parse_known_args(arguments)
    if len(selected.task) > 1:
        parser.error("--task 必须唯一")
    return selected.task[0] if selected.task else None
