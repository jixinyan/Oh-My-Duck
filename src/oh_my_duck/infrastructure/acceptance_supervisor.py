import argparse
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys

from oh_my_duck.infrastructure.owned_process import owned_process


def main():
    parser = argparse.ArgumentParser(description="管理本次验收的子进程，控制连接结束时关闭所属进程")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.record.exists():
        raise ValueError("An actual command and a new lifecycle record are required")
    cancelled = False

    def cancel(_signum, _frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    with selectors.DefaultSelector() as control:
        control.register(sys.stdin, selectors.EVENT_READ)
        with owned_process(command, record_path=args.record, stdin=subprocess.DEVNULL) as process:
            while process.poll() is None and not cancelled:
                for key, _events in control.select(timeout=0.1):
                    message = os.read(key.fd, 64)
                    if message:
                        raise ValueError("Acceptance control channel must remain empty until closure")
                    cancelled = True
            if not cancelled:
                return process.wait()
    return 130


if __name__ == "__main__":
    raise SystemExit(main())
