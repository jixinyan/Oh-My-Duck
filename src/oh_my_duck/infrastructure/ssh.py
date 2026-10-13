from collections.abc import Sequence


def ssh_command(
    host: str,
    command: str,
    *,
    forwarded_ports: Sequence[tuple[int, int]] = (),
    server_alive_count_max: int = 3,
) -> list[str]:
    if (not isinstance(host, str) or not host or host.startswith("-")
            or any(character.isspace() or character == "\0" for character in host)):
        raise ValueError("SSH host must be a nonempty destination without whitespace or option prefixes")
    if not isinstance(command, str) or not command.strip() or "\0" in command:
        raise ValueError("SSH requires a nonempty remote command without NUL characters")
    if type(server_alive_count_max) is not int or server_alive_count_max < 1:
        raise ValueError("SSH keepalive count must be a positive integer")
    arguments = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                 "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                 "-o", f"ServerAliveCountMax={server_alive_count_max}"]
    for local_port, remote_port in forwarded_ports:
        if any(type(port) is not int or not 1 <= port <= 65535 for port in (local_port, remote_port)):
            raise ValueError("SSH forwarded ports must be integers from 1 to 65535")
        arguments += ["-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}"]
    return [*arguments, host, command]
