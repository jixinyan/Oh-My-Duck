import pytest

from oh_my_duck.infrastructure.ssh import ssh_command


@pytest.mark.parametrize("host", ["", "-oProxyCommand=command", "host name", "host\n", "host\0", None])
def test_invalid_destination_is_rejected_before_execution(host):
    with pytest.raises(ValueError):
        ssh_command(host, "hostname")


@pytest.mark.parametrize("command", ["", " ", "hostname\0", None])
def test_invalid_remote_command_is_rejected_before_execution(command):
    with pytest.raises(ValueError):
        ssh_command("server", command)


@pytest.mark.parametrize("count", [True, 0, -1, 1.0])
def test_invalid_keepalive_count_is_rejected_before_execution(count):
    with pytest.raises(ValueError):
        ssh_command("server", "hostname", server_alive_count_max=count)


@pytest.mark.parametrize("port", [True, 0, -1, 65536, 8784.0])
def test_invalid_forwarded_port_is_rejected_before_execution(port):
    with pytest.raises(ValueError):
        ssh_command("server", "hostname", forwarded_ports=[(port, 8784)])
