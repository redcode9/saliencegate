from __future__ import annotations

import _socket
import runpy
import socket
import sys
from typing import Never

_BLOCKED_MESSAGE = "socket or resolver access is disabled for this smoke"
_LOW_LEVEL_SOCKET = _socket.socket


class SocketAccessError(RuntimeError):
    """Raised when an offline smoke attempts to access a socket or resolver."""


class _NetworkBlockedSocket(socket.socket):
    __slots__ = ()

    def __new__(
        cls,
        family: int = socket.AF_INET,
        type: int = socket.SOCK_STREAM,
        proto: int = 0,
        fileno: int | None = None,
    ) -> _NetworkBlockedSocket:
        if family != socket.AF_UNIX or fileno is None:
            raise SocketAccessError(_BLOCKED_MESSAGE)
        return super().__new__(cls, family, type, proto, fileno)


class _LowLevelNetworkBlockedSocket(_LOW_LEVEL_SOCKET):
    __slots__ = ()

    def __new__(
        cls,
        family: int = socket.AF_INET,
        type: int = socket.SOCK_STREAM,
        proto: int = 0,
        fileno: int | None = None,
    ) -> _LowLevelNetworkBlockedSocket:
        if family != socket.AF_UNIX or fileno is None:
            raise SocketAccessError(_BLOCKED_MESSAGE)
        return _LOW_LEVEL_SOCKET.__new__(cls, family, type, proto, fileno)

    def __init__(
        self,
        family: int = socket.AF_INET,
        type: int = socket.SOCK_STREAM,
        proto: int = 0,
        fileno: int | None = None,
    ) -> None:
        _LOW_LEVEL_SOCKET.__init__(self, family, type, proto, fileno)


def _deny_socket(*_args: object, **_kwargs: object) -> Never:
    raise SocketAccessError(_BLOCKED_MESSAGE)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(2)

    for module in (socket, _socket):
        for attribute in (
            "create_connection",
            "create_server",
            "getaddrinfo",
            "gethostbyaddr",
            "gethostbyname",
            "gethostbyname_ex",
            "getnameinfo",
        ):
            if hasattr(module, attribute):
                setattr(module, attribute, _deny_socket)
        module.socket = _NetworkBlockedSocket if module is socket else _LowLevelNetworkBlockedSocket
        if hasattr(module, "SocketType"):
            module.SocketType = module.socket

    program = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(program, run_name="__main__")


if __name__ == "__main__":
    main()
