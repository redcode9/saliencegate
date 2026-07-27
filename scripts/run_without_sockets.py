from __future__ import annotations

import _socket
import runpy
import socket
import sys
from contextlib import suppress
from typing import Never

_BLOCKED_MESSAGE = "socket or resolver access is disabled for this smoke"
_LOW_LEVEL_SOCKET = _socket.socket
_LOCAL_SOCKET_FAMILY = getattr(socket, "AF_UNIX", None)
_PUBLIC_SOCKET = socket.socket


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
        if _LOCAL_SOCKET_FAMILY is None or family != _LOCAL_SOCKET_FAMILY or fileno is None:
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
        if _LOCAL_SOCKET_FAMILY is None or family != _LOCAL_SOCKET_FAMILY or fileno is None:
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


def _windows_local_socketpair(
    family: int | None = None,
    type: int = socket.SOCK_STREAM,
    proto: int = 0,
) -> tuple[socket.socket, socket.socket]:
    if family is None:
        family = socket.AF_INET
    if family == socket.AF_INET:
        host = "127.0.0.1"
    elif family == socket.AF_INET6:
        host = "::1"
    else:
        raise ValueError("local socket pairs require AF_INET or AF_INET6")
    if type != socket.SOCK_STREAM:
        raise ValueError("local socket pairs require SOCK_STREAM")
    if proto != 0:
        raise ValueError("local socket pairs require protocol zero")

    listener = _PUBLIC_SOCKET(family, type, proto)
    client: socket.socket | None = None
    server: socket.socket | None = None
    accepted_fileno: int | None = None
    try:
        listener.bind((host, 0))
        listener.listen()
        address, port = listener.getsockname()[:2]
        client = _PUBLIC_SOCKET(family, type, proto)
        client.setblocking(False)
        with suppress(BlockingIOError, InterruptedError):
            client.connect((address, port))
        accepted_fileno, _ = listener._accept()  # type: ignore[attr-defined]
        client.setblocking(True)
        server = _PUBLIC_SOCKET(family, type, proto, fileno=accepted_fileno)
        accepted_fileno = None
        return server, client
    except BaseException:
        if server is not None:
            server.close()
        if accepted_fileno is not None:
            socket.close(accepted_fileno)
        if client is not None:
            client.close()
        raise
    finally:
        listener.close()


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

    if sys.platform == "win32":
        socket.socketpair = _windows_local_socketpair

    program = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(program, run_name="__main__")


if __name__ == "__main__":
    main()
