"""Proof-only startup guard copied into installed-artifact environments."""

from __future__ import annotations

import _socket
import os
import socket
from typing import Never

SOCKET_DENIAL_ACTIVE = os.environ.get("SALIENCEGATE_ARTIFACT_SOCKET_DENIAL") == "1"
_BLOCKED_MESSAGE = "installed artifact socket or resolver access is disabled"
_LOW_LEVEL_SOCKET = _socket.socket
_LOCAL_SOCKET_FAMILY = getattr(socket, "AF_UNIX", None)
_STARTUP_LOG = os.environ.get("SALIENCEGATE_ARTIFACT_SOCKET_STARTUP_LOG")
_PROVIDER_CREDENTIAL_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_ORGANIZATION",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT",
        "OPENAI_PROJECT_ID",
    }
)
_PRESENT_PROVIDER_CREDENTIAL_KEYS = tuple(
    sorted(key for key in os.environ if key.upper() in _PROVIDER_CREDENTIAL_KEYS)
)
if SOCKET_DENIAL_ACTIVE and _PRESENT_PROVIDER_CREDENTIAL_KEYS:
    raise SystemExit("installed artifact child retained provider credentials")
PROVIDER_CREDENTIAL_DENIAL_ACTIVE = SOCKET_DENIAL_ACTIVE


class ArtifactSocketAccessError(RuntimeError):
    """Raised when an installed-artifact child attempts network access."""


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
            raise ArtifactSocketAccessError(_BLOCKED_MESSAGE)
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
            raise ArtifactSocketAccessError(_BLOCKED_MESSAGE)
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
    raise ArtifactSocketAccessError(_BLOCKED_MESSAGE)


if SOCKET_DENIAL_ACTIVE:
    if type(_STARTUP_LOG) is not str or not _STARTUP_LOG:
        raise RuntimeError("installed artifact socket guard has no startup log")
    _flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        _flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        _flags |= os.O_NOFOLLOW
    _descriptor = os.open(_STARTUP_LOG, _flags, 0o600)
    try:
        os.write(_descriptor, b"installed-artifact-socket-denial-active\n")
    finally:
        os.close(_descriptor)
    for _module in (socket, _socket):
        for _attribute in (
            "create_connection",
            "create_server",
            "getaddrinfo",
            "gethostbyaddr",
            "gethostbyname",
            "gethostbyname_ex",
            "getnameinfo",
        ):
            if hasattr(_module, _attribute):
                setattr(_module, _attribute, _deny_socket)
        _module.socket = (
            _NetworkBlockedSocket if _module is socket else _LowLevelNetworkBlockedSocket
        )
        if hasattr(_module, "SocketType"):
            _module.SocketType = _module.socket
