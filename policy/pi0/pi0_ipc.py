import pickle
import socket
import struct
from typing import Any


_HEADER_SIZE = 4


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("pi0 inference process closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_msg(sock: socket.socket, msg: Any) -> None:
    payload = pickle.dumps(msg, protocol=4)
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_msg(sock: socket.socket) -> Any:
    header = _recv_exact(sock, _HEADER_SIZE)
    (size,) = struct.unpack("!I", header)
    return pickle.loads(_recv_exact(sock, size))
