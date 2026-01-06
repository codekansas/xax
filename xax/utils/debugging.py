"""Defines some useful Jax debugging utilities."""

import socket
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any, Callable, Deque

import jax
import jax.numpy as jnp
from jaxtyping import Array


def get_local_ip() -> str | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually connect or send data
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        return ip
    except Exception:
        return None
    finally:
        s.close()


def get_ip_addrs() -> list[str]:
    hostname = socket.gethostname()
    ips: list[str] = []
    for info in socket.getaddrinfo(hostname, None):
        sockaddr = info[4]
        ip = sockaddr[0]
        if isinstance(ip, str):
            ips.append(ip)
    if (local_ip := get_local_ip()) is not None:
        ips.append(local_ip)
    return sorted(set(ips))


def get_named_leaves(
    obj: Any,  # noqa: ANN401
    is_leaf: Callable[[Any], bool] = lambda x: isinstance(x, Array),  # noqa: ANN401
    max_depth: int = 100,
) -> list[tuple[str, Any]]:  # noqa: ANN401
    ret: list[tuple[str, Any]] = []
    q: Deque[tuple[int, str, Any]] = deque()  # noqa: ANN401
    q.append((0, "", obj))

    while q:
        depth, name, node = q.popleft()

        if depth > max_depth:
            continue

        if hasattr(node, "__dict__") and isinstance(node.__dict__, Mapping):
            for cname, cnode in node.__dict__.items():
                gname = f"{name}.{cname}" if name else cname
                if is_leaf(cnode):
                    ret.append((gname, cnode))
                else:
                    q.append((depth + 1, gname, cnode))

        elif isinstance(node, Mapping):
            for cname, cnode in node.items():
                gname = f"{name}.{cname}" if name else cname
                if is_leaf(cnode):
                    ret.append((gname, cnode))
                else:
                    q.append((depth + 1, gname, cnode))

        elif isinstance(node, Iterable):
            for i, cnode in enumerate(node):
                gname = f"{name}.{i}" if name else str(i)
                if is_leaf(cnode):
                    ret.append((gname, cnode))
                else:
                    q.append((depth + 1, gname, cnode))

    return ret


def breakpoint_if_nonfinite(x: Array) -> None:
    is_finite = jnp.isfinite(x).all()

    def true_fn(x: Array) -> None:
        pass

    def false_fn(x: Array) -> None:
        jax.debug.breakpoint()

    jax.lax.cond(is_finite, true_fn, false_fn, x)


def log_if_nonfinite(x: Array, loc: str) -> None:
    is_finite = jnp.isfinite(x).all()

    def true_fn(x: Array) -> None:
        pass

    def false_fn(x: Array) -> None:
        jax.debug.print("=== NaNs: {loc} ===", loc=loc)

    jax.lax.cond(is_finite, true_fn, false_fn, x)
