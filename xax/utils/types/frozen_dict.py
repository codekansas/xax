"""Defines a frozen dictionary type.

This is mostly taken from Flax - we move it here to avoid having to use Flax as
a dependency in downstream projects.
"""

from types import MappingProxyType
from typing import Any, Iterator, KeysView, Mapping, Self, TypeVar, ValuesView, cast

import jax

K = TypeVar("K")
V = TypeVar("V")


def _prepare_freeze(xs: Any) -> Any:  # noqa: ANN401
    """Deep copy unfrozen dicts to make the dictionary FrozenDict safe."""
    if isinstance(xs, FrozenDict):
        return xs._dict  # pylint: disable=protected-access
    if not isinstance(xs, dict):
        return xs
    return {key: _prepare_freeze(val) for key, val in xs.items()}


def _indent(x: str, num_spaces: int) -> str:
    indent_str = " " * num_spaces
    lines = x.split("\n")
    assert not lines[-1]
    return "\n".join(indent_str + line for line in lines[:-1]) + "\n"


class FrozenKeysView(KeysView[K]):
    def __repr__(self) -> str:
        return f"frozen_dict_keys({list(self)})"


class FrozenValuesView(ValuesView[V]):
    def __repr__(self) -> str:
        return f"frozen_dict_values({list(self)})"


@jax.tree_util.register_pytree_with_keys_class
class FrozenDict(Mapping[K, V]):
    """An immutable variant of the Python dict."""

    __slots__ = ("_dict", "_hash")

    def __init__(self, *args: Any, __unsafe_skip_copy__: bool = False, **kwargs: Any) -> None:  # noqa: ANN401
        xs = cast(dict[K, V], dict(*args, **kwargs))
        if __unsafe_skip_copy__:
            self._dict = xs
        else:
            self._dict = _prepare_freeze(xs)

        self._hash: int | None = None

    def __getitem__(self, key: K) -> V:
        v = self._dict[key]
        if isinstance(v, dict):
            return FrozenDict(v)  # type: ignore[return-value]
        return v

    def __setitem__(self, key: K, value: V) -> None:
        raise ValueError("FrozenDict is immutable.")

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def __iter__(self) -> Iterator[K]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return self.pretty_repr()

    def __reduce__(self) -> tuple[type["FrozenDict[K, V]"], tuple[dict[K, V]]]:
        return FrozenDict, (self.unfreeze(),)

    def pretty_repr(self, num_spaces: int = 4) -> str:
        """Returns an indented representation of the nested dictionary."""

        def pretty_dict(x: Any) -> str:  # noqa: ANN401
            if not isinstance(x, dict):
                return repr(x)
            rep = ""
            for key, val in x.items():
                rep += f"{key}: {pretty_dict(val)},\n"
            if rep:
                return "{\n" + _indent(rep, num_spaces) + "}"
            else:
                return "{}"

        return f"FrozenDict({pretty_dict(self._dict)})"

    def __hash__(self) -> int:
        if self._hash is None:
            h = 0
            for key, value in self.items():
                h ^= hash((key, value))
            self._hash = h
        return self._hash

    def copy(self, add_or_replace: Mapping[K, V] = MappingProxyType({})) -> Self:
        return type(self)({**self, **unfreeze(add_or_replace)})  # type: ignore[arg-type]

    def keys(self) -> FrozenKeysView[K]:
        return FrozenKeysView(self)

    def values(self) -> FrozenValuesView[V]:
        return FrozenValuesView(self)

    def items(self) -> Iterator[tuple[K, V]]:  # type: ignore[override]
        for key in self._dict:
            yield (key, self[key])

    def pop(self, key: K) -> tuple["FrozenDict[K, V]", V]:
        value = self[key]
        new_dict = dict(self._dict)
        new_dict.pop(key)
        new_self = type(self)(new_dict)
        return new_self, value

    def unfreeze(self) -> dict[K, V]:
        return unfreeze(self)

    def tree_flatten_with_keys(self) -> tuple[tuple[tuple[jax.tree_util.DictKey, Any], ...], tuple[K, ...]]:
        sorted_keys = sorted(self._dict)
        return tuple([(jax.tree_util.DictKey(k), self._dict[k]) for k in sorted_keys]), tuple(sorted_keys)

    @classmethod
    def tree_unflatten(cls, keys: tuple[K, ...], values: tuple[Any, ...]) -> "FrozenDict[K, V]":
        return cls({k: v for k, v in zip(keys, values, strict=True)}, __unsafe_skip_copy__=True)


def unfreeze(x: FrozenDict[K, V] | dict[str, Any]) -> dict[Any, Any]:  # noqa: ANN401
    if isinstance(x, FrozenDict):
        return jax.tree.map(lambda y: y, x._dict)
    elif isinstance(x, dict):
        ys = {}
        for key, value in x.items():
            ys[key] = unfreeze(value)
        return ys
    else:
        return x


def freeze_dict(x: Mapping[K, V]) -> FrozenDict[K, V]:
    return FrozenDict(x)
