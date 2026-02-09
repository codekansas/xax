"""Updates the API name map from the module imports."""

import argparse
import inspect
import os
import re
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("-i", "--inplace", action="store_true")
    args = parser.parse_args()

    fpath = Path(__file__).parent.parent / "xax" / "__init__.py"
    if not fpath.exists():
        raise ValueError(f"File not found: {fpath.resolve()}")
    root_dir = fpath.parent

    os.environ["XAX_IMPORT_ALL"] = "1"

    import xax  # noqa: PLC0415

    def resolve_module_name(attr_name: str, obj: object) -> str | None:
        if inspect.ismodule(obj):
            return obj.__name__
        if inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isclass(obj):
            return inspect.unwrap(obj).__module__
        module_name = getattr(obj, "__module__", None)
        if isinstance(module_name, str) and module_name.startswith("xax."):
            return module_name
        for name, module in sys.modules.items():
            if not name.startswith("xax."):
                continue
            if getattr(module, attr_name, object()) is obj:
                return name
        return None

    location_map = {}
    for mod in dir(xax):
        if mod.startswith("_"):
            continue
        obj = getattr(xax, mod)
        module_name = resolve_module_name(mod, obj)
        if module_name is not None:
            if module_name == "xax":
                continue
            if module_name.startswith("xax."):
                location_map[mod] = module_name[len("xax.") :]
                continue
        try:
            location = Path(inspect.getfile(obj))
        except Exception:
            continue
        if location.name == "__init__.py":
            continue
        try:
            relative_path = location.relative_to(root_dir)
            import_line = ".".join(relative_path.parts)
            assert import_line.endswith(".py")
            import_line = import_line[: -len(".py")]
            location_map[mod] = import_line
        except Exception:
            continue

    # Sorts by module name, then object name.
    locations = [(k, v) for v, k in sorted([(v, k) for k, v in location_map.items()])]

    with open(fpath, "r") as f:
        lines = f.read()

    # Swaps the `__all__` items.
    all_lines = [f'\n    "{location}",' for location, _ in locations]
    new_all = "__all__ = [" + "".join(all_lines) + "\n]"
    lines = re.sub(r"__all__ = \[.+?\]", new_all, lines, flags=re.DOTALL | re.MULTILINE)

    # Swaps the `NAME_MAP` items.
    name_map_lines = [f'\n    "{k}": "{v}",' for k, v in locations]
    new_name_map = "NAME_MAP: dict[str, str] = {" + "".join(name_map_lines) + "\n}"
    lines = re.sub(r"NAME_MAP: dict\[str, str\] = \{.+?\}", new_name_map, lines, flags=re.DOTALL | re.MULTILINE)

    if args.inplace:
        with open(fpath, "w") as f:
            f.write(lines)
    else:
        sys.stdout.write(lines)
        sys.stdout.flush()


if __name__ == "__main__":
    # python -m scripts.update_api
    main()
