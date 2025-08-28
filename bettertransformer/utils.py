from typing import Optional, Tuple, Union, List
import importlib.util
import importlib.metadata
import functools


def recurse_getattr(obj, attr: str):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def recurse_setattr(module, name, value):
    """A function to recursively set attributes to a module."""
    if "." not in name:
        setattr(module, name, value)
    else:
        name, rest = name.split(".", 1)
        recurse_setattr(getattr(module, name), rest, value)

def _is_package_available(
    pkg_name: str,
    return_version: bool = False,
    pkg_distributions: Optional[List[str]] = None,
) -> Union[Tuple[bool, str], bool]:
    """
    Check if a package is available in the current environment and not just an importable module by checking its version.
    Optionally return the version of the package.

    Args:
        pkg_name (str): The name of the package to check.
        return_version (bool): Whether to return the version of the package.
        pkg_distributions (Optional[List[str]]): A list of package distributions (e.g. "package-name", "package-name-gpu", etc.) to check for the package.

    Returns:
        Union[Tuple[bool, str], bool]: A tuple of the package availability and the version of the package if `return_version` is `True`.
    """

    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"

    if pkg_distributions is None:
        pkg_distributions = [pkg_name]
    else:
        pkg_distributions.append(pkg_name)

    if package_exists:
        for pkg in pkg_distributions:
            try:
                package_version = importlib.metadata.version(pkg)
                package_exists = True
                break
            except importlib.metadata.PackageNotFoundError:
                package_exists = False
                pass

    if return_version:
        return package_exists, package_version
    else:
        return package_exists