"""Factory that maps a wrapper-name string to an instantiated method wrapper."""

import importlib
import sys
from pathlib import Path

# Ensure the repo root is importable regardless of the current working directory.
_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Exact-name methods: name -> (module, class). Imported lazily so only the chosen
# method's (potentially heavy) dependencies are loaded.
_SIMPLE_WRAPPERS = {
    "superpoint": ("wrappers.superpoint_wrapper", "SuperPointWrapper"),
    "ripe": ("wrappers.ripe_wrapper", "RIPEWrapper"),
    "aliked": ("wrappers.aliked_wrapper", "AlikedWrapper"),
    "sift": ("wrappers.sift_wrapper", "SIFTPyColmapWrapper"),
    "rdd": ("wrappers.rdd_wrapper", "RDDWrapper"),
    "strek": ("wrappers.strek_wrapper", "StrekWrapper"),
    "roma": ("wrappers.roma_wrapper", "RoMaWrapper"),
    "loftr": ("wrappers.loftr_wrapper", "LoFTRWrapper"),
    "mast3r": ("wrappers.mast3r_wrapper", "MAST3RWrapper"),
}


def get_wrappers_list():
    """Get list of available wrapper methods by scanning wrapper files."""
    wrapper_dir = Path(__file__).parent / "wrappers"
    wrappers = []

    for file in wrapper_dir.glob("*_wrapper.py"):
        if file.name not in ["wrapper.py", "example_wrapper.py"]:
            wrapper_name = file.stem.replace("_wrapper", "")
            wrappers.append(wrapper_name)
    return sorted(wrappers)


def _build_wrapper(name, device):
    """Construct the wrapper instance for ``name`` (without setting ``.name``)."""
    if name in ("disk", "disk-kornia"):
        from wrappers.disk_wrapper import DiskWrapperKornia

        return DiskWrapperKornia(device=device)  # use kornia version

    if "dedode" in name:
        from wrappers.dedode_wrapper import DeDoDeWrapper

        return DeDoDeWrapper(device=device, v2="2" in name, descriptor_G="-G" in name)

    if "lightglue" in name:  # call as detector+lightglue, e.g., superpoint+lightglue
        from wrappers.lightglue_wrapper import LightGlueWrapper

        return LightGlueWrapper(detector_name=name.split("+")[0], device=device)

    if name in _SIMPLE_WRAPPERS:
        module_name, cls_name = _SIMPLE_WRAPPERS[name]
        cls = getattr(importlib.import_module(module_name), cls_name)
        return cls(device=device)

    raise ValueError("Wrappers supported: {}".format(get_wrappers_list()))


def wrappers_manager(name, device="cpu"):
    """Build the wrapper for ``name`` on ``device`` and set its ``.name`` attribute."""
    print(f"Creating wrapper for {name} on device {device}.\n")
    wrapper = _build_wrapper(name, device)
    wrapper.name = name
    return wrapper
