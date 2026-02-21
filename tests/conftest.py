# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

_fvdb_available = True
try:
    import fvdb  # noqa: F401
except Exception:
    _fvdb_available = False


def pytest_ignore_collect(collection_path, config):
    """Skip fvdb test collection when fvdb is not importable.

    The fvdb package requires torch_scatter and compiled C++ extensions
    that are only present in the conda fvdb environment, not in the
    fvdb_cutile venv used for fvdb_tile development.
    """
    if not _fvdb_available:
        return True
