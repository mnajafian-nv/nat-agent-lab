"""Stub external dependencies not available in the test environment."""
import sys
import os
from unittest.mock import MagicMock

# gaia_submit.py imports requests and huggingface_hub at module level.
# Stub them before any test file imports gaia_submit so the module loads
# without requiring those packages to be installed.
for _mod in ("requests", "huggingface_hub"):
    sys.modules.setdefault(_mod, MagicMock())

# Make gaia_tools/ importable as a plain directory (gaia_submit is a script,
# not a package, so we add its parent to sys.path).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
