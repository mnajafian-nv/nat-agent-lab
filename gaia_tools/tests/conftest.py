"""Stub external dependencies not available in the test environment."""
import sys
import os
from unittest.mock import MagicMock

# gaia_submit.py and ask.py import these at module level.
# Stub them before any test file imports so modules load without
# requiring these packages to be installed.
for _mod in ("requests", "huggingface_hub"):
    sys.modules.setdefault(_mod, MagicMock())

# ask.py imports yaml and readline; yaml is needed for real tests,
# readline can be stubbed if missing.
sys.modules.setdefault("readline", MagicMock())

# Make gaia_tools/ importable as a plain directory (gaia_submit and ask
# are scripts, not packages, so we add their parent to sys.path).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
