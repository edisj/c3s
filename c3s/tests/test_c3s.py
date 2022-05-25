"""
Unit and regression test for the c3s package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import c3s


def test_c3s_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "c3s" in sys.modules
