"""A tool for stochastically simulating chemcial systems"""

from .simulators import ChemicalMasterEquation, Gillespie

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

__all__ = ['CME', 'Gillespie']