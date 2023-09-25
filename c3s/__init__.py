"""A tool for stochastically simulating chemcial systems"""

from .simulators.chemical_master_equation import ChemicalMasterEquation
from .simulators.gillespie import Gillespie
#from .h5io import build_system_from_file

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

__all__ = ['ChemicalMasterEquation', 'Gillespie']