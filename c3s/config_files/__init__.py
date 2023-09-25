from importlib import resources

__all__ = ['BINARY', 'ISOLATED_2and2', 'ISOLATED_4and2', 'ALLOSTERY', 'NO_ALLOSTERY']

# location of config files used in tests
_data_ref = resources.files('c3s.config_files')
BINARY = (_data_ref/'binary.yml')
ISOLATED_2and2 = (_data_ref/'2+2_isolated.yml')
ISOLATED_4and2 = (_data_ref/'4+2_isolated.yml')
ALLOSTERY = (_data_ref/'allostery.yml')
NO_ALLOSTERY = (_data_ref/'no_allostery.yml')
