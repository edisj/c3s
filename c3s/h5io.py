import c3s
import h5py
import numpy as np


def read(filename, mode='r', trajectory_name=None, low_memory=False):

    with CMEReader(
        filename=filename,
        mode=mode,
        trajectory_name=trajectory_name,
        low_memory=low_memory) as R:

        return R.system_from_file

def read_h5(filename, cfg, mode='r', trajectory=False, runid=None, low_memory=False):

    with h5py.File(filename, mode=mode) as root:

        '''_reaction_dict = {}
        for reaction in root['cfg/reactions']:
            print(reaction)
            for rate in root[f'cfg/reactions/{reaction}']:
                value = root[f'cfg/reactions/{reaction}/{rate}'][()]
                _reaction_dict[reaction] = [rate, value]
        _config_dictionary = dict(reactions=_reaction_dict)'''
        initial_populations = {key: root[f'initial_populations/{key}'][()] for key in root['initial_populations'].keys()}
        if 'max_populations' in root:
            max_populations = {key: root[f'max_populations/{key}'][()] for key in root['max_populations'].keys()}
        else:
            max_populations=None
        G_ids = {int(key): root[f'G_ids/{key}'][()].tolist() for key in root['G_ids']}

        system = c3s.ChemicalMasterEquation(config=cfg, initial_populations=initial_populations, max_populations=max_populations,
                                            empty=True, low_memory=low_memory)
        system._nonzero_G_elements = G_ids
        system._constitutive_states = root['constitutive_states'][()]
        system.M = len(system._constitutive_states)
        if trajectory:
            system._trajectory = root[f'runs/{runid}/trajectory'][()]

    return system

class CMEReader:

    def __init__(self, filename, mode, trajectory_name=None, low_memory=False):
        self.filename = filename
        self.mode = mode
        self.trajectory_name = trajectory_name
        self.low_memory = low_memory
        self.file = self._open_file()

        config = self._get_config()
        initial_populations, max_populations, self.nonzero_G_elements = self._get_system_info()

        self.system_from_file = c3s.ChemicalMasterEquation(
            config=config,
            initial_populations=initial_populations,
            max_populations=max_populations,
            low_memory=low_memory,
            states_from='combinatorics',
            empty=True)

        self._fill_system_from_file()

    def _open_file(self):
        return h5py.File(self.filename, self.mode, track_order=True)

    def _fill_system_from_file(self):

        self.system_from_file._nonzero_G_elements = self.nonzero_G_elements

        if not self.low_memory:
            self.system_from_file.states = self._get_states()
            self.system_from_file.M = len(self.system_from_file.states )
            self.system_from_file._set_generator_matrix()

        if self.trajectory_name:
            self.system_from_file.trajectory = self._get_trajectory()

    def _get_config(self):
        config_dictionary = {'reactions': {}, 'constraints': {}}
        for reaction_string, rate_group in self.file['original_config/reactions'].items():
            rate_string = list(rate_group.keys())[0]
            rate_value = self._get_attr(rate_group[rate_string], name='value')
            config_dictionary['reactions'][reaction_string] = [rate_string, rate_value]
        for constraint_string, constraint_group in self.file['original_config/constraints'].items():
            value = list(constraint_group.keys())[0]
            config_dictionary['constraints'][constraint_string] = value
        return config_dictionary

    def _get_system_info(self):
        initial_populations = {key : self.file[f'initial_populations/{key}'][()]
                               for key in self.file['initial_populations'].keys()}
        if 'max_populations' in self.file:
            max_populations = {key : self.file[f'max_populations/{key}'][()]
                               for key in self.file['max_populations'].keys()}
        else:
            max_populations = None
        nonzero_G_elements = {int(key) : self.file[f'nonzero_G_elements/{key}'][()].tolist()
                              for key in self.file['nonzero_G_elements']}

        return initial_populations, max_populations, nonzero_G_elements

    def _get_attr(self, parent, name):
        return parent.attrs[name]

    def _get_states(self):
        return self.file['constitutive_states'][()]

    def _get_trajectory(self):

        trajectory_group = self.file[f'trajectories/{self.trajectory_name}']
        rate_group = trajectory_group['rates']
        rates_from_trajectory = {key: rate_group[key].attrs['value'] for key in rate_group}

        # should probably do this somewhere else
        self.system_from_file.update_rates(rates_from_trajectory)

        return trajectory_group['trajectory'][()]



    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class CMEWriter:
    """rough draft of a more robust file writer"""

    def __init__(self, filename, system, mode='x', **kwargs):
        self.system = system
        self.filename = filename
        self.mode = mode
        self.file = self._open_file()

    def _open_file(self):
        return h5py.File(self.filename, self.mode, track_order=True)
    def _create_group(self, name):
        return self.file.create_group(name, track_order=True)
    def _require_group(self, name):
        return self.file.require_group(name)
    def _create_dataset(self, name, data):
        return self.file.create_dataset(name, data=data, track_order=True)
    def _set_attr(self, parent, name, value):
        parent.attrs[name] = value

    def _dump_config(self):
        self._write_original_config()
        self._write_species()
        self._write_reaction_matrix()
        self._write_propensities()

    def _write_original_config(self):
        reaction_strings = self.system.reactions._reaction_strings
        rates = self.system.reactions._rates
        constraint_strings = self.system.reactions._constraint_strings
        constraints = self.system.reactions._constraints
        for rxn, rate in zip(reaction_strings, rates):
            rate_group = self._create_group(f'original_config/reactions/{rxn}/{rate[0]}')
            self._set_attr(rate_group, name='value', value=rate[1])
        for string, constraint in zip(constraint_strings, constraints):
            value = constraint[-1]
            constraint_group = self._create_group(f'original_config/constraints/{string}/{value}')

    def _write_species(self):
        ...
    def _write_reaction_matrix(self):
        ...
    def _write_propensities(self):
        ...

    def close(self):
        self.file.close()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def group_to_dict(group):
    """stores nested hdf5 groups into python dictionary

    needs work..
    """
    dictionary = {}
    def recurse_though_group(group):
        for key in group:
            if isinstance(group, h5py.Group):
                dictionary[key] = list(group[key].keys())
                # recurseGroup(g[key])

    # need to use 1 of these
    '''def nested_set(dic, keys, value):
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})
        dic[keys[-1]] = value'''

    """def nested_set(dic, keys, value, create_missing=True):
    d = dic
    for key in keys[:-1]:
        if key in d:
            d = d[key]
        elif create_missing:
            d = d.setdefault(key, {})
        else:
            return dic
    if keys[-1] in d or create_missing:
        d[keys[-1]] = value
    return dic"""
    return dictionary


def dict_to_group(dictionary, group):
    """recursively iterate through a dictionary and store data into `h5py.Group`s"""

    # use tempfile instead to return a group object ??
    def recurse_though_dictionary(d, previous_keys=None):
        for key, value in d.items():
            fullpath = f'{previous_keys}/{key}' if previous_keys else f'{key}'
            if isinstance(value, dict):
                recurse_though_dictionary(value, previous_keys=fullpath)
            else:
                _store_data_based_on_type(key, group, value)

    recurse_though_dictionary(dictionary)


def _store_data_based_on_type(key, group, data):
    """probably a cleaner way to do this w decorator"""

    possibilities = dict(
        is_scalar = isinstance(data, int) or isinstance(data, float),
        is_array = isinstance(data, np.ndarray))
        #is_list = isinstance(data, list),
        #is_tuple = isinstance(data, tuple))
        # string ??

    dispatch_function = dict(
        is_scalar = group.attrs.create,
        is_array = group.create_dataset)
        #is_list
        #is_tuple
        #is_string

    for type_, is_type in possibilities.items():
        if is_type:
            return dispatch_function[type_]


