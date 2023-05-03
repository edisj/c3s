import h5py
import c3s
import numpy as np


def Group_to_Dict(group):
    """stores nested hdf5 groups into python dictionary

    needs work.."""

    dictionary = {}
    def recurseGroup(group):
        for key in group:
            if isinstance(group, h5py.Group):
                dictionary[key] = list(group[key].keys())
                # recurseGroup(g[key])
    return dictionary


def Dict_to_Group(dictionary, group):
    """stores python dictionary into nested hdf5 groups"""

    def recurseDict(d, extended_key=None):
        for key, value in d.items():
            if isinstance(value, dict):
                fullpath = f'{extended_key}/{key}' if extended_key else f'{key}'
                recurseDict(value, extended_key=fullpath)
            else:
                fullpath = f'{extended_key}/{key}' if extended_key else f'{key}'
                group.create_dataset(fullpath, data=value)

    recurseDict(dictionary)


class c3sH5IO:

    def __init__(self, filename, mode, *args, **kwargs):

        self.filename = filename
        self.mode = mode

        self._file = self._open_file(filename, mode)



    def save_system(self):
        ...

    def _open_file(self, filename, mode):
        return h5py.File(filename, mode, track_order=True)

    def create_group(self, name):
        self._file.create_group(name)

    def create_dataset(self, name, data):
        self._file.create_dataset(name, data=data)

    def write_original_config(self, config_file):
        ...





def write_h5(filename, system, mode='a', store_trajectory=False, runid=None):

    if mode == 'w':
        with h5py.File(filename, mode=mode) as root:
            root.require_group('cfg/reactions')
            for reaction, rate in system.Reactions._original_config['reactions'].items():
                root.create_dataset(f"cfg/reactions/{reaction.replace(' ', '')}/{rate[0]}", data=rate[1])

            root.require_dataset('constitutive_states', data=system.states, shape=system.states.shape, dtype=system.states.dtype)

            root.require_group('G_ids')
            for k, value in system._G_ids.items():
                ids = np.array(value)
                root['G_ids'].require_dataset(f'{k}', data=ids, shape=ids.shape, dtype=ids.dtype)

            root.require_group('initial_populations')
            for species, value in system._initial_populations.items():
                count = np.array(value)
                root['initial_populations'].require_dataset(f'{species}', data=count, shape=count.shape, dtype=count.dtype)

            if system._max_populations:
                root.require_group('max_populations')
                for species, value in system._max_populations.items():
                    count = np.array(value)
                    root['max_populations'].require_dataset(f'{species}', data=count, shape=count.shape, dtype=count.dtype)

    if store_trajectory:
        with h5py.File(filename, mode=mode) as root:
            root.create_group(f'runs/{runid}')
            if store_trajectory:
                traj = system._trajectory
                root.create_dataset(f'runs/{runid}/trajectory', data=traj)
                for rate, value in system.rates.items():
                    root[f'runs/{runid}/trajectory'].attrs[rate] = value



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

        system = c3s.ChemicalMasterEquation(config_file=cfg, initial_populations=initial_populations, max_populations=max_populations,
                                            empty=True, low_memory=low_memory)
        system._G_ids = G_ids
        system._constitutive_states = root['constitutive_states'][()]
        system.M = len(system._constitutive_states)
        if trajectory:
            system._trajectory = root[f'runs/{runid}/trajectory'][()]

    return system
