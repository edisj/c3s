import h5py
import numpy as np
import c3s


class c3sFile(h5py.File):

    def __init__(self, filename, mode=None):

        super(c3sFile, self).__init__(filename, mode)


    def _initalize_datasets(self):
        pass


def write_h5(filename, system, mode='a', store_trajectory=False, runid=None):

    if mode == 'w':
        with c3sFile(filename, mode=mode) as root:

            root.require_group('cfg/reactions')
            for reaction, rate in system._config_dictionary['reactions'].items():
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

            root.require_group('max_populations')
            for species, value in system._max_populations.items():
                count = np.array(value)
                root['max_populations'].require_dataset(f'{species}', data=count, shape=count.shape, dtype=count.dtype)

    if mode == 'a':
        with c3sFile(filename, mode=mode) as root:

            root.create_group(f'runs/{runid}')

            if store_trajectory:
                traj = system.P_trajectory
                root.create_dataset(f'runs/{runid}/trajectory', data=traj)

            root.require_group(f'runs/{runid}/rates')
            for rate, value in system.rates.items():
                val = np.array(value)
                root[f'runs/{runid}/rates'].create_dataset(f'{rate}', data=val)


def _dict_to_dset(f, group):
    pass

def _dset_to_dict():
    pass

def read_h5(filename, cfg, mode='r', trajectory=False, runid=None):

    with c3sFile(filename, mode=mode) as root:

        '''_reaction_dict = {}
        for reaction in root['cfg/reactions']:
            print(reaction)
            for rate in root[f'cfg/reactions/{reaction}']:
                value = root[f'cfg/reactions/{reaction}/{rate}'][()]
                _reaction_dict[reaction] = [rate, value]
        _config_dictionary = dict(reactions=_reaction_dict)'''

        initial_populations = {key: root[f'initial_populations/{key}'][()] for key in root['initial_populations'].keys()}
        max_populations = {key: root[f'max_populations/{key}'][()] for key in root['max_populations'].keys()}
        states = root['constitutive_states'][()]
        G_ids = {int(key): root[f'G_ids/{key}'][()].tolist() for key in root['G_ids']}
        if trajectory:
            traj = root[f'runs/{runid}/trajectory'][()]

    system = c3s.ChemicalMasterEquation(cfg=cfg, initial_populations=initial_populations, max_populations=max_populations,
                                        empty=True)
    system._G_ids = G_ids
    system._constitutive_states = states
    system._set_generator_matrix()
    if trajectory:
        system._trajectory = traj

    return system
