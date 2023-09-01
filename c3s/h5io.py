import c3s
import h5py
import numpy as np
from pathlib import Path


def read_c3s(filename, mode='r', trajectory_name=None):
    with CMEReader(filename=filename, mode=mode, trajectory_name=trajectory_name) as R:
        return R.system_from_file


def write_c3s(filename, system, mode='x', trajectory_name=None):
    """writes simulation data to an hdf5 file"""

    fresh_file = not Path(filename).exists()
    with CMEWriter(filename, system, mode, fresh=fresh_file) as W:
        W.write(filename)


class CMEReader:
    def __init__(self, filename, mode, trajectory_name=None):
        self.filename = filename
        self.mode = mode
        self.trajectory_name = trajectory_name
        self.file = self._open_file()
        config = self._get_config()
        initial_populations, max_populations, self.nonzero_G_elements = self._get_system_info()
        self.system_from_file = c3s.ChemicalMasterEquation(
            config=config,
            initial_populations=initial_populations,
            max_populations=max_populations,
            empty=True)

        self._fill_system_from_file()

    def _open_file(self):
        return h5py.File(self.filename, self.mode, track_order=True)

    def _fill_system_from_file(self):

        self.system_from_file._nonzero_G_elements = self.nonzero_G_elements

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

    def _write_system_info(self, filename, mode):
        with CMEWriter(filename, system=self, mode=mode) as W:
            # basic reaction info
            W._dump_config()
            W._create_dataset('constitutive_states', data=self.states)
            for k, indices in self.system._nonzero_G_elements.items():
                W._create_dataset(f'nonzero_G_elements/{k}', data=np.array(indices))
            for species, count in self.system._initial_populations.items():
                W._create_dataset(f'initial_populations/{species}', data=np.array(count))
            if self.system._max_populations:
                for species, count in self.system._max_populations.items():
                    W._create_dataset(f'max_populations/{species}', data=np.array(count))

    def _write_trajectory(self, filename, mode, trajectory_name):
        with CMEWriter(filename, system=self, mode=mode) as W:
            traj_group = W._require_group('trajectories')
            if trajectory_name is None:
                trajectory_name = f'trajectory00{len(traj_group) + 1}'
            W._create_dataset(f'trajectories/{trajectory_name}/trajectory', data=self.trajectory)
            for rate, value in self.rates.items():
                rate_group = W._create_group(f'trajectories/{trajectory_name}/rates/{rate}')
                W._set_attr(rate_group, name='value', value=value)


    def _write_species(self):
        ...
    def _write_reaction_matrix(self):
        ...
    def _write_propensities(self):
        ...

    def close(self):
        self.file.close()
    # define __enter__ and __exit__ so this class can be used as a context manager
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
