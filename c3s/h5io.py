import c3s
import h5py
import numpy as np
from pathlib import Path
from .sparse_matrix import SparseMatrix


def read_c3s(filename, mode='r', trajectory_name=None):
    with CMEReader(filename=filename, mode=mode, trajectory_name=trajectory_name) as R:
        return R.system_from_file


class CMEReader:
    def __init__(self, filename, mode, trajectory_name=None):
        self.filename = filename
        self.mode = mode
        self.trajectory_name = trajectory_name
        self.file = self._open_file()
        config = self._get_config()
        initial_populations, max_populations, self.G_lines, self.G_columns, self.G_values = self._get_system_info()
        self.system_from_file = c3s.ChemicalMasterEquation(
            config=config,
            initial_populations=initial_populations,
            max_populations=max_populations,
            empty=True)

        self._fill_system_from_file()

    def _open_file(self):
        return h5py.File(self.filename, self.mode, track_order=True)

    def _fill_system_from_file(self):
        self.system_from_file._constitutive_states = self._get_states()
        self.system_from_file._generator_matrix = SparseMatrix(self.G_lines, self.G_columns, self.G_values)
        if self.trajectory_name:
            self.system_from_file.trajectory = self._get_trajectory()

    def _get_config(self):
        config_dictionary = {'reactions': {}, 'constraints': {}}
        for reaction_string, rate_group in self.file['original_config/reactions'].items():
            rate_string = list(rate_group.keys())[0]
            rate_value = self._get_attr(rate_group[rate_string], name='value')
            config_dictionary['reactions'][reaction_string] = [rate_string, rate_value]
        config_dictionary['constraints'] = list(self.file['original_config/constraints'].keys())
        return config_dictionary

    def _get_system_info(self):
        if 'initial_popultaions' in self.file:
            initial_populations = {key : self.file[f'initial_populations/{key}'][()]
                                   for key in self.file['initial_populations'].keys()}
        else:
            initial_populations = None
        if 'max_populations' in self.file:
            max_populations = {key : self.file[f'max_populations/{key}'][()]
                               for key in self.file['max_populations'].keys()}
        else:
            max_populations = None
        G_lines = self.file['generator_matrix/G_rows'][()]
        G_columns = self.file['generator_matrix/G_columns'][()]
        G_values = self.file['generator_matrix/G_values'][()]

        return initial_populations, max_populations, G_lines, G_columns, G_values

    def _get_attr(self, parent, name):
        return parent.attrs[name]

    def _get_states(self):
        return self.file['constitutive_states'][()]

    def _get_trajectory(self):

        trajectory_group = self.file[f'trajectories/{self.trajectory_name}']
        rate_group = trajectory_group['rates']
        rates_from_trajectory = {key: rate_group[key].attrs['value'] for key in rate_group}

        return trajectory_group['trajectory'][()]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        return False


def write_c3s(filename, system, mode='r+', trajectory_name=None):
    """writes simulation data to an hdf5 file"""

    if system.trajectory is None:
        raise ValueError("no data in `system.trajectory`")
    if not Path(filename).exists():
        with CMEWriter(filename, system, mode='x') as W:
            W._write_system_info()
    with CMEWriter(filename, system, mode=mode) as W:
        W._write_trajectory(trajectory_name=trajectory_name)


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
        reaction_strings = [reaction.reaction for reaction in self.system.reaction_network.reactions]
        rate_names = [reaction.rate_name for reaction in self.system.reaction_network.reactions]
        rate_values = [reaction.rate for reaction in self.system.reaction_network.reactions]
        for rxn, rate_name, rate_value in zip(reaction_strings, rate_names, rate_values):
            rate_group = self._create_group(f'original_config/reactions/{rxn}/{rate_name}')
            self._set_attr(rate_group, name='value', value=rate_value)
        constraint_strings = [constraint.constraint for constraint in self.system.reaction_network.constraints]
        for constraint in constraint_strings:
            constraint_group = self._create_group(f'original_config/constraints/{constraint}')

    def _write_system_info(self):
        # basic reaction info
        self._dump_config()
        self._create_dataset(name='constitutive_states',
                             data=self.system.states)
        self._create_dataset(name='generator_matrix/G_rows',
                             data=self.system.G.lines)
        self._create_dataset(name='generator_matrix/G_columns',
                             data=self.system.G.columns)
        self._create_dataset(name='generator_matrix/G_values',
                             data=self.system.G.values)
        if self.system._initial_populations:
            for species, count in self.system._initial_populations.items():
                self._create_dataset(name=f'initial_populations/{species}',
                                     data=np.array(count))
        if self.system._max_populations:
            for species, count in self.system._max_populations.items():
                self._create_dataset(name=f'max_populations/{species}',
                                     data=np.array(count))

    def _write_trajectory(self, trajectory_name):
        traj_group = self._require_group('trajectories')
        if trajectory_name is None:
            trajectory_name = f'trajectory00{len(traj_group) + 1}'
        traj_dset = self._create_dataset(name=f'trajectories/{trajectory_name}/trajectory',
                             data=self.system.trajectory)
        self._set_attr(parent=traj_dset,
                       name='time',
                       value=self.system.timings[f't_run_{self.system._run_method}'])
        for reaction in self.system.reaction_network.reactions:
            rate_group = self._create_group(f'trajectories/{trajectory_name}/rates/{reaction.rate_name}')
            self._set_attr(parent=rate_group,
                           name='value',
                           value=reaction.rate)

    def _write_mutual_information(self, trajectory_name, X, Y):
        traj_group = self._require_group('trajectories')
        mi_dset = self._create_dataset(name=f'trajectories/{trajectory_name}/mutual_information',
                                       data=self.system._mutual_information)
        self._set_attr(parent=mi_dset,
                       name='X',
                       value=X)

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
