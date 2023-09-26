import c3s
import h5py
import numpy as np
from .sparse_matrix import SparseMatrix


"""reads and writes an HDF5 file with the following format:

(name) is an HDF5 group with a mandatory name
{name} is an HDF5 group with arbitrary name
[name] is an HDF5 dataset with a mandatory name
+-- attribute is an attribute of a group or dataset with a mandatory name
< name > is float or int datatype for attribute, group, or dataset
'name' is string type for attribute

\-- root
    \-- (original_config)
        \-- (reactions)
            \-- {reaction}
                +-- 'rate_name'
                +-- < rate_value >
        \-- (constraints)
            \-- {constraint}
    \-- (intitial_populations)
        \-- {species}
            +-- < count >
    \-- < [constitutive_states] > 
        +-- < code_time >
    \-- (generator_matrix)
        +-- < code_time >
        \-- < [G_rows] >
        \-- < [G_columns] >
        \-- < [G_values] >
    \-- (trajectories)
        \-- {trajectory}
            +-- 'method', 'IMU' or 'EXMP'
            \-- (rates)
                \-- {rate_name}
                    +-- < value >
            \-- < [trajectory] > 
                +-- < dt >
                +-- < code_time >
            \-- (B_matrix), if method=='IMU'
                +-- < Omega >
                +-- < code_time >
                \-- < [B_rows] >
                \-- < [B_columns] >
                \-- < [B_values] >
            \-- < [Q] >, if method=='EXPM"
                +-- < code_time >
            \-- (calculations)
                \-- [mutual_information]
                    +-- X
                    +-- Y
                    +-- base
                \-- {avg_copy_number_species}

"""


def build_system_from_file(filename, mode='r', trajectory_name=None):
    with CMEReader(filename=filename, mode=mode, trajectory_name=trajectory_name) as Reader:
        return Reader.generate_system_from_file()


class CMEWriter:
    """rough draft of a more robust file writer"""

    def __init__(self, filename, mode, system, driver=None, comm=None):
        """
        Parameters
        ----------
        filename : str
            filename for data file
        mode : str
            mode of file access
        system : ChemicalMasterEquation
            ...
        driver : str (default=None)
            file driver used to open hdf5 file
        comm : MPI.Comm (default=None)
            MPI communicator used to open hdf5 file,
            must be passed with `'mpio'` file driver

        """

        self.system = system
        self.filename = filename
        self.mode = mode
        self._driver = driver
        self._comm = comm
        # open file at root
        self._file_root = h5py.File(name=self.filename,
                                    mode=self.mode,
                                    driver=self._driver,
                                    #comm=self._comm,
                                    track_order=True)

    def _write_system(self):
        self._write_original_config()
        self._write_states()
        self._write_generator_matrix()
        if self.system._initial_copy_numbers:
            self._write_initial_copy_numbers()

    def _write_original_config(self):
        """"""
        config_group = self._file_root.create_group('original_config', track_order=True)
        for reaction in self.system.reaction_network.reactions:
            reaction_group = config_group.create_group(f'reactions/{reaction.reaction}', track_order=True)
            reaction_group.attrs['rate_name'] = reaction.rate_name
            reaction_group.attrs['rate_value'] = reaction.rate

        constraint_strings = [constraint.constraint for constraint in self.system.reaction_network.constraints]
        for constraint in constraint_strings:
            constraint_group = config_group.create_group(f'constraints/{constraint}', track_order=True)

    def _write_states(self):
        """"""
        states_dataset = self._file_root.create_dataset(name='constitutive_states', data=self.system.states)
        if self.system.timings:
            states_dataset.attrs['code_time'] = self.system.timings['t_build_state_space']

    def _write_generator_matrix(self):
        """"""
        G_group = self._file_root.create_group('generator_matrix', track_order=True)
        G_group.create_dataset(name='G_rows', data=self.system.G.rows)
        G_group.create_dataset(name='G_columns', data=self.system.G.columns)
        G_group.create_dataset(name='G_values', data=self.system.G.values)
        if self.system.timings:
            G_group.attrs['code_time'] = self.system.timings['t_build_generator_matrix']

    def _write_initial_copy_numbers(self):
        """"""
        initial_pop_group = self._file_root.create_group(name='initial_copy_numbers', track_order=True)
        for species, count in self.system.initial_copy_numbers.items():
            species_group = initial_pop_group.create_group(name=species)
            species_group.attrs['count'] = count

    def _write_trajectory(self, trajectory_name):
        """"""
        Trajectory = self.system.Trajectory

        if 'trajectories' not in self._file_root:
            trajectories_group = self._file_root.create_group('trajectories', track_order=True)
        else:
            trajectories_group = self._file_root.require_group('trajectories')
        if trajectory_name is None:
            trajectory_name = f'trajectory00{len(trajectories_group) + 1}'

        trajectory_group = trajectories_group.create_group(name=trajectory_name, track_order=True)
        trajectory_group.attrs['method'] = Trajectory.method

        for rate_name, rate_value in zip(self.system.reaction_network._rate_names, self.system._rates):
            rate_group = trajectory_group.create_group(f'rates/{rate_name}', track_order=True)
            rate_group.attrs['value'] = rate_value

        trajectory_dataset = trajectory_group.create_dataset(name='trajectory', data=Trajectory.trajectory)
        trajectory_dataset.attrs['dt'] = Trajectory.dt
        if self.system.timings:
            trajectory_dataset.attrs['code_time'] = self.system.timings[f't_run_{Trajectory.method}']

        if Trajectory.method == 'IMU':
            B_group = trajectory_group.create_group('B_matrix', track_order=True)
            B_group.create_dataset(name='B_rows', data=Trajectory.B.rows)
            B_group.create_dataset(name='B_columns', data=Trajectory.B.columns)
            B_group.create_dataset(name='B_values', data=Trajectory.B.values)
            B_group.attrs['Omega'] = Trajectory.Omega
            if self.system.timings:
                B_group.attrs['code_time'] = self.system.timings['t_build_B_matrix']
        elif Trajectory.method == 'EXPM':
            Q_dataset = trajectory_group.create_dataset(name='Q', data=Trajectory.Q)
            if self.system.timings:
                Q_dataset.attrs['code_time'] = self.system.timings['t_matrix_exponential']

    def _write_mutual_information(self, trajectory_name, data, base, X, Y):
        """"""
        trajectory_group = self._file_root.require_group(f'trajectories/{trajectory_name}')
        if 'calculations' not in trajectory_group:
            calculations_group = trajectory_group.create_group('calculations', track_order=True)
        else:
            calculations_group = trajectory_group.require_group('calculations')

        mi_dataset = calculations_group.create_dataset(name='mutual_information', data=data)
        mi_dataset.attrs['X'] = str(X)
        mi_dataset.attrs['Y'] = str(Y)
        mi_dataset.attrs['base'] = base
        if self.system.timings:
            mi_dataset.attrs['code_time'] = self.system.timings['t_calculate_mi']

    def _write_avg_copy_number(self, trajectory_name, species, avg_copy_number):
        trajectory_group = self._file_root.require_group(f'trajectories/{trajectory_name}')
        if 'calculations' not in trajectory_group:
            calculations_group = trajectory_group.create_group('calculations', track_order=True)
        else:
            calculations_group = trajectory_group.require_group('calculations')
        calculations_group.create_dataset(name=f'<c_{species}>', data=avg_copy_number)

    def _write_marginalized_trajectory(self, trajectory_name, species_subset, marginalized_trajectory):
        trajectory_group = self._file_root.require_group(f'trajectories/{trajectory_name}')
        if 'calculations' not in trajectory_group:
            calculations_group = trajectory_group.create_group('calculations', track_order=True)
        else:
            calculations_group = trajectory_group.require_group('calculations')

        marginalized_trajectory_group = calculations_group.create_group(name=f'{species_subset}_trajectory', track_order=True)
        for state, marginalized_trajectory in marginalized_trajectory.items():
            marginalized_trajectory_group.create_dataset(name=str(state), data=marginalized_trajectory)

    def close(self):
        self._file_root.close()

    # define __enter__ and __exit__ so this class can be used as a context manager
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class CMEReader:
    """rough draft of a more robust file reader"""

    def __init__(self, filename, mode, driver=None, comm=None, trajectory_name=None):
        """
        Parameters
        ----------
        filename : str
            filename for data file
        mode : str
            mode of file access
        driver : str (default=None)
            file driver used to open HDF5 file
        comm : MPI.Comm (default=None)
            MPI communicator used to open hdf5 file, must be passed with `'mpio'` file driver
        trajectory_name : str (default=None)
            ...

        """

        self.filename = filename
        self.mode = mode
        self._driver = driver
        self._comm = comm
        self.trajectory_name = trajectory_name
        # open file at root
        self._file_root = h5py.File(name=self.filename,
                                    mode=self.mode,
                                    driver=self._driver)
                                    #comm=self._comm)

    def generate_system_from_file(self):
        # first get reaction network info
        config_dictionary = self._read_original_config()
        if 'initial_copy_numbers' in self._file_root:
            initial_copy_numbers = {species: species.attrs['count'] for species in self._file_root['initial_copy_numbers']}
        else:
            initial_copy_numbers = None
        # start with an empty system
        system = c3s.ChemicalMasterEquation(config=config_dictionary,
                                            initial_copy_numbers=initial_copy_numbers,
                                            empty=True)
        # fill in attribute data from file
        system._constitutive_states = self._read_states()
        system._generator_matrix = self._read_G()
        if self.trajectory_name is not None:
            system._Trajectory = self._read_trajectory(self.trajectory_name)

        return system

    def _read_original_config(self):
        config_dictionary = {'reactions': {}, 'constraints': {}}
        config_group = self._file_root['original_config']
        for key in config_group['reactions'].keys():
            rate_name = config_group[f'reactions/{key}'].attrs['rate_name']
            rate_value = config_group[f'reactions/{key}'].attrs['rate_value']
            config_dictionary['reactions'][key] = [rate_name, rate_value]
        config_dictionary['constraints'] = list(config_group['constraints'].keys())
        return config_dictionary

    def _read_states(self):
        return self._file_root['constitutive_states'][()]

    def _read_G(self):
        G_rows = self._file_root['generator_matrix/G_rows'][()]
        G_columns = self._file_root['generator_matrix/G_columns'][()]
        G_values = self._file_root['generator_matrix/G_values'][()]
        return SparseMatrix(G_rows, G_columns, G_values)

    def _read_trajectory(self, trajectory_name):

        trajectory_group = self._file_root[f'trajectories/{self.trajectory_name}']

        trajectory = trajectory_group['trajectory'][()]
        method = trajectory_group.attrs['method']
        dt = trajectory_group['trajectory'].attrs['dt']
        rates = np.array(
            [trajectory_group[f'rates/{rate_name}'].attrs['value'] for rate_name in trajectory_group['rates']])

        if method == 'IMU':
            B_group = trajectory_group['B_matrix']
            Omega = B_group.attrs['Omega']
            B = SparseMatrix(B_group['B_rows'][()], B_group['B_columns'][()], B_group['B_values'][()])
            Q = None
        else:
            assert(method=='EXPM')
            Q = trajectory_group['Q'][()]
            Omega = None
            B = None

        return c3s.ChemicalMasterEquation.CMETrajectory(
            trajectory=trajectory, method=method, dt=dt, rates=rates,Omega=Omega, B=B, Q=Q)

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file_root.close()
        return False
