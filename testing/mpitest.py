from mpi4py import MPI
import c3s


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

system = c3s.simulators.MasterEquation(cfg='config_files/binary.yml', A=1)

print(f'rank: {rank} has rates {system.rates} \n {system.G} \n')

if rank == 0:
    system.update_rates(k_1=2, k_2=2)
comm.Barrier()
print(f'after updating, rank: {rank} has rates {system.rates} \n {system.G} \n')

if rank == 1:
    system.update_rates(k_1=2, k_2=2)
comm.Barrier()
print(f'after updating, rank: {rank} has rates {system.rates} \n {system.G} \n')
