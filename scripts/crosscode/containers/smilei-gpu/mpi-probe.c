/* Run by the smilei-gpu build: a singleton MPI start as Smilei makes it (MPI_Init_thread with
   MPI_THREAD_MULTIPLE, src/SmileiMPI/SmileiMPI.cpp), so a broken MPI stops the build instead of
   the first run. Needs no GPU. */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    int provided = 0, rank = -1, size = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("mpi singleton: rank %d of %d, thread level %d (multiple = %d)\n", rank, size, provided, MPI_THREAD_MULTIPLE);
    MPI_Finalize();
    return (size == 1 && provided == MPI_THREAD_MULTIPLE) ? 0 : 1;
}
