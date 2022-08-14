#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    double x,y,distance_squared;
    long long int In_circle = 0;
    long long int global = 0, local[world_size];
    long long int num_exp = tosses / world_size;
    // TODO: use MPI_Gather
    unsigned int rand_seed = world_rank*time(NULL);
        for(int i = 0;i < num_exp; i++){
            x =  1.0 * rand_r(&rand_seed)/RAND_MAX;
            y =  1.0 * rand_r(&rand_seed)/RAND_MAX;
            distance_squared= x*x + y*y;
            if(distance_squared <= 1){
                In_circle += 1;
            }
        }

    MPI_Gather(&In_circle,1,MPI_LONG,local,1,MPI_LONG,0,MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // TODO: PI result
        for(int i = 0; i < world_size; i++)global += local[i];
        pi_result = 4 *global/((double)tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}

