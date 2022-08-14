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
    double x, y, distance_squared;
    long long int In_circle = 0, global = 0;
    long long int num_exp = tosses / world_size;

    if (world_rank > 0)
    {
        // TODO: MPI workers
        unsigned int rand_seed = world_rank*time(NULL);
        for(int i = 0; i < num_exp; i++){
            x =  1.0 * rand_r(&rand_seed)/RAND_MAX;
            y =  1.0 * rand_r(&rand_seed)/RAND_MAX;
            distance_squared= x*x + y*y;
            if(distance_squared <= 1){
                In_circle += 1;
            }
        }
        MPI_Send(&In_circle, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[world_size-1];
        MPI_Status status[world_size-1];
        unsigned int rand_seed = world_rank*time(NULL);
        long long int local[world_size-1];
        for(int i = 0; i < num_exp; i++){
            x =  1.0 * rand_r(&rand_seed)/RAND_MAX;
            y =  1.0 * rand_r(&rand_seed)/RAND_MAX;
            distance_squared= x*x + y*y;
            if(distance_squared <= 1){
                In_circle++;
                }
        }
        for(int i=1;i<world_size;++i)MPI_Irecv(&local[i-1],1,MPI_LONG,i,0,MPI_COMM_WORLD,&requests[i-1]);
        MPI_Waitall(world_size-1,requests,status);
        for(int i = 1;i < world_size; i++){
            global += local[i-1];
        }
        global += In_circle;
    }

    if (world_rank == 0)
    {
        // TODO: PI result
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

