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

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    double x,y,distance_squared;
    long long int In_circle = 0;
    long long int global = 0;
    long long int num_exp = tosses / world_size;



    if (world_rank == 0)
    {
        // Master
        unsigned int rand_seed = world_rank*time(NULL);
        // Using time to create random values.
        long long int local;
        MPI_Alloc_mem(sizeof(long long int), MPI_INFO_NULL, &local);
        for(int i = 0; i < num_exp; i++){
                x =  1.0 * rand_r(&rand_seed)/RAND_MAX;
                y =  1.0 * rand_r(&rand_seed)/RAND_MAX;
                // Count the cardinlity of points in the circle
                distance_squared = x*x + y*y;
                if(distance_squared <= 1){
                        In_circle += 1;
                    }
        }
        global += In_circle;
        MPI_Win_create(NULL, sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        // Without the lock/unlock schedule stays forever filled with 0s
        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
        //global += local;
        MPI_Win_unlock(0, win);


    }
    else
    {
        // Workers
        unsigned int rand_seed = world_rank*time(NULL);
        long long int buffer = 0;
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        // Using time to create random values.
        for(int i = 0; i < num_exp; i++){
                x =  1.0 * rand_r(&rand_seed)/RAND_MAX;
                y =  1.0 * rand_r(&rand_seed)/RAND_MAX;
                // Count the cardinlity of points in the circle
                distance_squared = x*x + y*y;
                if(distance_squared <= 1){
                        buffer += 1;
                    }
        }
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        //MPI_Put(&buffer, 1, MPI_LONG, 0, world_rank, 1, MPI_LONG, win);
        MPI_Win_unlock(0, win);
    }

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4*world_size*((double)global)/((double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}

