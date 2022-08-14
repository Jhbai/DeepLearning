#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---
    double x, y, distance_squared;
    long long int In_circle = 0, global = 0;
    // TODO: init MPI
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank); // Return the rank of process in a group to world_rank(the number order of processes)
    MPI_Comm_size(MPI_COMM_WORLD,&world_size); // Assign number of processes in a MPI comm to world_size(the total number of processes)
    long long int num_exp = tosses / world_size;
    if (world_rank > 0)
    {
        // TODO: handle workers
        unsigned int rand_seed = world_rank*time(NULL);
        // Using time to create random values.
        for(int i = 0; i < num_exp; i++){
                // create a square whose both length and width are 2
                // Moreover the central of circle is (1,1), the radius is 1
                // We prefer to get a random percentage, therefore we compute
                // this random value devided by RAND_MAX
                x =  1.0 * rand_r(&rand_seed)/RAND_MAX;
                y =  1.0 * rand_r(&rand_seed)/RAND_MAX;
                // Count the cardinlity of points in the circle
                distance_squared = x*x + y*y;
                if(distance_squared <= 1){
                        In_circle += 1;
                    }
        }
        // Message Passage -- Send
        MPI_Send(&In_circle, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: master
        unsigned int rand_seed = world_rank*time(NULL);
        // Using time to create random values.
        long long int local;
        for(int i = 0; i < num_exp; i++){
                // create a square whose both length and width are 2
                // Moreover the central of circle is (1,1), the radius is 1
                // We prefer to get a random percentage, therefore we compute
                // this random value devided by RAND_MAX
                x =  1.0 * rand_r(&rand_seed)/RAND_MAX;
                y =  1.0 * rand_r(&rand_seed)/RAND_MAX;
                // Count the cardinlity of points in the circle
                distance_squared = x*x + y*y;
                if(distance_squared <= 1){
                        In_circle += 1;
                    }
    }
        // Message Passage -- Receive
        for(int i = 1; i < world_size; i++){
            MPI_Recv(&local, 1, MPI_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            global += local;
        }
        global += In_circle;
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        pi_result = 4*((double)global)/((double)tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}

