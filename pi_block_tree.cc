#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

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
    // The initialization of MPI library
    MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    // As the prvious Homework, we set two random point and get a squared distance to identify whether the point is in the circle or not.
    double x,y,distance_squared;
    long long int In_circle = 0, global = 0, local = 0, temp; // temp is for MPI_Recv
    long long int num_exp = tosses / world_size;
    // Using the assume of power-of-two number of processes
    int k = 0; // the total processes for the MPI below
    int size = world_size;
    while(size != 1){
        size /= 2;
        k++;
    }

    // TODO: binary tree redunction
    // Count the point in the circle
    unsigned int rand_seed = world_rank*time(NULL);
    for(int i = 0; i < num_exp; i++){
            x =  1.0 * rand_r(&rand_seed)/RAND_MAX;
            y =  1.0 * rand_r(&rand_seed)/RAND_MAX;
            distance_squared= x*x + y*y;
            if(distance_squared <= 1){
                In_circle++;
            }
            local = In_circle;
    }
    // The logic is as the code in the pi_block_linear.

    // Redunction
    for(int i = 0;i < k;i++){
        if(world_rank % ((int)pow(2,(i+1)))==0){
            MPI_Recv(&temp, 1, MPI_LONG, world_rank + pow(2, i), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local += temp;
        }
        else if(world_rank%(int)pow(2,(i+1))==(int)pow(2,i))MPI_Send(&local, 1, MPI_LONG,world_rank-pow(2,i), 0, MPI_COMM_WORLD);
    }
    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 *local/((double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
