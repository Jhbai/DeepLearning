# include <stdio.h>
# include <stdlib.h>
# include <pthread.h>
# include <iostream>

long thread_counter;
long long iteration;
double PI;

pthread_mutex_t mutex;

void* Thread_sum(void* rank);

int  main(int argc, char* argv[]){
        pthread_t *thread_handles;
        long thread = 0;

        thread_counter = strtol(argv[1], NULL, 10);
        iteration = strtol(argv[2], NULL, 10);
        thread_handles = (pthread_t*) malloc (thread_counter*sizeof(pthread_t));

        pthread_mutex_init(&mutex, NULL);
        PI = 0.0;
        for(thread = 0; thread < thread_counter; thread++)

		                                pthread_create(&thread_handles[thread], NULL, Thread_sum, (void*) thread);
        for(thread = 0; thread < thread_counter; thread ++)
                pthread_join(thread_handles[thread], NULL);

        PI = 4.0*PI;
        pthread_mutex_destroy(&mutex);
        free(thread_handles);
        std::cout << PI << std::endl;
        return 0;
}

void *Thread_sum(void *rank){
        long my_rank = (long) rank;
        double factor, my_pi = 0.0;
        long long i;
        long long my_n = iteration/thread_counter;
        long long my_first_i = my_n*my_rank;
        long long my_last_i = my_first_i + my_n;
        if (my_rank == thread_counter)my_last_i = iteration;

        if(my_first_i % 2 == 0)factor = 1.0;
        else factor = -1.0;

        for(i = my_first_i; i < my_last_i; i++, factor = -factor)
                my_pi += factor/(2*i + 1);
        pthread_mutex_lock(&mutex);
        PI += my_pi;
        pthread_mutex_unlock(&mutex);

        return NULL;
}

