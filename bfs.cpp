#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances,
    int *DEPTH)
{
        int COUNT = 0; // This is shared data
        int LIST[4]={0}; // A shared array with first element 0
#pragma omp parallel
        {
#pragma parallel for
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];
        // Avoid out of range
        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];
            int index;
            if(distances[outgoing] == NOT_VISITED_MARKER) //parents[v] = -1
                distances[outgoing]= *DEPTH + 1;
            }
        }
    int count =0; // private data
    int id = omp_get_thread_num();
    #pragma omp for
    for(int i = 0; i< g->num_nodes;++i)if(distances[i]== (*DEPTH)+1)count++;
    LIST[id] = count;
    #pragma omp barrier

    #pragma omp atomic
    COUNT += count;
    int START = 0;

    for(int i =0; i < id ; ++i){
        START += LIST[i];
    }
    #pragma omp for
    for(int i=0 ;i<g->num_nodes;++i){
        if(distances[i]== (*DEPTH)+1){
            new_frontier->vertices[START]=i;
            START++;
        }
    }


    }

     new_frontier->count = COUNT;


}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{
    int DEPTH = 0;
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances, &DEPTH);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        DEPTH++;
    }
}
void bottom_up_step(
    Graph g,
    vertex_set *frontier,
    int *distances,
    vertex_set *new_frontier,
    int *DEPTH)
{
    // Global data
    int LIST[4]={0};
    int COUNT=0;

    // create the parallel region
    #pragma omp parallel
    {
        #pragma omp for
    for(int i =0 ;i < g -> num_nodes;++i){

        // if parents[v] = -1 then
        if(distances[i] == NOT_VISITED_MARKER){
            // take neighbor
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                           ? g->num_edges
                           : g->incoming_starts[i + 1];
            // for n in nighbors[v] do
            for(int j= start_edge ; j<end_edge ; ++j){

                // parents[v] <- n
                int k = g->incoming_edges[j];
                int index;
                if(distances[k] == *DEPTH){

                    distances[i] = (*DEPTH)+1;

                    break;
                }
            }
      }

    }



    // Since we want to parallelize next <- union(next, {v}), we seperate this
    // part to deal with it.
    int count =0;
    // default thread, system max thread = 4
    int id = omp_get_thread_num();

    // Each thread is independent
    #pragma omp for
    for(int i = 0; i< g->num_nodes;++i)if(distances[i]== (*DEPTH)+1)count++;
    LIST[id] = count;
    #pragma omp barrier

    #pragma omp atomic
    COUNT += count;

    int START=0;

    for(int i =0; i < id ; ++i)START += LIST[i];

    #pragma omp for
    for(int i=0;i<g->num_nodes;++i){
        if(distances[i]==(*DEPTH)+1){
            new_frontier->vertices[START]=i;
            START++;
        }
    }


    }
     // union
     new_frontier->count = COUNT;
}
void bfs_bottom_up(Graph graph, solution *sol)
{
    // According to the top-down sample code
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    // create frontier which will be used in top-down step and bottom-up step
    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // New parameter needed
    int DEPTH=0;

    #pragma omp parellel for
    for (int i = 0; i < graph->num_nodes; i++)sol->distances[i] = NOT_VISITED_MARKER;
    // initialize the root, silimar to the sample code
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;


    // while frontier !={} do
    while(frontier->count!=0){

        // do the bottom_up step
        vertex_set_clear(new_frontier);
        bottom_up_step(graph,frontier,sol->distances,new_frontier,&DEPTH);

        // swap the pointer
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        DEPTH++;

    }
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // initialization is similar to the sample code
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // new parameters to assign and deal with the step function
    int DEPTH =0;
    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;


    while(frontier->count!= 0){

        // The algorithm of hybrid bfs
        int mf = 0 ;
        int nf = frontier->count;
        int mu = 0;



        vertex_set_clear(new_frontier);
        if(nf < graph->num_nodes /24 ){
            top_down_step(graph,frontier,new_frontier,sol->distances,&DEPTH);
        }else{
            bottom_up_step(graph,frontier,sol->distances,new_frontier,&DEPTH);
        }

        // swap pointer
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        DEPTH++;

    }
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}

