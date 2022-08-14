#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"
#include "omp.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs
  //Parallel form

  int numNodes = num_nodes(g);
  double *score_old = (double *)malloc(numNodes*sizeof(double));
  double *score_new = (double *)malloc(numNodes*sizeof(double));
  double equal_prob = 1.0 / numNodes;
  double converged = 1.0;

  for(int i = 0; i < numNodes; i++){
          score_old[i] = equal_prob;
  }

  while(converged > convergence){
          double TEMP = 0;
#pragma omp parallel for
          for(int i = 0; i < numNodes; i++){
                  const int *in_vertex = incoming_begin(g, i);
                  const int *out_vertex = outgoing_end(g, i);

                  score_new[i] = 0;
                  for(int j = 0; j < incoming_size(g, i); j++){
                          double A = score_old[*(in_vertex + j)];
                          double B = outgoing_size(g, *(in_vertex + j));
                          score_new[i] += A/B;
                  }


                  //score_new[vi] = score_old[vj] / number of edges leaving vj 

                  score_new[i] = damping * score_new[i] + (1.0 - damping)/numNodes;

                  //score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;
                  if(outgoing_size(g, i) == 0){
#pragma omp critical(A)
                          TEMP += damping * score_old[i]/numNodes;
                  //score_new[vi] += damping * score_old[v] / numNodes
                  }
          }

          converged = 0.0;
#pragma omp parallel for reduction(+: converged)
          for(int i = 0; i < numNodes; i++){
                  score_new[i] += TEMP;
                  converged += abs(score_old[i] - score_new[i]);
                  score_old[i] = score_new[i];
          }
}

  for(int i = 0; i < numNodes; i++){
          solution[i] = score_old[i];
  }
  free (score_old);
  free (score_new);
  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}

