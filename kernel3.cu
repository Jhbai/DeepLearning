#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Executed on Device, Call on Device:
__device__ int mandel(float C1, float C2, int count){
	float Z1 = C1, Z2 = C2;
	// TODO
	// Extended ANSI C
	// Each iteration as one thread.
	int i;
	for(i = 0; i < count; i++){
		if((Z1*Z1 + Z2*Z2) > 4.f)break;
		float local1 = Z1*Z1 - Z2*Z2;
		float local2 = 2.f*Z1 * Z2;
		Z1 = C1 + local1;
		Z2 = C2 + local2;
	}
	return i;
}
// Executed on Device, Call on Host 
__global__ void mandelKernel(int *devices, float X0, float Y0, float dX, float dY, int rest_X, int rest_Y, int MaxIteration, int Group) {
    // TODO : New thing is To organize the group size cut
    // To avoid error caused by the floating number, use the following pseudo code
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int i, j;
    // threadIdx : the index of threads
    // blockIdx : the index of block of threads
    // blockDim : the size of each thread blocks
    // dx = (x1 - x0)/width; dy = (y1 - y0)/length;
    // here compute i and j
    i = (threadIdx.x + blockIdx.x * blockDim.x)*Group;
    j = (threadIdx.y + blockIdx.y * blockDim.y)*Group;
    // dx = (x1 - x0)/width; dy = (y1 - y0)/length;
    // X = x + row*dx, Y = y + col*dy;
    int I, J;
    float tempx, tempy;
    for(I = i; I < i + Group; I++){
	    for(J = j; J < j + Group; J++){
                if(I >= rest_X || J >= rest_Y){continue;}
                tempx = X0 + I*dX;
                tempy = Y0 + J*dY;
    // mandel can be called by device. (the actual execute result)
                int result = mandel(tempx, tempy, MaxIteration);
                devices[rest_X*J + I] = result;
		}
    }
}

//// Host code
// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
 
    // TODO : Use cudaMallocPitch and cudaHostAlloc and try different size
    int *DEVICE;
    dim3 TPB(16, 16);
    int G = 4;
    // resX = 1600, rexY = 1200
    dim3 NB(ceil(resX*1.0/(TPB.x*G)), ceil(resY*1.0/(TPB.y*G)));
    int *HOST;
    // Define for cudaMallocPitch and cudaHostAlloc
    size_t pitch;

    // allocate cude memory
    //cudaMalloc(&DEVICE, resX * resY * sizeof(int));
    cudaHostAlloc(&HOST, sizeof(int)*resX*resY, cudaHostAllocMapped);
    cudaMallocPitch(&DEVICE, &pitch, resX*sizeof(int), resY);
    // call the global function, which is computing the result
    mandelKernel <<< NB, TPB >>> (DEVICE, lowerX, lowerY, stepX, stepY, resX, resY, maxIterations, G);
    // the cuda's synchronization
    cudaDeviceSynchronize();
    // move the result from GPU to CPU
    cudaMemcpy(HOST, DEVICE, resX*resY*sizeof(int), cudaMemcpyDeviceToHost);
    
    // Put the result in the output
    cudaMemcpy(img, HOST, resX*resY*sizeof(int), cudaMemcpyHostToHost);

    cudaFree(DEVICE);
    cudaFreeHost(HOST);
}
