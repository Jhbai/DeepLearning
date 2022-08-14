#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int mandel(float C1, float C2, int count)
{
        float Z1 = C1, Z2 = C2;
        int i;
        for (i = 0; i < count; i++)
        {
                if (Z1 * Z1 + Z2 * Z2 > 4.f)
                break;

                float local1 = Z1 * Z1 - Z2 * Z2;
                float local2 = 2.f * Z1 * Z2;
                Z1 = C1 + local1;
                Z2 = C2 + local2;
        }

        return i;
}


__global__ void mandelKernel(int *devices, float X0, float Y0, float dX, float dY, int restX, int restY, int Maxiterations){
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if(i >= restX || j >= restY) return;

        float tempX = X0 + i * dX, tempY = Y0 + j * dY;
        int I = restX*j + i;
        devices[I] = mandel(tempX, tempY, Maxiterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
	// Default
        float stepX = (upperX - lowerX) / resX;
        float stepY = (upperY - lowerY) / resY;

	// TODO
        int Xblocks = (int) ceil(resX/16.0);
        int Yblocks = (int) ceil(resY/16.0);
        dim3 TPB(16, 16);
        dim3 NT(Xblocks, Yblocks);
        int *DEVICE;
        int size = resX*resY*sizeof(int);
	
	// CUDA
        cudaMalloc(&DEVICE, size);
        mandelKernel <<< NT, TPB >>> (DEVICE, lowerX, lowerY, stepX, stepY, resX, resY, maxIterations);
        cudaMemcpy(img, DEVICE, size, cudaMemcpyDeviceToHost);
        cudaFree(DEVICE);
}
                                                               
