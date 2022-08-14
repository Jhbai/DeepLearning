#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
	// parameters setting
	cl_int status;
	int FS = filterWidth * filterWidth;
	int size = imageHeight * imageWidth;
	
	// command queue create
	cl_command_queue Queue = clCreateCommandQueue(*context, *device, 0, NULL);

	// buffer create
	cl_mem Input =  clCreateBuffer(*context, CL_MEM_READ_ONLY,  size*sizeof(float), NULL, NULL);
	cl_mem Filter = clCreateBuffer(*context, CL_MEM_READ_ONLY,  FS*sizeof(float), NULL, NULL);
	cl_mem Output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, size*sizeof(float), NULL, NULL);

	// kernel create
	cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

	// kernel parameters enqueue
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&Input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&Output);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&Filter);
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&imageHeight);
	clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&imageWidth);
	clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&filterWidth);

	// buffer enqueue
	clEnqueueWriteBuffer(Queue, Input, CL_TRUE, 0,size*sizeof(float), inputImage, 0, NULL, NULL);
	clEnqueueWriteBuffer(Queue, Filter, CL_TRUE, 0,FS*sizeof(float), filter, 0, NULL, NULL);

	// ND Range setting (that is data size)
	size_t Global_item_size = size;
	size_t local_item_size = 64;
	clEnqueueNDRangeKernel(Queue, kernel, 1, NULL, &Global_item_size, &local_item_size, 0, NULL, NULL);

	// Device to Host
	clEnqueueReadBuffer(Queue, Output, CL_TRUE, 0, size*sizeof(float), outputImage, 0, NULL, NULL);

	// Free Memory
	clReleaseCommandQueue(Queue);
	clReleaseCommandQueue(Input);
	clReleaseCommandQueue(Output);
	clReleaseCommandQueue(kernel);
	
}

