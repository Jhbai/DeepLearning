__kernel void convolution(__global float *Input, __global float *Output, __global float *filter, int Height, int Width, int FS) 
{
    int I = get_global_id(0);
    int row = I/Width, col = I%Width; 
    int HF = FS/2;
    float result = 0.0;
    for (int i = (-1)*HF; i <= HF; ++i){
        for (int j = (-1)*HF; j <= HF; ++j){
            if(filter[(i + HF) * FS + j + HF] != 0){
	        int A = row + i;
		int B = col + j;
                if ((A >= 0) && (A < Height) && (B >= 0) && (B < Width)){
		    int i1 = A*Width + B, i2 = (A - row + HF) * FS + B - col + HF;
                    result += Input[i1] * filter[i2];
		}
            }
        }
    }
    Output[I] = result;
}
