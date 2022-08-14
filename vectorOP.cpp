#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // In0verse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  // Define the vector will be used afterward.
  __pp_vec_float x;
  __pp_vec_int y;
  __pp_vec_float result;
  __pp_vec_int zeros = _pp_vset_int(0);
  __pp_vec_int ones = _pp_vset_int(1);
  __pp_vec_float ones2 = _pp_vset_float(1);
  __pp_vec_float nines = _pp_vset_float(9.999999);
  __pp_mask ALL, ZERO, NOTZERO, NINE, NOTNINE;
  // find max times
  int MAX = 0;
  int P = 0;
  for(int i = 0; i < N; i++)if(MAX <= exponents[i])MAX = exponents[i];
  // for each vectorwidth, we take the amount of data into the register.
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
          P = i;
	  if(N - i < VECTOR_WIDTH){
          break;}
          // assign the mask.
          ALL = _pp_init_ones();
          ZERO = _pp_init_ones(0);
          NOTZERO = _pp_init_ones(0);
          NINE = _pp_init_ones(0);
          NOTNINE = _pp_init_ones(0);
          // assign the value in the vector.
          _pp_vload_float(x, values + i, ALL);
          _pp_vload_int(y, exponents+ i, ALL);
          _pp_vset_float(result, 1, ALL);
          // Is zero exponents?
          _pp_veq_int(ZERO, y, zeros, ALL);
          NOTZERO = _pp_mask_not(ZERO);
          // If zero, then change it 1.
          _pp_vmove_float(result, ones2, ZERO);
          // multiply times
          for(int j = 0; j < MAX; j++){
                  _pp_vmult_float(result, result, x, NOTZERO);
                  _pp_vsub_int(y, y, ones, NOTZERO);
                  _pp_veq_int(ZERO, y, zeros, ALL);
                  NOTZERO = _pp_mask_not(ZERO);
        }
         _pp_vgt_float(NINE, result, nines, ALL);
         NOTNINE = _pp_mask_not(NINE);
         _pp_vmove_float(result, nines, NINE);
        _pp_vstore_float(output + i, result, ALL);
  }
  for(int i = P; i < N; i ++){
	  float temp = 1;
          for(int j = 0; j < exponents[i]; j ++)temp *= values[i];
          if(temp >= 9.999999)temp = 9.999999;
          output[i] = temp;
     }
}
// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  float TEMP[VECTOR_WIDTH];
  __pp_vec_float result = _pp_vset_float(0.f);
  __pp_vec_float data;
  __pp_mask ALL;
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
	ALL = _pp_init_ones();
	_pp_vload_float(data, values + i, ALL);
	_pp_vadd_float(result, result, data, ALL);
  }
  _pp_vstore_float(TEMP, result, ALL);
  float RESULT = 0;
  for(int i = 0; i < VECTOR_WIDTH; i++)RESULT += TEMP[i];

  return RESULT;
}
