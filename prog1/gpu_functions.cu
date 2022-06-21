
/**
 * @brief Applies the Gauss Elimination Formula
 * 
 * @param kj 
 * @param ki_ii 
 * @param ij 
 */
__device__ void formulaGPU (double *kj, double ki_ii, double ij)
{
    *kj = *kj - (ki_ii * ij);
}


/**
 * @brief Swap columns on gpu
 * 
 * @param mat 
 * @param x 
 * @param y 
 */
__device__ void swapColsOnGPU(double *mat, int x, int y){
    
    double aux = mat[x];
    mat[x] = mat[y];
    mat[y] = aux;
}


/**
 * @brief Calculate the determinats of matrices on GPU
 * 
 * @param mat 
 * @param determinant_results 
 * @param sector_size 
 */
__global__ static void determinantOnGPU (double * __restrict__ mat, double * __restrict__ determinant_results,  int sector_size)
{   
 
    int bx = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
    int idx = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

    mat += bx * sector_size * sector_size;

    
    for (int j = 0; j<=idx; j++) {

        /* check if diagonal is 0 to change column*/
        if (mat[j*sector_size + j] == 0) {
            int colSwap = -1;

            for (int i=j+1; i<sector_size; i++) {
                if (mat[j*sector_size + i] != 0.0) {
                colSwap = i;
                }
            }

            if (colSwap == -1) {
                determinant_results[bx] = 0;
                break;
            } else {
                swapColsOnGPU(mat, j + idx * sector_size, colSwap + idx * sector_size);
            }

            __syncthreads(); 
        }

        // Apply formula process
        for (int i = j+1; i<sector_size; i++) {
            
            /*get the division value before apply on formula*/
            double division =  mat[i*sector_size + j] / mat[j*sector_size + j];

            __syncthreads(); 

            // Apply formula
            formulaGPU(&mat[i*sector_size + idx], division, mat[j*sector_size+idx]);
        }
        
        if (idx == j) {
            determinant_results[bx] = determinant_results[bx] * mat[ (idx*sector_size) + idx ];
        }
    }

}

