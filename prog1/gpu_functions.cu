
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
 * @param order_matrix 
 */
__global__ static void determinantOnGPU (double * __restrict__ mat, double * __restrict__ determinant_results,  int order_matrix)
{   
 
    int bx = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
    int idx = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

    mat += bx * order_matrix * order_matrix;

    
    for (int j = 0; j<=idx; j++) {

        /* check if diagonal is 0 to change column*/
        if (mat[j*order_matrix + j] == 0) {
            int colSwap = -1;

            for (int i=j+1; i<order_matrix; i++) {
                if (mat[j*order_matrix + i] != 0.0) {
                colSwap = i;
                }
            }

            if (colSwap == -1) {
                determinant_results[bx] = 0;
                break;
            } else {
                swapColsOnGPU(mat, j + idx * order_matrix, colSwap + idx * order_matrix);
            }

            __syncthreads(); 
        }

        // Apply formula process
        for (int i = j+1; i<order_matrix; i++) {
            
            /*get the division value before apply on formula*/
            double division =  mat[i*order_matrix + j] / mat[j*order_matrix + j];

            __syncthreads(); 

            // Apply formula
            formulaGPU(&mat[i*order_matrix + idx], division, mat[j*order_matrix+idx]);
        }
        
        if (idx == j) {
            determinant_results[bx] = determinant_results[bx] * mat[ (idx*order_matrix) + idx ];
        }
    }

}

