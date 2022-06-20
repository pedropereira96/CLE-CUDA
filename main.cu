#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../common.h"


// Swaps the values of columns X and Y
__device__ void swap_columns(double *matrix, int x, int y, int size)
{
    __syncthreads();

    for (int i = 0; i < size; i++)
    {
        double tmp = matrix[size * i + x];
        matrix[size * i + x] = matrix[size * i + y];
        matrix[size * i + y] = tmp;
    }
}

// Applies Gaussian Elimination
__device__ void formula(double *kj, double ki, double ii, double ij)
{
    *kj = *kj - ((ki / ii) * ij);
}



__global__ void determinantOnGPU(double *matrix, double *determinant, int count, int order)
{
    // matrix
    int matrix_id = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;

    // row_id
    int row_id = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;

    // mat
    double *mat = matrix + (matrix_id * order * order);

    // row
    double *row = mat + (row_id * order);

    double *ii = row + row_id;
    unsigned int signal_reversion = 0;

    printf("-----%d-> *ii=%.3f\n",row_id, *ii);

/*
    if (*ii == 0.0)
    {
        for (double *i = ii + 1; i < ii + order; i++)
        {
            if (*i != 0.0)
            {
                swap_columns(mat, *ii, *i, order);
                signal_reversion = !signal_reversion;
                break;
            }
        }
    }*/

    for (int j = order - 1; j > row_id - 1; j--)
    {
        for (int k = row_id + 1; k < order; k++)
        {   
            printf("Aply formula thread(%d) = mat[%d] = %.3f\t,%.3f, %.3f, %.3f\n",threadIdx.x, order * k + j, mat[order * k + j], mat[order * k + row_id], *ii, mat[order * row_id + j]);
            formula(&mat[order * k + j], mat[order * k + row_id], *ii, mat[order * row_id + j]);
        }
    }

    if (*ii == 0)
    {
        determinant[matrix_id] = 0;
    }
    else
    {
        determinant[matrix_id] *= (*ii);
        printf("*ii=%.3f\n", *ii);
    }
    printf("determinant = %.3e\n", determinant[matrix_id] );
    
}


int main (int argc, char **argv)
{
  int number_of_matrix;
  int order_matrix;

    char *fName = argv[1];
                /*
                Os comentários com : TIRAR PARA LER AS MATRIZES 
                Servem para apenas fazer teste com matriz especifica.


                PAra testar com ficheiros, é necessário descomentar e remover/comentar os desnecessários

                */


/*/////////////////////TIRAR PARA LER DO FICHEIRO////////////////////////////////
  FILE * file;
  file = fopen(fName, "rb");

  if (file == NULL) {
      perror("File could not be read");
      exit(1);
  }


  
  // read number of matrix
  if(fread(&number_of_matrix, sizeof(int), 1, file) != 1)
     {
        perror("Error reading number of matrices!");
        exit(1);
    }
  printf("Number of matrices to be read = %i \n", number_of_matrix);   

  // read orders
  if(fread(&order_matrix, sizeof(int), 1, file) != 1)
   {
        perror("Error reading matrices order!");
        exit(1);
    }
  printf("Matrices order = %i \n", order_matrix);
*/

number_of_matrix = 1;
order_matrix = 4;
    

   // get device information 
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK (cudaGetDeviceProperties (&deviceProp, dev));
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  CHECK (cudaSetDevice (dev));


  int mat_size = order_matrix * order_matrix * sizeof(double);
  size_t mat_area_size = number_of_matrix * mat_size;
  double *h_mat, *h_determinants;
  double *d_mat, *d_determinants;

  printf ("Total mat size: %d\n", (int) mat_area_size);
  
  h_mat = (double *) malloc (mat_area_size);
  h_determinants = (double *) malloc (number_of_matrix*sizeof(double));
  CHECK (cudaMalloc ((void **) &d_mat, mat_area_size));
  CHECK (cudaMalloc ((void **) &d_determinants, number_of_matrix*sizeof(double)));


/*

    h_mat[0] = 0;
    h_mat[1] = 3;
    h_mat[2] = 2;
    h_mat[3] = 0;
    h_mat[4] = 3;
    h_mat[5] = 1;
    h_mat[6] = 0;
    h_mat[7] = 2;
    h_mat[8] = 2;
    h_mat[9] = 3;
    h_mat[10] = 0;
    h_mat[11] = 1;
    h_mat[12] = 0;
    h_mat[13] = 2;
    h_mat[14] = 1;
    h_mat[15] = 3;
    */

    h_mat[0] = 0;
    h_mat[1] = 1;
    h_mat[2] = 2;
    h_mat[3] = 3;
    h_mat[4] = 4;
    h_mat[5] = 5;
    h_mat[6] = 6;
    h_mat[7] = 7;
    h_mat[8] = 8;
    h_mat[9] = 9;
    h_mat[10] = 10;
    h_mat[11] = 11;
    h_mat[12] = 12;
    h_mat[13] = 13;
    h_mat[14] = 14;
    h_mat[15] = 15;
/*////////////TIRAR PARA LER AS MATRIZES /////////////////////////////////////////////////////////////////////////////////
  if(fread(h_mat, mat_area_size, 1, file) != 1)
      strerror(1);
*/



  
  for (int i = 0; i<number_of_matrix; i++) {
    h_determinants[i] = 1;
  }


// copy matrices to gpu memory
  CHECK (cudaMemcpy (d_mat, h_mat, mat_area_size, cudaMemcpyHostToDevice));
// copy determinants to gpu memory
  CHECK (cudaMemcpy (d_determinants, h_determinants, number_of_matrix * sizeof(double), cudaMemcpyHostToDevice));



  for (int col = 0 ; col < order_matrix * order_matrix ;col++){

    if (col % order_matrix == 0) printf("\n");
    printf("%.3f\t", h_mat[col ]);
    }




    unsigned int gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ;
    int n_sectors, sector_size;

    n_sectors = number_of_matrix * order_matrix;
    sector_size = order_matrix;
    blockDimX = order_matrix;
    blockDimY = 1 << 0;                                             // optimize!
    blockDimZ = 1 << 0;                                             // do not change!
    gridDimX = number_of_matrix;
    gridDimY = 1 << 0;                                              // optimize!
    gridDimZ = 1 << 0;                                              // do not change!

    dim3 grid (gridDimX, gridDimY, gridDimZ);
    dim3 block (blockDimX, blockDimY, blockDimZ);

        
    determinantOnGPU <<<grid, block>>> (d_mat, d_determinants, n_sectors, sector_size);

    CHECK (cudaDeviceSynchronize ());                            // wait for kernel to finish
    CHECK (cudaGetLastError ());                                 // check for kernel errors

    /* copy kernel result back to host side */

    double *h_mat_modified;
    double *determinants;

    //modified_device_sector_data = (unsigned int *) malloc (sector_data_size);
    h_mat_modified = (double *) malloc (mat_area_size);
    determinants = (double *) malloc (number_of_matrix*sizeof(double));
    CHECK (cudaMemcpy (h_mat_modified, d_mat, mat_area_size, cudaMemcpyDeviceToHost));
    CHECK (cudaMemcpy (determinants, d_determinants, number_of_matrix*sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < number_of_matrix; i++) {
        printf("Processing matrix %d \n", i + 1);
        printf("Determinant: %.3e \n\n", determinants[i]);
    }

    /* free device global memory */

    CHECK (cudaFree (d_mat));
    //CHECK (cudaFree (device_sector_number));

    /* reset the device */

    CHECK (cudaDeviceReset ());


    for (int col = 0 ; col < order_matrix * order_matrix ;col++){

    if (col % order_matrix == 0) printf("\n");
    printf("%.3f\t", h_mat_modified[col ]);
    }

    /* free host memory */

    free (h_mat);
    free (h_mat_modified);

    return 0;
}
