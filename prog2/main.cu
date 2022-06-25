#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <libgen.h>
#include <unistd.h> 

#include "common.h"
#include "cpu_functions.h"
#include "gpu_functions.cu"

#include <cuda_runtime.h>


/**
 * @brief main function
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main (int argc, char **argv)
{

    /* set up the device */
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK (cudaGetDeviceProperties (&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK (cudaSetDevice (dev));



    /* process command line*/
    char *fName = argv[1];   


    int number_of_matrix, order_matrix;
    double start_GPU_time, end_GPU_time,  start_CPU_time, end_CPU_time;


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
    printf("Number of matrices = %i \n", number_of_matrix);   
    // read orders
    if(fread(&order_matrix, sizeof(int), 1, file) != 1)
    {
        perror("Error reading matrices order!");
        exit(1);
    }
    printf("Matrices order = %i \n\n", order_matrix);
    



    int mat_size = order_matrix * order_matrix * sizeof(double);
    size_t mat_area_size = number_of_matrix * mat_size;
    double * h_mat, * h_determinants;
    double * d_mat, * d_determinants;

    
    h_mat = (double *) malloc (mat_area_size);
    h_determinants = (double *) malloc (number_of_matrix*sizeof(double));
    CHECK (cudaMalloc ((void **) &d_mat, mat_area_size));
    CHECK (cudaMalloc ((void **) &d_determinants, number_of_matrix*sizeof(double)));

    /* initialize the host data */


    if(fread(h_mat, mat_area_size, 1, file) != 1)
    {
        perror("Error reading all matrices!");
        exit(1);
    }

    /* initialize determinants results */
    for (int i = 0; i<number_of_matrix; i++) {
        h_determinants[i] = 1;
    }


    /* copy the host to the device */
    CHECK (cudaMemcpy (d_mat, h_mat, mat_area_size, cudaMemcpyHostToDevice));
    CHECK (cudaMemcpy (d_determinants, h_determinants, number_of_matrix * sizeof(double), cudaMemcpyHostToDevice));

    /* 
        Start GPU process 
    */

    unsigned int gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ;

    blockDimX = order_matrix;
    blockDimY = 1 << 0;                                             // optimize!
    blockDimZ = 1 << 0;                                             // do not change!
    gridDimX = number_of_matrix;
    gridDimY = 1 << 0;                                              // optimize!
    gridDimZ = 1 << 0;                                              // do not change!

    dim3 grid (gridDimX, gridDimY, gridDimZ);
    dim3 block (blockDimX, blockDimY, blockDimZ);
 
 
    start_GPU_time = seconds();
    determinantOnGPU <<<grid, block>>> (d_mat, d_determinants, order_matrix);
    
    CHECK (cudaDeviceSynchronize ());                           
    CHECK (cudaGetLastError ());
 
    end_GPU_time = seconds();

    /* initialize results variables*/
    double *d_mat_aux;
    double *determinants;

    /* allocate memory and copy from device to host*/
    d_mat_aux = (double *) malloc (mat_area_size);
    CHECK (cudaMemcpy (d_mat_aux, d_mat, mat_area_size, cudaMemcpyDeviceToHost));

    determinants = (double *) malloc (number_of_matrix*sizeof(double));
    CHECK (cudaMemcpy (determinants, d_determinants, number_of_matrix*sizeof(double), cudaMemcpyDeviceToHost));

    
    /* free memory to cuda*/
    CHECK (cudaFree (d_mat));
    CHECK (cudaFree (d_determinants));
 
  
    /* reset the device */
    CHECK (cudaDeviceReset ());
    
    /* 
        Start process to calculate determinants using CPU
    */
 
    double *determinants_from_CPU = (double *) malloc (number_of_matrix*sizeof(double));
    
    start_CPU_time = seconds();
    for (int i = 0; i < number_of_matrix; i++) {
        determinants_from_CPU[i] = 1;
        determinantOnCPU (h_mat + (i*order_matrix*order_matrix), &determinants_from_CPU[i], order_matrix);  //calculate each matrix
    }

    end_CPU_time = seconds();
 
    /*  
        Print the results 
    */ 

    for (int i = 0; i < number_of_matrix; i++) {
        printf("Processing matrix %d \n", i + 1);
        printf("GPU Determinant: %.3e \nCPU Determinant: %.3e \n\n", determinants[i], determinants_from_CPU[i]);
    }
    int any_wrong = 0;
    /* Compare results */
    for(int i = 0; i < number_of_matrix; i++) {
        if (!compare_double(determinants[i], determinants_from_CPU[i]))
        { 
            printf ("Different results on matrix %d \n\tGPU calculated %.3e \n\tCPU calculated %.3e\n\n", i+1, determinants[i], determinants_from_CPU[i]);
            any_wrong = 1;
        }
    }

    if (any_wrong == 0)
        printf ("Determinants processed on CPU and GPU has the same results\n\n");

    printf("Time elapsed:\n\tGPU: %f sec\n\tCPU: %f sec\n\n", end_GPU_time - start_GPU_time, end_CPU_time - start_CPU_time);
    /* free memory */
    free (h_mat);
    free (d_mat_aux);
    free(determinants);
    free(determinants_from_CPU);

    return 0;
}







