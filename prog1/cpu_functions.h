#include <math.h>

/**
 * @brief Compare 2 doubles with 0.0001 of precision. If is equal return 1
 * 
 * @param f1 
 * @param f2 
 * @return int 
 */
int compare_double(double f1, double f2)
{
    //double precision = 0.01;
    float res = f1 / f2;
    if (res <= 1.001 && res >= 0.999 ) 
        return 1;
    return 0;
}


/**
 * @brief Swaps the values of columns X and Y
 * 
 * @param matrix 
 * @param x 
 * @param y 
 * @param size 
 */
void swap_columns(double *matrix, int *x, int *y, int size)
{
    for (int i = 0; i < size; i++)
    {
        double tmp = matrix[size * i + (*x)];
        matrix[size * i + (*x)] = matrix[size * i + (*y)];
        matrix[size * i + (*y)] = tmp;
    }
}

/**
 * @brief Applies Gaussian Elimination
 * 
 * @param kj 
 * @param ki 
 * @param ii 
 * @param ij 
 */
void formulaCPU(double *kj, double ki, double ii, double ij)
{
    *kj = *kj - ((ki / ii) * ij);
}

/**
 * @brief Calculate the determinant of matrix
 * 
 * @param matrix 
 * @param determinant 
 * @param order 
 */
static void determinantOnCPU(double *matrix, double *determinant,  int order){
    int signal_reversion;
    for (int i = 0; i < order; i++)
    {
        int index = order * i + i;


        if (matrix[index] == 0)
        {
            for (int j = i + 1; j < order; j++)
            {
                if (matrix[order * i + j] != 0)
                {
                    swap_columns(matrix, &i, &j, order);
                    signal_reversion = !signal_reversion;
                    break;
                }
            }  
        }
        
        for (int j = order - 1; j > i - 1; j--)
        {
            for (int k = i + 1; k < order; k++)
            {
                formulaCPU(&matrix[ order * k + j], matrix[ order * k + i], matrix[ order * i + i], matrix[ order * i + j]);
            }
        }

        if (matrix[ order * i + i] == 0)
            determinant = 0;

        *determinant *= matrix[ order * i + i];
    }
}

