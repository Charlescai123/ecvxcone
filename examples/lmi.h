#ifndef __LMI__
#define __LMI__

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "lapack.h"
#include "cvxopt.h"
#include "cpg_solve.h"
#include "cpg_workspace.h"

extern CPG_Prim_t CPG_Prim;
extern Canon_Params_t Canon_Params_conditioning;

typedef double Matrix3x3[3][3];
typedef double Matrix6x6[6][6];

matrix *P_mat = NULL;  // Global variable for the inverse of Q
matrix *Q_mat = NULL;  // Global variable for Q
matrix *R_mat = NULL;  // Global variable for R
matrix *aB = NULL;  // Global variable for aB
matrix *aF = NULL;  // Global variable for aF

matrix *ipiv = NULL;  // Pivot indices for LU factorization

Matrix6x6 F_kp = {0}; // Feedback gain matrix for kp
Matrix6x6 F_kd = {0}; // Feedback gain matrix for kd

double Ts = 1 / 20.0;  // Sampling period

double tracking_err[10] = {0};      // Tracking error vector (real-time)
double tracking_err_square[10] = {0};       // Square of tracking error vector (real-time)

static Matrix3x3 Rx = {0};
static Matrix3x3 Ry = {0};
static Matrix3x3 Rz = {0};
static Matrix3x3 Rzyx = {0};

static inline void matrix_multiply(Matrix3x3 A, Matrix3x3 B, Matrix3x3 result) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            result[i][j] = 0;
            for (int k = 0; k < 3; ++k)
                result[i][j] += A[i][k] * B[k][j];
        }
}

extern void lmi_init(); // Function to initialize LMI variables

extern ECVXConeWorkspace* ecvxcone_setup(int n_var, int n_ineq, int n_eq, int nnz_G, int nnz_A, DIMs *dims, ECVXConeSettings *settings);
extern ECVXConeWorkspace* ecvxcone_init(matrix *c, spmatrix *G, matrix *h, spmatrix *A, matrix *b, DIMs *dims, ECVXConeSettings *settings);

/********************************  Rotation Matrices  *******************************/
extern void update_Rx(double roll);
extern void update_Ry(double pitch);
extern void update_Rz(double yaw);
extern void update_Rzyx(double roll, double pitch, double yaw);

/*******************************  Update Functions  *******************************/
extern void update_Matrix_A();
extern void update_Matrix_B();
extern void update_TrackingErrorSquare();

extern void update_Matrices();
extern void post_processing();

extern void blas_gemm(matrix *A, matrix *B, matrix *C, char transA, char transB, 
              void* alpha, void* beta, int m, int n, int k, int ldA, int ldB, 
              int ldC, int offsetA, int offsetB, int offsetC);

extern void blas_gemv(matrix *A, matrix *x, matrix *y, char trans, void* alpha, void* beta, 
            int m, int n, int ldA, int incx, int incy, int offsetA, int offsetx, int offsety);

#endif // __LMI__