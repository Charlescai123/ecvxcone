#define _POSIX_C_SOURCE 199309L
#include "cvxopt.h"
#include "misc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void print_matrix_(matrix *m);

extern void lapack_potrf(matrix* A, char uplo, int n, int ldA, int offsetA);
extern void lapack_gesvd(matrix *A, matrix *S, char jobu, char jobvt, matrix *U, matrix *Vt, int m, int n, 
                int ldA, int ldU, int ldVt, int offsetA, int offsetS, int offsetU, int offsetVt);
extern void lapack_ormqr(matrix *A, matrix *tau, matrix *C, char side, char trans, 
                int m, int n, int k, int ldA, int ldC, int offsetA, int offsetC);
extern void lapack_geqrf(matrix *A, matrix *tau, int m, int n, int ldA, int offsetA);
extern void lapack_trtrs(matrix *A, matrix *B, char uplo, char trans, char diag, 
                int n, int nrhs, int ldA, int ldB, int oA, int oB);            
extern void lapack_getrf(matrix* A, matrix* ipiv, int m, int n, int ldA, int offsetA);
extern void lapack_getri(matrix* A, matrix* ipiv, int n, int ldA, int offsetA);

extern matrix fill_random_matrix(int rows, int cols);

extern matrix generate_lower_triangular_matrix(int n);

extern int TEST_TIMES;

void test_lapack_potrf() {
    printf("==== Running test_lapack_potrf ====\n");
    // 初始化对称正定矩阵 A（3x3，列主序）
    double Abuf[] = {
         4.0, 12.0, -16.0,
        12.0, 37.0, -43.0,
       -16.0, -43.0, 98.0
    };

    matrix A = {
        .id = DOUBLE,
        .buffer = Abuf,
        .nrows = 3,
        .ncols = 3,
        .mat_type = MAT_DENSE
    };

    double Bbuf[] = {
         4.0, 12.0, -16.0,
        12.0, 37.0, -43.0,
       -16.0, -43.0, 98.0
    };

    matrix B = {
        .id = DOUBLE,
        .buffer = Bbuf,
        .nrows = 3,
        .ncols = 3,
        .mat_type = MAT_DENSE
    };

    // int n = A.nrows;
    // int lda = A.nrows;
    int info = -1;  // Initialize info to -1 for error checking
    char uplo = 'L';  // Use lower triangular part

    // Call lapack_gesvd：A = U * S * V^T
    
    lapack_potrf(&B, uplo, 3, 3, 0);  // offset=0
    clock_t start, end;
    double elapsed;
    start = clock(); 
    lapack_potrf(&A, uplo, 3, 3, 0);  // offset=0
    end = clock(); 
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("⏱️ Elapsed time: %.6f seconds\n", elapsed);
    
    if (info != 0) {
        printf("dpotrf failed! info = %d\n", info);
        return;
    }

    // Print the Cholesky factor L (A = L * L^T)
    printf("Cholesky factor L (A = L * L^T):\n");
    for (int i = 0; i < A.nrows; ++i) {
        for (int j = 0; j < A.ncols; ++j) {
            double *buf = (double *)A.buffer;
            double val = (i >= j) ? buf[i + j * A.nrows] : 0.0;
            printf("%8.3f ", val);
        }
        printf("\n");
    }
}

void test_lapack_gesvd() {
    printf("==== Running test_lapack_gesvd ====\n");
    double Abuf[] = {
        1.0, 2.0, 3.0,  // col 0
        4.0, 5.0, 6.0   // col 1
    };
    matrix A = {
        .id = DOUBLE,
        .buffer = Abuf,
        .nrows = 3,
        .ncols = 2,
        .mat_type = MAT_DENSE
    };

    // Singular values S (min(m,n) = 2)
    double Sbuf[2] = {0};
    matrix S = {
        .id = DOUBLE,
        .buffer = Sbuf,
        .nrows = 2,
        .ncols = 1,
        .mat_type = MAT_DENSE
    };

    // U: 3x3, Vt: 2x2
    double Ubuf[9] = {0};
    double Vtbuf[4] = {0};
    matrix U = {
        .id = DOUBLE,
        .buffer = Ubuf,
        .nrows = 3,
        .ncols = 3,
        .mat_type = MAT_DENSE
    };
    matrix Vt = {
        .id = DOUBLE,
        .buffer = Vtbuf,
        .nrows = 2,
        .ncols = 2,
        .mat_type = MAT_DENSE
    };

    // Call lapack_gesvd：A = U * S * V^T
    lapack_gesvd(&A, &S, 'A', 'A', &U, &Vt,
                 -1, -1, 0, 0, 0, 0, 0, 0, 0);  // m,n,ldX，offset=0

    struct timespec start, end;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);  

    for(int i = 0; i < TEST_TIMES; ++i) {
        // Call LAPACK SVD：A = U * S * V^T
        lapack_gesvd(&A, &S, 'A', 'A', &U, &Vt,
                     -1, -1, 0, 0, 0, 0, 0, 0, 0);  
    }

    clock_gettime(CLOCK_MONOTONIC, &end);  
    elapsed = (end.tv_sec - start.tv_sec)
            + (end.tv_nsec - start.tv_nsec) / 1e9; 
    printf("⏱️ Elapsed time: %.6f seconds\n", elapsed);

    // Print Singular values S
    printf("Singular values:\n");
    for (int i = 0; i < 2; ++i)
        printf("  %.6f\n", Sbuf[i]);

    // Print left singular vectors U (column vectors)
    printf("\nLeft singular vectors U (3x3):\n");
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            printf("%8.4f ", Ubuf[i + j * 3]);
        printf("\n");
    }

    // Print right singular vectors V^T (2x2)
    printf("\nRight singular vectors V^T (2x2):\n");
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j)
            printf("%8.4f ", Vtbuf[i + j * 2]);
        printf("\n");
    }
}

void test_lapack_geqrf() {
    printf("==== Running test_lapack_geqrf ====\n");
    int m = 3, n = 2, k = 2;

    // ==== Initialize A (3x2), column-major ====
    double Abuf[6] = {
        1.0, 3.0, 5.0,   // col 0
        2.0, 4.0, 6.0    // col 1
    };
    matrix A = {
        .nrows = m,
        .ncols = n,
        .buffer = Abuf,
        .mat_type = MAT_DENSE,
        .id = DOUBLE
    };

    // ==== Initialize tau (2,) ====
    double tau_buf[2] = {0.0, 0.0};
    matrix tau = {
        .nrows = k,
        .ncols = 1,
        .buffer = tau_buf,
        .mat_type = MAT_DENSE,
        .id = DOUBLE
    };

    // ==== Call lapack_geqrf ====
    lapack_geqrf(&A, &tau, -1, -1, 0, 0);

    // === Print A directly ===
    double *a = (double *) A.buffer;
    printf("QR-factorized A (R upper, reflectors lower):\n");
    for (int i = 0; i < A.nrows; ++i) {
        for (int j = 0; j < A.ncols; ++j) {
            printf("%10.6f ", a[i + j * A.nrows]);  // visiting in column-major order
        }
        printf("\n");
    }

    // === Print tau ===
    printf("tau = [ ");
    for (int i = 0; i < 2; ++i)
        printf("%g ", tau_buf[i]);
    printf("]\n");

}

void test_lapack_ormqr() {
    printf("==== Running test_lapack_ormqr ====\n");

    int m = 3, k = 2, n = 2;

    // ==== Initialize matrix A (3x2), column-major ====
    double Abuf[6] = {
        1.0, 3.0, 5.0,   // col 0
        2.0, 4.0, 6.0    // col 1
    };
    matrix A = {
        .nrows = m,
        .ncols = k,
        .buffer = Abuf,
        .mat_type = MAT_DENSE,
        .id = DOUBLE
    };

    // ==== Initialize tau (2,) ====
    double tau_buf[2] = {0.0, 0.0};
    matrix tau = {
        .nrows = 2,
        .ncols = 1,
        .buffer = tau_buf,
        .mat_type = MAT_DENSE,
        .id = DOUBLE
    };

    // ==== Initialize C (3x2), column-major ====
    double Cbuf[6] = {
        7.0, 9.0, 0.0,
        8.0, 10.0, 0.0
    };
    matrix C = {
        .nrows = m,
        .ncols = n,
        .buffer = Cbuf,
        .mat_type = MAT_DENSE,
        .id = DOUBLE
    };

    // ==== QR factorization ====
    lapack_geqrf(&A, &tau, -1, -1, 0, 0);

    // ==== Apply Q to C (left-multiply Q) ====
    lapack_ormqr(&A, &tau, &C, 'L', 'N', -1, -1, -1, 0, 0, 0, 0);

    // ==== Print C's result ====
    double *c = (double *)C.buffer;
    printf("Q @ C =\n");
    for (int i = 0; i < C.nrows; ++i) {
        for (int j = 0; j < C.ncols; ++j) {
            printf("%10.6f ", c[i + j * C.nrows]);  // visiting in column-major order
        }
        printf("\n");
    }
}

void test_lapack_trtrs() {
    printf("==== Running test_lapack_trtrs ====\n");

    // Solve A X = B，where A is 3x3 lower triangular, B is 3x2
    // A = [1 0 0; 2 3 0; 4 5 6]
    // B = [1 2; 3 4; 5 6]
    // double Abuf[9] = {
    //     1.0, 2.0, 4.0,  // col 0
    //     0.0, 3.0, 5.0,  // col 1
    //     0.0, 0.0, 6.0   // col 2
    // };
    double Bbuf[6] = {
        1.0, 8.0, 32.0,  // col 0 (rhs 1)
        2.0, 13.0, 47.0   // col 1 (rhs 2)
    };

    // Construct matrix structures
    // matrix A = {.nrows = 3, .ncols = 3, .buffer = Abuf, .mat_type = MAT_DENSE, .id = DOUBLE};
    // matrix B = {.nrows = 3, .ncols = 2, .buffer = Bbuf, .mat_type = MAT_DENSE, .id = DOUBLE};

    matrix A = generate_lower_triangular_matrix(600);
    matrix B = fill_random_matrix(600, 100);  // Generate random matrix B (600x100)

    // Call LAPACK trtrs to solve AX = B
    lapack_trtrs(&A, &B, 'L', 'N', 'N', 3, 2, 3, 3, 0, 0);

    struct timespec start, end;

    // Get start time
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Run the function you want to measure
    for(int i = 0; i < TEST_TIMES; ++i) {  // Call multiple times to measure time
        // Call LAPACK SVD: A = U * S * V^T
        lapack_trtrs(&A, &B, 'L', 'N', 'N', -1, -1, 0, 0, 0, 0);
    }

    // Get end time
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate elapsed time (in seconds + nanoseconds)
    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Elapsed time: %.6f seconds\n", elapsed);

    // Output the solution X (overwritten in B)
    printf("Solution X (overwritten in B):\n");
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            printf("%8.4f ", Bbuf[i + j * 3]);  // Column-major order
        }
        printf("\n");
    }

}

void test_lapack_inverse() {
    matrix* A = Matrix_New(3, 3, DOUBLE);
    double* A_data = MAT_BUFD(A);
    A_data[0] = 4; A_data[3] = 7; A_data[6] = 2;
    A_data[1] = 3; A_data[4] = 6; A_data[7] = 1;
    A_data[2] = 2; A_data[5] = 5; A_data[8] = 3;
    
    matrix* ipiv = Matrix_New(3, 1, INT);  // pivot indices

    print_matrix_(A);

    // LU factorization
    lapack_getrf(A, ipiv, A->nrows, A->ncols, A->nrows, 0);

    // Inversion
    lapack_getri(A, ipiv, A->nrows, A->nrows, 0);

    print_matrix_(A);

    Matrix_Free(A);
    Matrix_Free(ipiv);
}

void test_lapack() {
    printf("==== Running test_lapack ====\n");
    // test_lapack_potrf();

    // test_lapack_ormqr();

    // test_lapack_gesvd();

    // test_lapack_geqrf();

    // test_lapack_trtrs();

    test_lapack_inverse();
}
