#define _POSIX_C_SOURCE 199309L
#include "cvxopt.h"
#include "misc.h"
#include "base.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

extern void print_matrix(matrix *m);

extern int TEST_TIMES;

void test_base_syrk() {
    printf("==== Running test_base_syrk ====\n");

    // A: 3x2, column-major: A = [1 2; 3 4; 5 6]
    double Abuf[] = {
        1.0, 3.0, 5.0,  // col 0
        2.0, 4.0, 6.0   // col 1
    };
    matrix A = {
        .id = DOUBLE,
        .buffer = Abuf,
        .nrows = 3,
        .ncols = 2,
        .mat_type = MAT_DENSE
    };

    // C: 3x3, column-major: C = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    double Cbuf[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    matrix C = {
        .id = DOUBLE,
        .buffer = Cbuf,
        .nrows = 3,
        .ncols = 3,
        .mat_type = MAT_DENSE
    };

    // alpha, beta
    double alpha = 2.0;
    double beta = 1.0;

    base_syrk(&A, &C, 'L', 'N', &alpha, &beta, false);

    clock_t start, end;
    double elapsed;
    start = clock();  
    // base_syrk
    for(int i = 0; i < TEST_TIMES; ++i) {  
        base_syrk(&A, &C, 'L', 'N', &alpha, &beta, false);
    }
    end = clock();
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("⏱️ Elapsed time: %.6f seconds\n", elapsed);

    int n = 3;
    for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
        printf("%8.3f ", Cbuf[i * n + j]);
    printf("\n");
    }
}

void test_base_emul() {
    printf("==== Running test_base_emul ====\n");
    matrix* A = Matrix_New(2, 3, DOUBLE);
    double* Abuf = MAT_BUFD(A);
    Abuf[0] = 1.0; Abuf[1] = 2.0;  // col 0
    Abuf[2] = 3.0; Abuf[3] = 4.0;  // col 1
    Abuf[4] = 5.0; Abuf[5] = 6.0;  // col 2

    // Construct matrix B
    matrix* B = Matrix_New(2, 3, DOUBLE);
    double* Bbuf = MAT_BUFD(B);
    Bbuf[0] = 6.0; Bbuf[1] = 5.0;
    Bbuf[2] = 4.0; Bbuf[3] = 3.0;
    Bbuf[4] = 2.0; Bbuf[5] = 1.0;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrix* C;

    C = (matrix*)base_emul(A, B, 0, 0, DOUBLE, DOUBLE);
    for (size_t i = 0; i < TEST_TIMES; i++)
    {
        // 调用 base_emul
       C = (matrix*)base_emul(A, B, 0, 0, DOUBLE, DOUBLE);
    }

    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Elapsed time: %.6f seconds\n", elapsed);

    // printf("Result of element-wise multiplication (A * B):\n");
    print_matrix(C);

    free(A->buffer); free(A);
    free(B->buffer); free(B);
    free(C->buffer); free(C);
}

void test_base_ediv() {
    printf("==== Running test_base_ediv ====\n");
    matrix* A = Matrix_New(2, 3, DOUBLE);
    double* Abuf = MAT_BUFD(A);
    Abuf[0] = 1.0; Abuf[1] = 2.0;  // col 0
    Abuf[2] = 3.0; Abuf[3] = 4.0;  // col 1
    Abuf[4] = 5.0; Abuf[5] = 6.0;  // col 2

    matrix* B = Matrix_New(2, 3, DOUBLE);
    double* Bbuf = MAT_BUFD(B);
    Bbuf[0] = 6.0; Bbuf[1] = 5.0;
    Bbuf[2] = 4.0; Bbuf[3] = 3.0;
    Bbuf[4] = 2.0; Bbuf[5] = 1.0;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrix* C;

    C = (matrix*)base_ediv(A, B, 0, 0, DOUBLE, DOUBLE);
    for (size_t i = 0; i < TEST_TIMES; i++)
    {
       C = (matrix*)base_ediv(A, B, 0, 0, DOUBLE, DOUBLE);
    }

    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Elapsed time: %.6f seconds\n", elapsed);

    printf("Result of element-wise multiplication (A * B):\n");
    print_matrix(C);

    free(A->buffer); free(A);
    free(B->buffer); free(B);
    free(C->buffer); free(C);
}

void test_base_sqrt() {
    printf("==== Running test_base_sqrt ====\n");
    number n1 = {.i = 16}; // INT
    number n2 = {.d = 25.0}; // DOUBLE
    number n3 = {.z = 3.0 + 4.0*I}; // COMPLEX

    number* r1 = (number*)base_sqrt(&n1, 1, INT);
    number* r2 = (number*)base_sqrt(&n2, 1, DOUBLE);
    number* r3 = (number*)base_sqrt(&n3, 1, COMPLEX);

    // Construct a 2x2 matrix A
    double Abuf[] = {
        1.0, 20.0,  
        4.0, 16.0  
    };

    matrix A = {
        .id = DOUBLE,
        .buffer = Abuf,
        .nrows = 2,
        .ncols = 2,
        .mat_type = MAT_DENSE
    };

    // base_sqrt
    matrix* ret = (matrix*)base_sqrt(&A, 0, DOUBLE);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrix* C;

    for (size_t i = 0; i < TEST_TIMES; i++)
    {
       ret = (matrix*)base_sqrt(&A, 0, DOUBLE);
    }

    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Elapsed time: %.6f seconds\n", elapsed);

    if (!ret) {
        fprintf(stderr, "sqrt failed\n");
        return;
    }

    free(ret->buffer);
    free(ret);
}

void test_base_pow() {
    printf("=== Running test_base_pow ===\n");

    double Abuf[] = {
        1.0, 3.0, 
        2.0, 4.0   
    };
    matrix A = {
        .id = DOUBLE,
        .buffer = Abuf,
        .nrows = 2,
        .ncols = 2,
        .mat_type = MAT_DENSE
    };

    number exponent = {
        .d = -1.0,
    };

    matrix* result = (matrix*) base_pow(&A, &exponent, 0, DOUBLE, DOUBLE);
    if (!result) {
        printf("base_pow returned NULL.\n");
        return;
    }

    printf("Result of A ** 2.0:\n");
    for (int i = 0; i < result->nrows; ++i) {
        for (int j = 0; j < result->ncols; ++j) {
            double val = MAT_BUFD(result)[i + j * result->nrows];  // 列主序
            printf("%8.3f ", val);
        }
        printf("\n");
    }

    Matrix_Free(result);
}


void test_base() {
    printf("==== Running test_base ====\n");

    // test_base_syrk();

    // test_base_emul();

    // test_base_ediv();

    // test_base_sqrt(); 

    // test_base_exp();

    test_base_pow();
}