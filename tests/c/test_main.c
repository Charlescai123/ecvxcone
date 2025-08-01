#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cvxopt.h" 
#include "misc.h"
#include "solver.h"
#include <sys/time.h>
#include <unistd.h> 
#include <limits.h>

int TEST_TIMES = 1000;

extern matrix* load_txt_as_matrix(const char* filename);
extern spmatrix* load_spmatrix_from_triplet_file(const char *filename, int nrows, int ncols);
extern ECVXConeWorkspace* ecvxcone_init(matrix *c, void *G, matrix *h, void *A, matrix *b, DIMs *dims, 
                                ECVXConeSettings* settings);
extern void ecvxcone_free(ECVXConeWorkspace *ecvxcone_ws);
extern int conelp(ECVXConeWorkspace* ecvxcone_ws, ECVXConeSettings* settings);

extern void print_matrix_(matrix *m);
void test_conelp();

const char mat_c_path[] = "../tests/c/dummy_input/c.txt";        // For running
const char mat_h_path[] = "../tests/c/dummy_input/h.txt";
const char spmat_G_path[] = "../tests/c/dummy_input/G_sp.txt";

void test_all() {
    // test_base();
    // test_blas();
    // test_lapack();
    // test_sparse();
    // test_misc_solvers();
    test_conelp(); 
}

void test_conelp() {
    printf("==== Running test_conelp ====\n");

    matrix *c = load_txt_as_matrix(mat_c_path);
    matrix *h = load_txt_as_matrix(mat_h_path);
    spmatrix *G_sp = load_spmatrix_from_triplet_file(spmat_G_path, 973, 136);

    if (!c || !h || !G_sp) {
        fprintf(stderr, "Failed to load matrices\n");
        return;
    }

    matrix *b = Matrix_New(0, 1, DOUBLE);
    void *A = SpMatrix_New(0, 136, 0, DOUBLE);
    // matrix *A = SpMatrix_New(0, 21, 0, DOUBLE);

    int sbuf[] = { 10, 6, 20, 16, 10, 3, 6, 6 };
    // int sbuf[] = { 4, 2, 8, 1, 2};
    // DIMs dims = {.l = 0, .q_size = 0, .s_size = 5, .q = NULL, .s = sbuf};     
    DIMs dims = {.l = 0, .q_size = 0, .s_size = 8, .q = NULL, .s = sbuf};     
    // PrimalStart *primalstart = NULL;
    // DualStart *dualstart = NULL;    

    ECVXConeWorkspace *ecvxcone_ws = ecvxcone_init(c, G_sp, h, A, b, &dims, &ecvxcone_settings);

    conelp(ecvxcone_ws, &ecvxcone_settings);
    
    printf("Result x:\n");
    print_matrix_(ecvxcone_ws->result->x);
    // print_matrix_(ecvxcone_ws->result->s, "Result s");
    // print_matrix_(ecvxcone_ws->result->y, "Result y");
    // print_matrix_(ecvxcone_ws->result->z, "Result z");
    printf("ConeLPResult status: %d\n", ecvxcone_ws->result->status);

    // int test_time = 1;
    int test_time = TEST_TIMES;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < test_time; ++i) {
        conelp(ecvxcone_ws, &ecvxcone_settings);
    }
    gettimeofday(&end, NULL);

    double elapsed = (end.tv_sec - start.tv_sec)
                   + (end.tv_usec - start.tv_usec) / 1e6;
    double avg_time = elapsed / test_time * 1e3;
    printf("Average time per iteration (Running %d times): %.6f milliseconds\n", test_time, avg_time);
    printf("ConeLPResult status: %d\n", ecvxcone_ws->result->status);

    // Matrix_Free(c);
    // Matrix_Free(h);
    // Matrix_Free(b);
    // SpMatrix_Free(A);         
    // SpMatrix_Free(G_sp);  

    // ECVXConeWorkspace_Free(ecvxcone_ws);
}


void test_matrix_assign() {
    printf("==== Running test_matrix_assign ====\n");
    matrix *src = Matrix_New(4, 4, DOUBLE);
    matrix *dst = Matrix_New(4, 4, DOUBLE);

    // [ 1  5  9 13
    //   2  6 10 14
    //   3  7 11 15
    //   4  8 12 16 ]
    double *data = (double*)src->buffer;
    for (int i = 0; i < 16; ++i)
        data[i] = i + 1;

    printf("Source matrix:\n");
    print_matrix_(src);

    // Copy: src to dst (0:4, 0:4)
    matrix_slice_assign(dst, src, 0, 4, 0, 4);

    printf("\nDestination matrix after assignment:\n");
    print_matrix_(dst);

    Matrix_Free(src);
    Matrix_Free(dst);
    return;
}

int main() {

    char cwd[PATH_MAX];
    
    if (getcwd(cwd, sizeof(cwd))) {
        printf("Current working directory: %s\n", cwd);
    } else {
        perror("getcwd() error");
        return 1;
    }

    // Start timing
    // clock_t start, end;
    // double elapsed;  start = clock();  // Start timing

    struct timeval start, end;
    gettimeofday(&start, NULL);

    test_all();  // Run all tests
    // test_matrix_assign();  // Test matrix_assign
    // test_number_sqrt();  // Test number_sqrt
    // test_base_emul();  // Test base_emul
    // test_base_ediv();  // Test base_ediv

    // End timing
    // end = clock();  // End timing

    // elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    // printf("⏱️ Elapsed time: %.6f seconds\n", elapsed);

    gettimeofday(&end, NULL);

    // double elapsed = (end.tv_sec - start.tv_sec)
    //                + (end.tv_usec - start.tv_usec) / 1e6;
    // printf("Elapsed time: %.6f seconds\n", elapsed);
    
    return 0;
}


