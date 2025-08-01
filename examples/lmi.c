#include "lmi.h"

extern void cpg_solve();
extern void print_matrix_(matrix *m);

double time_diff(struct timespec start, struct timespec end, int count) {
    double diff_ms = (end.tv_sec - start.tv_sec) * 1e3
                   + (end.tv_nsec - start.tv_nsec) / 1e6;  // ns to ms
    return diff_ms / count;  // Average time per iteration
}

void test_tracking_err() {
    tracking_err[0] = -0.0158;
    tracking_err[1] = -0.0417;
    tracking_err[2] = -0.1517;
    tracking_err[3] = 0.0032;
    tracking_err[4] = 0.2703;
    tracking_err[5] = -0.1057;
    tracking_err[6] = 0.0472;
    tracking_err[7] = 0.3559;
    tracking_err[8] = -0.2925;
    tracking_err[9] = -0.6624;
}

/**
 * Setup the ECVXConeWorkspace structure.
 */
ECVXConeWorkspace* ecvxcone_setup(int n_var, int n_ineq, int n_eq, int nnz_G, int nnz_A, DIMs *dims, ECVXConeSettings *settings)
{
    // c
    if (Canon_Params_conditioning.c == NULL) {
        ecvxcone_data.c = Matrix_New(n_var, 1, DOUBLE);
        if (!ecvxcone_data.c) err_no_memory;
    } else {
        ecvxcone_data.c = malloc(sizeof(matrix));
        ecvxcone_data.c->mat_type = MAT_DENSE;
        ecvxcone_data.c->id = DOUBLE;
        ecvxcone_data.c->nrows = n_var;
        ecvxcone_data.c->ncols = 1;
        ecvxcone_data.c->buffer = Canon_Params_conditioning.c;
    }

    // b
    if (Canon_Params_conditioning.b == NULL) {
        ecvxcone_data.b = Matrix_New(n_eq, 1, DOUBLE);
        if (!ecvxcone_data.b) err_no_memory;
    } else {
        ecvxcone_data.b = malloc(sizeof(matrix));
        ecvxcone_data.b->mat_type = MAT_DENSE;
        ecvxcone_data.b->id = DOUBLE;
        ecvxcone_data.b->nrows = n_eq;
        ecvxcone_data.b->ncols = 1;
        ecvxcone_data.b->buffer = Canon_Params_conditioning.b;
    }

    // h
    if (Canon_Params_conditioning.h == NULL) {
        ecvxcone_data.h = Matrix_New(n_ineq, 1, DOUBLE);
        if (!ecvxcone_data.h) err_no_memory;
    } else {
        ecvxcone_data.h = malloc(sizeof(matrix));
        ecvxcone_data.h->mat_type = MAT_DENSE;
        ecvxcone_data.h->id = DOUBLE;
        ecvxcone_data.h->nrows = n_ineq;
        ecvxcone_data.h->ncols = 1;
        ecvxcone_data.h->buffer = Canon_Params_conditioning.h;
    }

    // G
    if (Canon_Params_conditioning.G == NULL) {
        ecvxcone_data.G = SpMatrix_New(n_ineq, n_var, nnz_G, DOUBLE);
        if (!ecvxcone_data.G) err_no_memory;
    } else{
        ecvxcone_data.G = malloc(sizeof(spmatrix));
        if (!ecvxcone_data.G) err_no_memory;
    
        spmatrix *G_sp = (spmatrix *)ecvxcone_data.G;

        G_sp->obj = malloc(sizeof(ccs));
        if (!G_sp->obj) err_no_memory;

        G_sp->mat_type = MAT_SPARSE;
        G_sp->obj->nrows = n_ineq;
        G_sp->obj->ncols = n_var;
        G_sp->obj->id = DOUBLE;
        G_sp->obj->colptr = (int_t *) Canon_Params_conditioning.G->p;
        G_sp->obj->rowind = (int_t *) Canon_Params_conditioning.G->i;
        G_sp->obj->values = (int_t *) Canon_Params_conditioning.G->x;
    }

    // A
    if (Canon_Params_conditioning.A == NULL) {
        ecvxcone_data.A = SpMatrix_New(n_eq, n_var, nnz_A, DOUBLE);
        if (!ecvxcone_data.A) err_no_memory;
    } else{
        ecvxcone_data.A = malloc(sizeof(spmatrix));
        if (!ecvxcone_data.A) err_no_memory;

        spmatrix *A_sp = (spmatrix *)ecvxcone_data.A;

        A_sp->obj = malloc(sizeof(ccs));
        if (!A_sp->obj) err_no_memory;

        A_sp->mat_type = MAT_SPARSE;
        A_sp->obj->nrows = n_eq;
        A_sp->obj->ncols = n_var;
        A_sp->obj->id = DOUBLE;
        A_sp->obj->colptr = Canon_Params_conditioning.A->p;
        A_sp->obj->rowind = Canon_Params_conditioning.A->i;
        A_sp->obj->values = Canon_Params_conditioning.A->x;
    }

    if (!ecvxcone_data.c || !ecvxcone_data.h || !ecvxcone_data.b || !ecvxcone_data.G || !ecvxcone_data.A) {
        err_no_memory;
    }
    ECVXConeWorkspace *ecvxcone_ws = ecvxcone_init(ecvxcone_data.c, ecvxcone_data.G, ecvxcone_data.h, ecvxcone_data.A, ecvxcone_data.b, dims, settings);
    return ecvxcone_ws;
}

/********************************  LMI Initialization  *******************************/
void lmi_init() {
    // Initialize Q
    Q_mat = malloc(sizeof(matrix)); if (!Q_mat) err_no_memory;
    Q_mat->mat_type = MAT_DENSE;
    Q_mat->id = DOUBLE;
    Q_mat->nrows = 10;
    Q_mat->ncols = 10;
    Q_mat->buffer = CPG_Prim.Q;

    // Initialize R
    R_mat = malloc(sizeof(matrix)); if (!R_mat) err_no_memory;
    R_mat->mat_type = MAT_DENSE;
    R_mat->id = DOUBLE;
    R_mat->nrows = 6;
    R_mat->ncols = 10;
    R_mat->buffer = CPG_Prim.R;

    // Initialize P
    // P = Matrix_New(10, 10, DOUBLE);

    // Initialize aF
    aF = Matrix_New(10, 10, DOUBLE);

    // Initialize aB
    aB = Matrix_New(10, 6, DOUBLE);
    for (int i = 0; i < 6; ++i) {
        int idx = 4 + i * 11;
        MAT_BUFD(aB)[idx] = 1.0;  // Initialize to identity-like structure
    }

    // Initialize ipiv for LU factorization
    ipiv = Matrix_New(10, 1, INT);
}

/********************************  Rotation Matrices  *******************************/
void update_Rx(double roll) {
    Rx[0][0] = 1; Rx[0][1] = 0;         Rx[0][2] = 0;
    Rx[1][0] = 0; Rx[1][1] = cos(roll); Rx[1][2] = -sin(roll);
    Rx[2][0] = 0; Rx[2][1] = sin(roll); Rx[2][2] = cos(roll);
}

void update_Ry(double pitch) {
    Ry[0][0] = cos(pitch);  Ry[0][1] = 0; Ry[0][2] = sin(pitch);
    Ry[1][0] = 0;           Ry[1][1] = 1; Ry[1][2] = 0;
    Ry[2][0] = -sin(pitch); Ry[2][1] = 0; Ry[2][2] = cos(pitch);
}

void update_Rz(double yaw) {
    Rz[0][0] = cos(yaw);  Rz[0][1] = -sin(yaw); Rz[0][2] = 0;
    Rz[1][0] = sin(yaw);  Rz[1][1] = cos(yaw);  Rz[1][2] = 0;
    Rz[2][0] = 0;         Rz[2][1] = 0;         Rz[2][2] = 1;
}

void update_Rzyx(double roll, double pitch, double yaw) {
    update_Rx(roll);
    update_Ry(pitch);
    update_Rz(yaw);
    
    Matrix3x3 temp;
    matrix_multiply(Rz, Ry, temp);
    matrix_multiply(temp, Rx, Rzyx);
}

/*******************************  Update Functions  *******************************/

// Function to update the rotation matrices based on tracking error
void update_Matrix_A(){
    cpg_update_A(71, Rzyx[0][0] * Ts);  cpg_update_A(81, Rzyx[0][1] * Ts);  cpg_update_A(91, Rzyx[0][2] * Ts);
    cpg_update_A(72, Rzyx[1][0] * Ts);  cpg_update_A(82, Rzyx[1][1] * Ts);  cpg_update_A(92, Rzyx[1][2] * Ts);
    cpg_update_A(73, Rzyx[2][0] * Ts);  cpg_update_A(83, Rzyx[2][1] * Ts);  cpg_update_A(93, Rzyx[2][2] * Ts);
}

void update_Matrix_B(){

}

void update_TrackingErrorSquare() {
    for (int i = 0; i < 10; ++i) {
        tracking_err_square[i] = tracking_err[i] * tracking_err[i];
    }
    cpg_update_tracking_err_square(0, tracking_err_square[0]);
    cpg_update_tracking_err_square(1, tracking_err_square[1]);
    cpg_update_tracking_err_square(2, tracking_err_square[2]);
    cpg_update_tracking_err_square(3, tracking_err_square[3]);
    cpg_update_tracking_err_square(4, tracking_err_square[4]);   
    cpg_update_tracking_err_square(5, tracking_err_square[5]);
    cpg_update_tracking_err_square(6, tracking_err_square[6]);
    cpg_update_tracking_err_square(7, tracking_err_square[7]);
    cpg_update_tracking_err_square(8, tracking_err_square[8]);
    cpg_update_tracking_err_square(9, tracking_err_square[9]);
}

// Post-processing function to finalize the feedback gain matrices
void post_processing() {
    if (!P_mat) Matrix_Free(P_mat);  // Free P if it was previously allocated
    P_mat = Matrix_NewFromMatrix(Q_mat, Q_mat->id);  // Ensure P is initialized

    // P = inverse(Q)
    lapack_getrf(P_mat, ipiv, P_mat->nrows, P_mat->ncols, P_mat->nrows, 0);   // LU factorization
    lapack_getri(P_mat, ipiv, P_mat->nrows, P_mat->nrows, 0);   // Inversion

    // aF = round(aB * R * P)
    matrix *temp = Matrix_NewFromMatrix(aF, aF->id);  // Temporary matrix for multiplication
    blas_gemm(aB, R_mat, temp, 'N', 'N', NULL, NULL, -1, -1, -1, 0, 0, 0, 0, 0, 0);
    blas_gemm(temp, P_mat, aF, 'N', 'N', NULL, NULL, -1, -1, -1, 0, 0, 0, 0, 0, 0);
    Matrix_Free(temp);  // Free the temporary matrix

    // Fb2 = aF[6:10, 0:4]
    // F_kp = -np.block([
    //     [np.zeros((2, 6))], 
    //     [np.zeros((4, 2)), Fb2]])
    for (int i = 6; i < 10; ++i) {
        for (int j = 0; j < 4; ++j) {
            F_kp[i-4][j+2] = -MAT_ELEMD(aF, i, j);
        }
    }

    // F_kd = -aF[4:10, 4:10]
    for (int i = 4; i < 10; ++i) {
        for (int j = 4; j < 10; ++j) {
            F_kd[i-4][j-4] = -MAT_ELEMD(aF, i, j);
        }
    }
}

// Function to update the rotation matrices based on tracking error
void update_Matrices() {
    
    double roll = tracking_err[1];
    double pitch = tracking_err[2];
    double yaw = tracking_err[3];

    update_Rzyx(roll, pitch, yaw);

    update_Matrix_A();
    update_Matrix_B();
    update_TrackingErrorSquare();

    cpg_solve();

    post_processing();
}

void print_matrix6x6(Matrix6x6 mat, const char *name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < 6; ++i) {
        printf("  [");
        for (int j = 0; j < 6; ++j) {
            printf("%8.4f", mat[i][j]);
            if (j < 5) printf(", ");
        }
        printf("]\n");
    }
    printf("\n");
}

void benchmark(int iterations, bool verbose, double threshold) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 10; j++) {
            tracking_err[j] = ((double) rand() / RAND_MAX) * threshold;  // Random tracking error
        }
        // test_tracking_err();
        update_Matrices();  // Update matrices based on tracking error
        
        // Print the resulting rotation matrix
        if (verbose) {
            printf("Iteration %d:\n", i + 1);
            print_matrix6x6(F_kp, "F_kp");
            print_matrix6x6(F_kd, "F_kd");
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Average solving time per iteration: %.3f ms\n", time_diff(start, end, iterations));
}

int main() {
    int ITER = 1000;
    srand(time(NULL));  // Initialize random seed

    lmi_init();  // Initialize LMI structures

    // test_tracking_err();  // Test tracking error
    update_Matrices();
    print_matrix6x6(F_kp, "F_kp");
    print_matrix6x6(F_kd, "F_kd");

    tracking_err[0] = -0.123;  // Test tracking error
    update_Matrices();
    print_matrix6x6(F_kp, "F_kp");
    print_matrix6x6(F_kd, "F_kd");

    // Benchmark full update
    benchmark(ITER, true, 0.1);

    return 0;
}