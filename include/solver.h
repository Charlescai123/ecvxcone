#include "cvxopt.h"
#include "misc.h"

#ifndef __SOLVER__
#define __SOLVER__

#define OPTIMAL 0
#define PRIMAL_INFEASIBLE 1
#define DUAL_INFEASIBLE 2
#define UNKNOWN 3

extern const char* defaultsolvers[];
extern char msg[256];

// Static variables for ECVXCONE
// static matrix *c;
// static spmatrix *G;
// static matrix *h;
// static spmatrix *A;
// static matrix *b;

typedef struct {
    bool debug;
    double kktreg;
    int maxiters;
    double abstol;
    double reltol;
    double feastol;
    bool show_progress;
    int refinement;
    bool use_correction;
    int EXPON;
    double STEP;
    char kktsolver[10]; // KKT solver name
} ECVXConeSettings;


static ECVXConeSettings ecvxcone_settings = {
    .debug = false,
    .kktreg = -1.0, // -1 indicates None/unset
    .maxiters = 100,
    // .abstol = 1e-7,  // Original default values
    // .reltol = 1e-6,  // Original default values
    // .feastol = 1e-7, // Original default values
    .abstol = 1e-1, // Adjusted for better solving time
    .reltol = 1e-1, // Adjusted for better solving time
    .feastol = 1e-1, // Adjusted for better solving time
    .show_progress = false,
    .refinement = -1, // -1 indicates None/unset
    .use_correction = true,
    .EXPON = 3,
    .STEP = 0.99,
    .kktsolver = "chol" // Default KKT solver
};

typedef struct {
    matrix *c; // Objective function coefficients
    matrix *b; // Right-hand side vector for the constraints
    matrix *h; // Right-hand side vector
    void *G; // Coefficient matrix for the linear term
    void *A; // Coefficient matrix for the constraints
} ECVXConeData;

static ECVXConeData ecvxcone_data = {
    .c = NULL,
    .b = NULL,
    .h = NULL,
    .G = NULL,
    .A = NULL
};

// Primal start structure
typedef struct {
    matrix* x;
    matrix* s;
} PrimalStart;

// Dual start structure  
typedef struct {
    matrix* y;
    matrix* z;
} DualStart;

// Result structure
typedef struct {
    char* status;
    matrix* x;
    matrix* s;
    matrix* y;
    matrix* z;
    double primal_objective;
    double dual_objective;
    double gap;
    double relative_gap;
    double primal_infeasibility;
    double dual_infeasibility;
    double primal_slack;
    double dual_slack;
    double residual_as_primal_infeasibility_certificate;
    double residual_as_dual_infeasibility_certificate;
    int iterations;
} ECVXConeResult;

typedef struct {
    DIMs *dims; // Dimensions of the problem

    PrimalStart *primalstart; // Primal start values
    DualStart *dualstart; // Dual start values
    
    int sum_dims_q;     // Sum of q dimensions
    int sum_dims_s;     // Sum of s dimensions

    int cdim;   // Cone dimension
    int cdim_pckd; // Packed cone dimension
    int cdim_diag; // Diagonal cone dimension

    int *indq;     // Index array for 'q' constraints
    int *inds;     // Index array for 's' constraints

    scaling *W_init; // Scaling structure (identity) for initialization
    scaling *W_nt; // Scaling structure for iterations

    ECVXConeData *data; // Data structure
    ECVXConeResult *result; // Result structure
} ECVXConeWorkspace;


// Function declarations
void Gf_gemv(matrix *x, matrix *y, void *G, DIMs *dims, char trans, void* alpha, void* beta);
void Af_gemv(matrix *x, matrix *y, void *A, char trans, void* alpha, void* beta);
void xy_copy(matrix *x, matrix *y);

/************************* Declarations *************************/
extern KKTCholContext* kkt_chol(void *G, DIMs *dims, void *A, int mnl);
extern void factor_function(scaling *W, matrix *H, matrix *Df, KKTCholContext *ctx, DIMs *dims);
extern void solve_function(matrix *x, matrix *y, matrix *z, KKTCholContext *ctx, DIMs *dims);

/* misc library */
extern void misc_sgemv(void *A, matrix *x, matrix *y, DIMs *dims, char trans, double alpha, 
                        double beta, int n, int offsetA, int offsetx, int offsety);
extern void misc_scale(matrix *x, scaling *W, char trans, char inverse);
extern void misc_scale2(matrix *lmbda, matrix *x, DIMs *dims, int mnl, char inverse);
extern void misc_sprod(matrix *x, matrix *y, DIMs *dims, int mnl, char diag);
extern double misc_snrm2(matrix *x, DIMs *dims, int mnl);
extern double misc_max_step(matrix* x, DIMs* dims, int mnl, matrix* sigma);
extern double misc_sdot(matrix *x, matrix *y, DIMs *dims, int mnl);
extern void misc_symm(matrix *x, int n, int offset);
extern scaling* misc_compute_scaling(matrix *s, matrix *z, matrix *lmbda, DIMs *dims, int mnl);
extern void misc_compute_scaling2(scaling *W, matrix *s, matrix *z, matrix *lmbda, DIMs *dims, int mnl);
extern void misc_update_scaling(scaling *W, matrix *lmbda, matrix *s, matrix *z);
extern void misc_ssqr(matrix *x, matrix *y, DIMs *dims, int mnl);
extern void misc_sinv(matrix *x, matrix *y, DIMs *dims, int mnl);

/* base library */
extern void base_gemv(void *A, matrix *x, matrix *y, char trans, void *alpha, void *beta, 
              int m, int n, int incx, int incy, int offsetA, int offsetx, int offsety);

/* blas library */
extern void blas_copy(matrix *x, matrix *y, int n, int ix, int iy, int ox, int oy);
extern void blas_axpy(matrix *x, matrix *y, number *alpha, int n, int incx, int incy, int offsetx, int offsety);
extern void blas_scal(void* alpha, matrix* x, int n, int inc, int offset);
extern number blas_dot(matrix *x, matrix *y, int n, int incx, int incy, int offsetx, int offsety);
extern double blas_nrm2(matrix *x, int n, int inc, int offset);
extern void blas_tbsv(matrix *A, matrix *x, char uplo, char trans, char diag, 
          int n, int k, int ldA, int incx, int offsetA, int offsetx);

/* dense library */
extern matrix * dense(spmatrix *sp_mat);

/* sparse library */
extern ccs * transpose(ccs *A, int conjugate);

extern void debug_matrix_by_project2vector(matrix *m);

// KKT solver function pointer
typedef void* (*KKTSolverFunc)(void* W);

extern void KKTCholContext_Free(KKTCholContext *ctx);

static inline ECVXConeResult *ECVXConeResult_Init() {
    ECVXConeResult *result = (ECVXConeResult *)malloc(sizeof(ECVXConeResult));
    if (!result) err_no_memory;

    result->status = NULL;
    result->x = NULL;
    result->s = NULL;
    result->y = NULL;
    result->z = NULL;
    result->primal_objective = 0.0;
    result->dual_objective = 0.0;
    result->gap = 0.0;
    result->relative_gap = 0.0;
    result->primal_infeasibility = 0.0;
    result->dual_infeasibility = 0.0;
    result->primal_slack = 0.0;
    result->dual_slack = 0.0;
    result->residual_as_primal_infeasibility_certificate = 0.0;
    result->residual_as_dual_infeasibility_certificate = 0.0;
    result->iterations = 0;
    return result;
}

static inline void ECVXConeResult_Free(ECVXConeResult *result) {
    if (result) {
        // if (result->status) free(result->status);
        if (result->x) Matrix_Free(result->x);
        if (result->s) Matrix_Free(result->s);
        if (result->y) Matrix_Free(result->y);
        if (result->z) Matrix_Free(result->z);
        free(result);
    }
}

static inline void ECVXConeData_Free(ECVXConeData *data) {
    if (data) {
        if (data->c) Matrix_Free(data->c);
        if (data->b) Matrix_Free(data->b);
        if (data->h) Matrix_Free(data->h);

        if (Matrix_Check(data->G) && !SpMatrix_Check(data->G)) {
            // G is a dense matrix
            Matrix_Free(data->G);
        } else if (SpMatrix_Check(data->G) && !Matrix_Check(data->G)) {
            // G is a sparse matrix
            SpMatrix_Free(data->G);
        }

        if (Matrix_Check(data->A) && !SpMatrix_Check(data->A)) {
            // A is a dense matrix
            Matrix_Free(data->A);
        } else if (SpMatrix_Check(data->A) && !Matrix_Check(data->A)) {
            // A is a sparse matrix
            SpMatrix_Free(data->A);
        }

        // Free the data structure itself
        free(data);
    }
}

static inline void ECVXConeWorkspace_Free(ECVXConeWorkspace *ws) 
{
    if (ws) {
        // if (ws->dims) {
        //     if (ws->dims->q) free(ws->dims->q);
        //     if (ws->dims->s) free(ws->dims->s);
        //     free(ws->dims);
        // }

        if (ws->primalstart) {
            if (ws->primalstart->x) Matrix_Free(ws->primalstart->x);
            if (ws->primalstart->s) Matrix_Free(ws->primalstart->s);
            free(ws->primalstart);
        }
        if (ws->dualstart) {
            if (ws->dualstart->y) Matrix_Free(ws->dualstart->y);
            if (ws->dualstart->z) Matrix_Free(ws->dualstart->z);
            free(ws->dualstart);
        }

        if(ws->W_init) {
            Scaling_Free(ws->W_init);
        }
        if(ws->W_nt) {
            Scaling_Free(ws->W_nt);
        }

        if (ws->data) {
            ECVXConeData_Free(ws->data);
        }

        if(ws->result) {
            ECVXConeResult_Free(ws->result);
        }

        // Free any allocated resources within the workspace
        free(ws);
    }
    return NULL;
}

#endif