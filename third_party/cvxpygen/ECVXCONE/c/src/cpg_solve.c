
/*
Auto-generated by CVXPYgen on August 01, 2025 at 01:40:53.
Content: Function definitions.
*/

#include "cpg_solve.h"
#include "cpg_workspace.h"

static cpg_int i;
static cpg_int j;

// Update user-defined parameters
void cpg_update_A(cpg_int idx, cpg_float val){
  cpg_params_vec[idx+0] = val;
  Canon_Outdated.G = 1;
}

void cpg_update_B(cpg_int idx, cpg_float val){
  cpg_params_vec[idx+100] = val;
  Canon_Outdated.G = 1;
}

void cpg_update_tracking_err_square(cpg_int idx, cpg_float val){
  cpg_params_vec[idx+160] = val;
  Canon_Outdated.h = 1;
}

// Map user-defined to canonical parameters
void cpg_canonicalize_G(){
  for(i=0; i<3973; i++){
    Canon_Params.G->x[i] = 0;
    for(j=canon_G_map.p[i]; j<canon_G_map.p[i+1]; j++){
      Canon_Params.G->x[i] += canon_G_map.x[j]*cpg_params_vec[canon_G_map.i[j]];
    }
  }
}

void cpg_canonicalize_h(){
  for(i=0; i<973; i++){
    Canon_Params.h[i] = 0;
    for(j=canon_h_map.p[i]; j<canon_h_map.p[i+1]; j++){
      Canon_Params.h[i] += canon_h_map.x[j]*cpg_params_vec[canon_h_map.i[j]];
    }
  }
}

// Retrieve primal solution in terms of user-defined variables
void cpg_retrieve_prim(){
  CPG_Prim.Q[0] = MAT_BUFD(ecvxcone_ws->result->x)[0];
  CPG_Prim.Q[1] = MAT_BUFD(ecvxcone_ws->result->x)[1];
  CPG_Prim.Q[2] = MAT_BUFD(ecvxcone_ws->result->x)[2];
  CPG_Prim.Q[3] = MAT_BUFD(ecvxcone_ws->result->x)[3];
  CPG_Prim.Q[4] = MAT_BUFD(ecvxcone_ws->result->x)[4];
  CPG_Prim.Q[5] = MAT_BUFD(ecvxcone_ws->result->x)[5];
  CPG_Prim.Q[6] = MAT_BUFD(ecvxcone_ws->result->x)[6];
  CPG_Prim.Q[7] = MAT_BUFD(ecvxcone_ws->result->x)[7];
  CPG_Prim.Q[8] = MAT_BUFD(ecvxcone_ws->result->x)[8];
  CPG_Prim.Q[9] = MAT_BUFD(ecvxcone_ws->result->x)[9];
  CPG_Prim.Q[10] = MAT_BUFD(ecvxcone_ws->result->x)[1];
  CPG_Prim.Q[11] = MAT_BUFD(ecvxcone_ws->result->x)[10];
  CPG_Prim.Q[12] = MAT_BUFD(ecvxcone_ws->result->x)[11];
  CPG_Prim.Q[13] = MAT_BUFD(ecvxcone_ws->result->x)[12];
  CPG_Prim.Q[14] = MAT_BUFD(ecvxcone_ws->result->x)[13];
  CPG_Prim.Q[15] = MAT_BUFD(ecvxcone_ws->result->x)[14];
  CPG_Prim.Q[16] = MAT_BUFD(ecvxcone_ws->result->x)[15];
  CPG_Prim.Q[17] = MAT_BUFD(ecvxcone_ws->result->x)[16];
  CPG_Prim.Q[18] = MAT_BUFD(ecvxcone_ws->result->x)[17];
  CPG_Prim.Q[19] = MAT_BUFD(ecvxcone_ws->result->x)[18];
  CPG_Prim.Q[20] = MAT_BUFD(ecvxcone_ws->result->x)[2];
  CPG_Prim.Q[21] = MAT_BUFD(ecvxcone_ws->result->x)[11];
  CPG_Prim.Q[22] = MAT_BUFD(ecvxcone_ws->result->x)[19];
  CPG_Prim.Q[23] = MAT_BUFD(ecvxcone_ws->result->x)[20];
  CPG_Prim.Q[24] = MAT_BUFD(ecvxcone_ws->result->x)[21];
  CPG_Prim.Q[25] = MAT_BUFD(ecvxcone_ws->result->x)[22];
  CPG_Prim.Q[26] = MAT_BUFD(ecvxcone_ws->result->x)[23];
  CPG_Prim.Q[27] = MAT_BUFD(ecvxcone_ws->result->x)[24];
  CPG_Prim.Q[28] = MAT_BUFD(ecvxcone_ws->result->x)[25];
  CPG_Prim.Q[29] = MAT_BUFD(ecvxcone_ws->result->x)[26];
  CPG_Prim.Q[30] = MAT_BUFD(ecvxcone_ws->result->x)[3];
  CPG_Prim.Q[31] = MAT_BUFD(ecvxcone_ws->result->x)[12];
  CPG_Prim.Q[32] = MAT_BUFD(ecvxcone_ws->result->x)[20];
  CPG_Prim.Q[33] = MAT_BUFD(ecvxcone_ws->result->x)[27];
  CPG_Prim.Q[34] = MAT_BUFD(ecvxcone_ws->result->x)[28];
  CPG_Prim.Q[35] = MAT_BUFD(ecvxcone_ws->result->x)[29];
  CPG_Prim.Q[36] = MAT_BUFD(ecvxcone_ws->result->x)[30];
  CPG_Prim.Q[37] = MAT_BUFD(ecvxcone_ws->result->x)[31];
  CPG_Prim.Q[38] = MAT_BUFD(ecvxcone_ws->result->x)[32];
  CPG_Prim.Q[39] = MAT_BUFD(ecvxcone_ws->result->x)[33];
  CPG_Prim.Q[40] = MAT_BUFD(ecvxcone_ws->result->x)[4];
  CPG_Prim.Q[41] = MAT_BUFD(ecvxcone_ws->result->x)[13];
  CPG_Prim.Q[42] = MAT_BUFD(ecvxcone_ws->result->x)[21];
  CPG_Prim.Q[43] = MAT_BUFD(ecvxcone_ws->result->x)[28];
  CPG_Prim.Q[44] = MAT_BUFD(ecvxcone_ws->result->x)[34];
  CPG_Prim.Q[45] = MAT_BUFD(ecvxcone_ws->result->x)[35];
  CPG_Prim.Q[46] = MAT_BUFD(ecvxcone_ws->result->x)[36];
  CPG_Prim.Q[47] = MAT_BUFD(ecvxcone_ws->result->x)[37];
  CPG_Prim.Q[48] = MAT_BUFD(ecvxcone_ws->result->x)[38];
  CPG_Prim.Q[49] = MAT_BUFD(ecvxcone_ws->result->x)[39];
  CPG_Prim.Q[50] = MAT_BUFD(ecvxcone_ws->result->x)[5];
  CPG_Prim.Q[51] = MAT_BUFD(ecvxcone_ws->result->x)[14];
  CPG_Prim.Q[52] = MAT_BUFD(ecvxcone_ws->result->x)[22];
  CPG_Prim.Q[53] = MAT_BUFD(ecvxcone_ws->result->x)[29];
  CPG_Prim.Q[54] = MAT_BUFD(ecvxcone_ws->result->x)[35];
  CPG_Prim.Q[55] = MAT_BUFD(ecvxcone_ws->result->x)[40];
  CPG_Prim.Q[56] = MAT_BUFD(ecvxcone_ws->result->x)[41];
  CPG_Prim.Q[57] = MAT_BUFD(ecvxcone_ws->result->x)[42];
  CPG_Prim.Q[58] = MAT_BUFD(ecvxcone_ws->result->x)[43];
  CPG_Prim.Q[59] = MAT_BUFD(ecvxcone_ws->result->x)[44];
  CPG_Prim.Q[60] = MAT_BUFD(ecvxcone_ws->result->x)[6];
  CPG_Prim.Q[61] = MAT_BUFD(ecvxcone_ws->result->x)[15];
  CPG_Prim.Q[62] = MAT_BUFD(ecvxcone_ws->result->x)[23];
  CPG_Prim.Q[63] = MAT_BUFD(ecvxcone_ws->result->x)[30];
  CPG_Prim.Q[64] = MAT_BUFD(ecvxcone_ws->result->x)[36];
  CPG_Prim.Q[65] = MAT_BUFD(ecvxcone_ws->result->x)[41];
  CPG_Prim.Q[66] = MAT_BUFD(ecvxcone_ws->result->x)[45];
  CPG_Prim.Q[67] = MAT_BUFD(ecvxcone_ws->result->x)[46];
  CPG_Prim.Q[68] = MAT_BUFD(ecvxcone_ws->result->x)[47];
  CPG_Prim.Q[69] = MAT_BUFD(ecvxcone_ws->result->x)[48];
  CPG_Prim.Q[70] = MAT_BUFD(ecvxcone_ws->result->x)[7];
  CPG_Prim.Q[71] = MAT_BUFD(ecvxcone_ws->result->x)[16];
  CPG_Prim.Q[72] = MAT_BUFD(ecvxcone_ws->result->x)[24];
  CPG_Prim.Q[73] = MAT_BUFD(ecvxcone_ws->result->x)[31];
  CPG_Prim.Q[74] = MAT_BUFD(ecvxcone_ws->result->x)[37];
  CPG_Prim.Q[75] = MAT_BUFD(ecvxcone_ws->result->x)[42];
  CPG_Prim.Q[76] = MAT_BUFD(ecvxcone_ws->result->x)[46];
  CPG_Prim.Q[77] = MAT_BUFD(ecvxcone_ws->result->x)[49];
  CPG_Prim.Q[78] = MAT_BUFD(ecvxcone_ws->result->x)[50];
  CPG_Prim.Q[79] = MAT_BUFD(ecvxcone_ws->result->x)[51];
  CPG_Prim.Q[80] = MAT_BUFD(ecvxcone_ws->result->x)[8];
  CPG_Prim.Q[81] = MAT_BUFD(ecvxcone_ws->result->x)[17];
  CPG_Prim.Q[82] = MAT_BUFD(ecvxcone_ws->result->x)[25];
  CPG_Prim.Q[83] = MAT_BUFD(ecvxcone_ws->result->x)[32];
  CPG_Prim.Q[84] = MAT_BUFD(ecvxcone_ws->result->x)[38];
  CPG_Prim.Q[85] = MAT_BUFD(ecvxcone_ws->result->x)[43];
  CPG_Prim.Q[86] = MAT_BUFD(ecvxcone_ws->result->x)[47];
  CPG_Prim.Q[87] = MAT_BUFD(ecvxcone_ws->result->x)[50];
  CPG_Prim.Q[88] = MAT_BUFD(ecvxcone_ws->result->x)[52];
  CPG_Prim.Q[89] = MAT_BUFD(ecvxcone_ws->result->x)[53];
  CPG_Prim.Q[90] = MAT_BUFD(ecvxcone_ws->result->x)[9];
  CPG_Prim.Q[91] = MAT_BUFD(ecvxcone_ws->result->x)[18];
  CPG_Prim.Q[92] = MAT_BUFD(ecvxcone_ws->result->x)[26];
  CPG_Prim.Q[93] = MAT_BUFD(ecvxcone_ws->result->x)[33];
  CPG_Prim.Q[94] = MAT_BUFD(ecvxcone_ws->result->x)[39];
  CPG_Prim.Q[95] = MAT_BUFD(ecvxcone_ws->result->x)[44];
  CPG_Prim.Q[96] = MAT_BUFD(ecvxcone_ws->result->x)[48];
  CPG_Prim.Q[97] = MAT_BUFD(ecvxcone_ws->result->x)[51];
  CPG_Prim.Q[98] = MAT_BUFD(ecvxcone_ws->result->x)[53];
  CPG_Prim.Q[99] = MAT_BUFD(ecvxcone_ws->result->x)[54];
  CPG_Prim.R[0] = MAT_BUFD(ecvxcone_ws->result->x)[76];
  CPG_Prim.R[1] = MAT_BUFD(ecvxcone_ws->result->x)[77];
  CPG_Prim.R[2] = MAT_BUFD(ecvxcone_ws->result->x)[78];
  CPG_Prim.R[3] = MAT_BUFD(ecvxcone_ws->result->x)[79];
  CPG_Prim.R[4] = MAT_BUFD(ecvxcone_ws->result->x)[80];
  CPG_Prim.R[5] = MAT_BUFD(ecvxcone_ws->result->x)[81];
  CPG_Prim.R[6] = MAT_BUFD(ecvxcone_ws->result->x)[82];
  CPG_Prim.R[7] = MAT_BUFD(ecvxcone_ws->result->x)[83];
  CPG_Prim.R[8] = MAT_BUFD(ecvxcone_ws->result->x)[84];
  CPG_Prim.R[9] = MAT_BUFD(ecvxcone_ws->result->x)[85];
  CPG_Prim.R[10] = MAT_BUFD(ecvxcone_ws->result->x)[86];
  CPG_Prim.R[11] = MAT_BUFD(ecvxcone_ws->result->x)[87];
  CPG_Prim.R[12] = MAT_BUFD(ecvxcone_ws->result->x)[88];
  CPG_Prim.R[13] = MAT_BUFD(ecvxcone_ws->result->x)[89];
  CPG_Prim.R[14] = MAT_BUFD(ecvxcone_ws->result->x)[90];
  CPG_Prim.R[15] = MAT_BUFD(ecvxcone_ws->result->x)[91];
  CPG_Prim.R[16] = MAT_BUFD(ecvxcone_ws->result->x)[92];
  CPG_Prim.R[17] = MAT_BUFD(ecvxcone_ws->result->x)[93];
  CPG_Prim.R[18] = MAT_BUFD(ecvxcone_ws->result->x)[94];
  CPG_Prim.R[19] = MAT_BUFD(ecvxcone_ws->result->x)[95];
  CPG_Prim.R[20] = MAT_BUFD(ecvxcone_ws->result->x)[96];
  CPG_Prim.R[21] = MAT_BUFD(ecvxcone_ws->result->x)[97];
  CPG_Prim.R[22] = MAT_BUFD(ecvxcone_ws->result->x)[98];
  CPG_Prim.R[23] = MAT_BUFD(ecvxcone_ws->result->x)[99];
  CPG_Prim.R[24] = MAT_BUFD(ecvxcone_ws->result->x)[100];
  CPG_Prim.R[25] = MAT_BUFD(ecvxcone_ws->result->x)[101];
  CPG_Prim.R[26] = MAT_BUFD(ecvxcone_ws->result->x)[102];
  CPG_Prim.R[27] = MAT_BUFD(ecvxcone_ws->result->x)[103];
  CPG_Prim.R[28] = MAT_BUFD(ecvxcone_ws->result->x)[104];
  CPG_Prim.R[29] = MAT_BUFD(ecvxcone_ws->result->x)[105];
  CPG_Prim.R[30] = MAT_BUFD(ecvxcone_ws->result->x)[106];
  CPG_Prim.R[31] = MAT_BUFD(ecvxcone_ws->result->x)[107];
  CPG_Prim.R[32] = MAT_BUFD(ecvxcone_ws->result->x)[108];
  CPG_Prim.R[33] = MAT_BUFD(ecvxcone_ws->result->x)[109];
  CPG_Prim.R[34] = MAT_BUFD(ecvxcone_ws->result->x)[110];
  CPG_Prim.R[35] = MAT_BUFD(ecvxcone_ws->result->x)[111];
  CPG_Prim.R[36] = MAT_BUFD(ecvxcone_ws->result->x)[112];
  CPG_Prim.R[37] = MAT_BUFD(ecvxcone_ws->result->x)[113];
  CPG_Prim.R[38] = MAT_BUFD(ecvxcone_ws->result->x)[114];
  CPG_Prim.R[39] = MAT_BUFD(ecvxcone_ws->result->x)[115];
  CPG_Prim.R[40] = MAT_BUFD(ecvxcone_ws->result->x)[116];
  CPG_Prim.R[41] = MAT_BUFD(ecvxcone_ws->result->x)[117];
  CPG_Prim.R[42] = MAT_BUFD(ecvxcone_ws->result->x)[118];
  CPG_Prim.R[43] = MAT_BUFD(ecvxcone_ws->result->x)[119];
  CPG_Prim.R[44] = MAT_BUFD(ecvxcone_ws->result->x)[120];
  CPG_Prim.R[45] = MAT_BUFD(ecvxcone_ws->result->x)[121];
  CPG_Prim.R[46] = MAT_BUFD(ecvxcone_ws->result->x)[122];
  CPG_Prim.R[47] = MAT_BUFD(ecvxcone_ws->result->x)[123];
  CPG_Prim.R[48] = MAT_BUFD(ecvxcone_ws->result->x)[124];
  CPG_Prim.R[49] = MAT_BUFD(ecvxcone_ws->result->x)[125];
  CPG_Prim.R[50] = MAT_BUFD(ecvxcone_ws->result->x)[126];
  CPG_Prim.R[51] = MAT_BUFD(ecvxcone_ws->result->x)[127];
  CPG_Prim.R[52] = MAT_BUFD(ecvxcone_ws->result->x)[128];
  CPG_Prim.R[53] = MAT_BUFD(ecvxcone_ws->result->x)[129];
  CPG_Prim.R[54] = MAT_BUFD(ecvxcone_ws->result->x)[130];
  CPG_Prim.R[55] = MAT_BUFD(ecvxcone_ws->result->x)[131];
  CPG_Prim.R[56] = MAT_BUFD(ecvxcone_ws->result->x)[132];
  CPG_Prim.R[57] = MAT_BUFD(ecvxcone_ws->result->x)[133];
  CPG_Prim.R[58] = MAT_BUFD(ecvxcone_ws->result->x)[134];
  CPG_Prim.R[59] = MAT_BUFD(ecvxcone_ws->result->x)[135];
  CPG_Prim.T[0] = MAT_BUFD(ecvxcone_ws->result->x)[55];
  CPG_Prim.T[1] = MAT_BUFD(ecvxcone_ws->result->x)[56];
  CPG_Prim.T[2] = MAT_BUFD(ecvxcone_ws->result->x)[57];
  CPG_Prim.T[3] = MAT_BUFD(ecvxcone_ws->result->x)[58];
  CPG_Prim.T[4] = MAT_BUFD(ecvxcone_ws->result->x)[59];
  CPG_Prim.T[5] = MAT_BUFD(ecvxcone_ws->result->x)[60];
  CPG_Prim.T[6] = MAT_BUFD(ecvxcone_ws->result->x)[56];
  CPG_Prim.T[7] = MAT_BUFD(ecvxcone_ws->result->x)[61];
  CPG_Prim.T[8] = MAT_BUFD(ecvxcone_ws->result->x)[62];
  CPG_Prim.T[9] = MAT_BUFD(ecvxcone_ws->result->x)[63];
  CPG_Prim.T[10] = MAT_BUFD(ecvxcone_ws->result->x)[64];
  CPG_Prim.T[11] = MAT_BUFD(ecvxcone_ws->result->x)[65];
  CPG_Prim.T[12] = MAT_BUFD(ecvxcone_ws->result->x)[57];
  CPG_Prim.T[13] = MAT_BUFD(ecvxcone_ws->result->x)[62];
  CPG_Prim.T[14] = MAT_BUFD(ecvxcone_ws->result->x)[66];
  CPG_Prim.T[15] = MAT_BUFD(ecvxcone_ws->result->x)[67];
  CPG_Prim.T[16] = MAT_BUFD(ecvxcone_ws->result->x)[68];
  CPG_Prim.T[17] = MAT_BUFD(ecvxcone_ws->result->x)[69];
  CPG_Prim.T[18] = MAT_BUFD(ecvxcone_ws->result->x)[58];
  CPG_Prim.T[19] = MAT_BUFD(ecvxcone_ws->result->x)[63];
  CPG_Prim.T[20] = MAT_BUFD(ecvxcone_ws->result->x)[67];
  CPG_Prim.T[21] = MAT_BUFD(ecvxcone_ws->result->x)[70];
  CPG_Prim.T[22] = MAT_BUFD(ecvxcone_ws->result->x)[71];
  CPG_Prim.T[23] = MAT_BUFD(ecvxcone_ws->result->x)[72];
  CPG_Prim.T[24] = MAT_BUFD(ecvxcone_ws->result->x)[59];
  CPG_Prim.T[25] = MAT_BUFD(ecvxcone_ws->result->x)[64];
  CPG_Prim.T[26] = MAT_BUFD(ecvxcone_ws->result->x)[68];
  CPG_Prim.T[27] = MAT_BUFD(ecvxcone_ws->result->x)[71];
  CPG_Prim.T[28] = MAT_BUFD(ecvxcone_ws->result->x)[73];
  CPG_Prim.T[29] = MAT_BUFD(ecvxcone_ws->result->x)[74];
  CPG_Prim.T[30] = MAT_BUFD(ecvxcone_ws->result->x)[60];
  CPG_Prim.T[31] = MAT_BUFD(ecvxcone_ws->result->x)[65];
  CPG_Prim.T[32] = MAT_BUFD(ecvxcone_ws->result->x)[69];
  CPG_Prim.T[33] = MAT_BUFD(ecvxcone_ws->result->x)[72];
  CPG_Prim.T[34] = MAT_BUFD(ecvxcone_ws->result->x)[74];
  CPG_Prim.T[35] = MAT_BUFD(ecvxcone_ws->result->x)[75];
}

// Retrieve solver info
void cpg_retrieve_info(){
  CPG_Info.obj_val = (ecvxcone_ws->result->primal_objective);
  CPG_Info.iter = ecvxcone_ws->result->iterations;
  CPG_Info.status = ecvxcone_ws->result->status;
  CPG_Info.pri_res = ecvxcone_ws->result->primal_infeasibility;
  CPG_Info.dua_res = ecvxcone_ws->result->dual_infeasibility;
}

// Copy canonical parameters for preconditioning
void cpg_copy_c(){
  for (i=0; i<136; i++){
    Canon_Params_conditioning.c[i] = Canon_Params.c[i];
  }
}

void cpg_copy_A(){
}

void cpg_copy_b(){
}

void cpg_copy_G(){
  for (i=0; i<3973; i++){
    Canon_Params_conditioning.G->x[i] = Canon_Params.G->x[i];
  }
}

void cpg_copy_h(){
  for (i=0; i<973; i++){
    Canon_Params_conditioning.h[i] = Canon_Params.h[i];
  }
}

void cpg_copy_all(){
  cpg_copy_c();
  cpg_copy_A();
  cpg_copy_b();
  cpg_copy_G();
  cpg_copy_h();
}

// Solve via canonicalization, canonical solve, retrieval
void cpg_solve(){

  // Canonicalize if necessary
  if (Canon_Outdated.G) {
    cpg_canonicalize_G();
    cpg_copy_G();
    Canon_Outdated.G = 0; // Reset flag
  }

  if (Canon_Outdated.h) {
    cpg_canonicalize_h();
    cpg_copy_h();
    Canon_Outdated.h = 0; // Reset flag
  }

  if (!ecvxcone_ws ) {
    cpg_copy_all();
    ecvxcone_ws = ecvxcone_setup(136, 973, 0, 3973, 0, &ecvxcone_dims, &ecvxcone_settings);
  }
  // Solve with ECVXCONE
  ecvxcone_flag = conelp(ecvxcone_ws, &ecvxcone_settings);

  // Retrieve results
  cpg_retrieve_prim();
  cpg_retrieve_info();
}

// Update solver settings
void cpg_set_solver_default_settings(){
}
