import importlib
import logging
import cvxpy as cp
from cvxpygen import cpg
import numpy as np
import time
import sys

if __name__ == "__main__":

    '''
    1. Generate Code
    '''

    solver = 'CVXOPT'

    # tracking_err = np.array([0.1, 0.05, 0.1, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.16]) * 1.0
    # tracking_err = np.array([0.15, 0.05, 0.1, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.16]) * 1.0
    # tracking_err = np.array([-0.0158, -0.0417, -0.1517, 0.0032, 0.2703, -0.1057, 0.0472, 0.3559, -0.2925, -0.6624])
    tracking_err = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    roll = tracking_err[1]
    pitch = tracking_err[2]
    yaw = tracking_err[3]

    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    Rzyx = Rz.dot(Ry.dot(Rx))
    # print(f"Rzyx: {Rzyx}")

    # Parameters and variables
    zz_param = cp.Parameter(10, name='tracking_err_square')
    A_param = cp.Parameter((10, 10), name='A')
    B_param = cp.Parameter((10, 6), name='B')

    # Sampling period
    T = 1 / 20  # work in 25 to 30

    # System matrices (continuous-time)
    aA = np.zeros((10, 10))
    aA[0, 6] = 1
    aA[1:4, 7:10] = Rzyx
    aB = np.zeros((10, 6))
    aB[4:, :] = np.eye(6)

    # System matrices (discrete-time)
    B = aB * T
    A = np.eye(10) + T * aA

    alpha = 0.9
    hd = 1e-10
    phi = 0.15

    cc = 0.6
    b1 = 1 / 0.8  # yaw
    b2 = 1 / (1.0 * cc)  # height
    b3 = 1 / 1.5  # velocity
    b4 = 1 / 1

    D = np.matrix([[b2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   # [0, 0, b4, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, b3, 0, 0, 0, 0, 0],
                   [0, 0, 0, b1, 0, 0, 0, 0, 0, 0]])
    c1 = 1 / 25
    c2 = 1 / 50
    C = np.matrix([[c1, 0, 0, 0, 0, 0],
                   [0, c1, 0, 0, 0, 0],
                   [0, 0, c1, 0, 0, 0],
                   [0, 0, 0, c2, 0, 0],
                   [0, 0, 0, 0, c2, 0],
                   [0, 0, 0, 0, 0, c2]])

    Z = cp.diag(zz_param) 

    Q = cp.Variable((10, 10), PSD=True, name='Q')
    T = cp.Variable((6, 6), PSD=True, name='T')
    R = cp.Variable((6, 10), name='R')

    # Constraints
    constraints = [
        cp.bmat([[alpha * Q, (A_param @ Q).T + (B_param @ R).T],
                 [A_param @ Q + B_param @ R, Q / (1 + phi)]]) >> 0,
        cp.bmat([[Q, R.T],
                 [R, T]]) >> 0,
        Q - 10 * Z >> 0,
        np.identity(3) - D @ Q @ D.T >> 0,
        np.identity(6) - C @ T @ C.T >> 0,
        T - hd * np.identity(6) >> 0
    ]

    # Define problem and objective
    problem = cp.Problem(cp.Minimize(0), constraints)
    
    if not problem.is_dpp():
        raise ValueError("Problem is not DPP compliant. Please reformulate the problem" 
                         "to ensure it is DPP compliant.")

    # Set parameter values
    A_param.value = A
    B_param.value = B
    zz_param.value = tracking_err ** 2
    
    # Solve the problem
    # problem.solve(solver=solver, verbose=True)
    
    # if problem.status == 'optimal':
    #     logging.info("Optimization successful.")

    #     optimal_Q = Q.value
    #     optimal_R = R.value

    #     # print(optimal_Q)
    #     # print(optimal_R)

    #     P = np.linalg.inv(optimal_Q)

    #     # Compute aF
    #     aF = np.round(aB @ optimal_R @ P, 4)
    #     Fb2 = aF[6:10, 0:4]

    #     # Compute F_kp
    #     F_kp = -np.block([
    #         [np.zeros((2, 6))],
    #         [np.zeros((4, 2)), Fb2]])
    #     # Compute F_kd
    #     F_kd = -aF[4:10, 4:10]

    #     print(f"Solved F_kp is: {F_kp}")
    #     print(f"Solved F_kd is: {F_kd}")

    #     # Check if the problem is solved successfully
    #     if np.all(np.linalg.eigvals(P) > 0):
    #         logging.info("LMIs feasible")
    #     else:
    #         print("LMIs infeasible")

    #     res = (F_kp, F_kd)
    #     is_solved = True

    # # Failed to solve LMIs
    # else:
    #     print(f"tracking_err: {tracking_err}")
    #     print("Optimization failed.")
    #     res = None
    #     is_solved = False

    # module = importlib.import_module(f'{solver}.cpg_solver')
    # cpg_solve = getattr(module, 'cpg_solve')
    # problem.register_solve('CPG', cpg_solve)
    # generate code
    cpg.generate_code(problem, code_dir='ECVXCONE', solver='ECVXCONE', wrapper=False)

    # '''
    # 2. Solve & Compare
    # '''

    # # solve problem conventionally
    # t0 = time.time()
    # val = problem.solve(solver=solver, verbose=True)
    # t1 = time.time()
    # sys.stdout.write('\nCVXPY\nSolve time: %.3f ms\n' % (1000*(t1-t0)))
    # sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
    # sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
    # sys.stdout.write('Objective function value: %.6f\n' % val)

    # # solve problem with C code via python wrapper
    # t0 = time.time()
    # val = problem.solve(method='CPG', updated_params=['A', 'b'])
    # t1 = time.time()
    # sys.stdout.write('\nCVXPYgen\nSolve time: %.3f ms\n' % (1000 * (t1 - t0)))
    # sys.stdout.write('Primal solution: x = [%.6f, %.6f]\n' % tuple(x.value))
    # sys.stdout.write('Dual solution: d0 = [%.6f, %.6f]\n' % tuple(problem.constraints[0].dual_value))
    # sys.stdout.write('Objective function value: %.6f\n' % val)

