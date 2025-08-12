import importlib
import logging
import cvxpy as cp
from cvxpygen import cpg
import numpy as np
import time
import sys

def cvxpy_ddp(random=False, threshold=0.1):
    '''
    Write your DDP-compliant Python Code here
    '''

    solver = 'CVXOPT'
    
    if random:
        tracking_err = np.random.uniform(-threshold, threshold, size=10)
    else:
        tracking_err = np.array([-0.0158, -0.0417, -0.1517, 0.0032, 0.2703, -0.1057, 0.0472, 0.3559, -0.2925, -0.6624])
        # tracking_err = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

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
    
    # Check if the problem is DPP compliant
    if not problem.is_dpp():
        raise ValueError("Problem is not DPP compliant. Please reformulate the problem" 
                         "to ensure it is DPP compliant.")

    # Set parameter values
    A_param.value = A
    B_param.value = B
    zz_param.value = tracking_err ** 2
    
    # Solve the problem
    problem.solve(solver=solver, verbose=False)
    
    if problem.status == 'optimal':
        logging.info("Optimization successful.")

        optimal_Q = Q.value
        optimal_R = R.value

        # print(optimal_Q)
        # print(optimal_R)

        P = np.linalg.inv(optimal_Q)

        # Compute aF
        aF = np.round(aB @ optimal_R @ P, 4)
        Fb2 = aF[6:10, 0:4]

        # Compute F_kp
        F_kp = -np.block([
            [np.zeros((2, 6))],
            [np.zeros((4, 2)), Fb2]])
        # Compute F_kd
        F_kd = -aF[4:10, 4:10]

        print(f"Solved F_kp is: {F_kp}")
        print(f"Solved F_kd is: {F_kd}")

        # Check if the problem is solved successfully
        if np.all(np.linalg.eigvals(P) > 0):
            logging.info("LMIs feasible")
        else:
            print("LMIs infeasible")

        res = (F_kp, F_kd)
        is_solved = True

    # Failed to solve LMIs
    else:
        print("Optimization failed.")
        res = None
        is_solved = False
        
    return problem, res, is_solved, problem._solve_time
    

def benchmark_ddp(solver_func, repeat=1000, random=False, threshold=0.1):
    solve_time = 0.0
    total_time = 0.0

    for i in range(repeat):
        t1 = time.time()
        print(f"Iteration {i + 1}:")
        _, _, _, elapsed_time = solver_func(random=random, threshold=threshold)
        t2 = time.time()

        solve_time += elapsed_time
        total_time += (t2 - t1)

    avg_time = total_time / repeat * 1e3  # ms
    avg_solve_time = solve_time / repeat * 1e3  # ms

    print(f"\nAverage solution time over {repeat} runs: {avg_time:.6f} ms")
    print(f"Average solve time over {repeat} runs: {avg_solve_time:.6f} ms")



if __name__ == "__main__":

    '''
    1. Generate Code 
    '''
    benchmark_ddp(cvxpy_ddp, repeat=1000, random=False, threshold=0.1)
    
    # REPEAT = 1000  # Test counts
    # solve_time = 0.0
    # total_time = 0.0

    # for i in range(REPEAT):
    #     t1 = time.time()
    #     print(f"Iteration {i + 1}:")
    #     res, is_solved, elapsed_time = cvxpy_ddp(random=False, threshold=0.1)
    #     t2 = time.time()
    #     solve_time += elapsed_time
    #     total_time += (t2 - t1)

    # avg_time = total_time / REPEAT * 1e3
    # avg_solve_time = solve_time / REPEAT * 1e3
    # print(f"Average solution time over {REPEAT} runs: {avg_time:.6f} milliseconds")
    # print(f"Average solve time over {REPEAT} runs: {avg_solve_time:.6f} milliseconds")

    problem, res, is_solved, elapsed_time = cvxpy_ddp()
    # module = importlib.import_module(f'{solver}.cpg_solver')
    # cpg_solve = getattr(module, 'cpg_solve')
    # problem.register_solve('CPG', cpg_solve)
    # generate code
    cpg.generate_code(problem, code_dir='ECVXCONE', solver='ECVXCONE', wrapper=False)
