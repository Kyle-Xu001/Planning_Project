import numpy as np


def calc_coef(time, x):
    n = x.shape[0] - 1  # num of trajectory functions
    Aeq = np.zeros((8*n, 8*n))
    Beq = np.zeros((8*n, 1))

    for i in range(n): # 2n constraints
        # constraint 1: x(t) = waypoints[t.index()] with the traj start point
        Aeq[i, 8*i: 8*i+8] = poly_t(time[i], 0)
        Beq[i] = x[i]

        # constraint 2: x(t) = waypoints[t.index()] with the traj end point
        Aeq[i+n, 8*i: 8*i+8] = poly_t(time[i+1], 0)
        Beq[i+n] = x[i+1]

    for i in range(3): # 6 constraints
        # constraint 3: vel, acc, jerk at time start == 0
        Aeq[2*n+i, 0:8] = poly_t(0, i+1)

        # constraint 4: vel, acc, jerk at time end == 0
        Aeq[2*n+3+i, 8*(n-1):8*n] = poly_t(time[-1], i+1)

    for i in range(n-1): # 6(n-1) constraints
        # constraint 5: The 1st - 6th diffrential difference at intersaction waypoints == 0
        for j in range(6):
            Aeq[2*n+6 + 6*i + j, 8*i: 8*i+8] = poly_t(time[i+1], j+1)
            Aeq[2*n+6 + 6*i + j, 8*(i+1): 8*(i+1)+8] = -poly_t(time[i+1], j+1)

    coefs = np.linalg.solve(Aeq, Beq)
    return coefs


def poly_t(t, k):
    '''
    Calculate kth differential coefficients with time t
    x(t) = c1 + c2*t + c3*t^2 + ... + c8*t_7
    '''
    c = np.ones((8))  # constant coefficents
    # order coefficents, start with [0, 1, 2, 3, 4, 5, 6, 7]
    n = np.linspace(0, 7, 8)

    for i in range(k):
        for j in range(8):
            c[j] = c[j]*n[j]
            n[j] += -1
            if n[j] < 0:
                n[j] = 0

    coefs = np.zeros((8))
    for i in range(8):
        coefs[i] = c[i]*t**n[i]
    return coefs