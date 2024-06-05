import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def rot_X(t):
  return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])

def rot_Y(t):
  return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])

def rot_Z(t):
  return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0 ,0 ,1]])

def get_R(phi, psi, theta):
  '''
  TO DO: define rotation matrix as per convention in [4]
  '''
#   We also use Z − X − Y Euler angles
# to define the roll, pitch, and yaw angles (φ, θ, and ψ) as a
# local coordinate system

  Rz = rot_Z(theta)
  Rx = rot_X(phi)
  Ry = rot_Y(psi)

  R_BW = Rz @ Rx @ Ry

  return R_BW

def quad_control(sdes, dsdes, ddsdes, dddsdes, x,blocks):
    if blocks == 0:
        m = 2.05
        Ixx = 6.622e-3
        Iyy = 6.616e-3
        Izz = 1.240e-2
    if blocks == 1:
        m = 2.55
        Ixx = 7.065e-3
        Iyy = 7.059e-3
        Izz = 1.260e-2
    if blocks == 2:
        m = 3.05
        Ixx = 8.445e-3
        Iyy = 8.439e-3
        Izz = 1.281e-2
    if blocks == 3:
        m = 3.55
        Ixx = 1.139e-2
        Iyy = 1.139e-2
        Izz = 1.302e-2
    kF = 1.0
    kM = 1.0
    L = 0.07
    g = 9.81
    Kp = 50 * np.eye(3)  # for position error
    Kv = 5 * np.eye(3)  # for velocity error

    # let's assume we can observe the full state x but can specify only x, y, z, theta and their derivatives in the trajectory
    R = get_R(x[3], x[4], x[5])
    # PD control of force
    force = -Kp@np.transpose(x[0:3] - sdes[0:3]) - Kv@np.transpose(x[6:9] - dsdes[0:3]) + m*g*np.array([0,0,1]) + m*ddsdes[0:3]
    f = np.dot(force, R[:, 2]) # force magnitude

    # desired orientation (Bdes is desired body frame, Cdes is intermediate frame as defined in [4])
    zBdes = force*1/np.linalg.norm(force)
    xCdes = np.array([np.cos(sdes[3]), np.sin(sdes[3]), 0])
    yBdes = np.cross(zBdes, xCdes)/np.linalg.norm(np.cross(zBdes, xCdes))
    xBdes = np.cross(yBdes, zBdes)
    Rdes = np.column_stack((xBdes, yBdes, zBdes))

    ori = Rotation.from_matrix(Rdes)
    ang = ori.as_euler('zyx')

    hw = m/f*(dddsdes[0:3] - np.dot(zBdes, dddsdes[0:3])*zBdes)

    # desired angular velocity
    omegades = np.array([-np.dot(hw, yBdes), np.dot(hw, xBdes), dsdes[3]*np.dot(np.array([0,0,1]), zBdes)])

    reference_state = np.zeros(12)
    reference_state[0:3] = sdes[0:3]
    reference_state[3] = ang[2]
    reference_state[4] = ang[1]
    reference_state[5] = ang[0]
    reference_state[6:9] = dsdes[0:3]
    reference_state[9:] = omegades

    return reference_state



def solve_mpc(initial_state, reference_trajectory, Q, Qn, R, N=50, T=0.1, blocks=0):
    # Constants
    if blocks == 0:
        m = 2.05
        Ixx = 6.622e-3
        Iyy = 6.616e-3
        Izz = 1.240e-2
    if blocks == 1:
        m = 2.15
        Ixx = 6.711e-3
        Iyy = 6.705e-3
        Izz = 1.243e-2
    if blocks == 2:
        m = 2.25
        Ixx = 6.987e-3
        Iyy = 6.981e-3
        Izz = 1.248e-2
    if blocks == 3:
        m = 2.35
        Ixx = 7.575e-3
        Iyy = 7.569e-3
        Izz = 1.252e-2
    kF = 1.0
    kM = 1.0
    k = 1.0
    L = 0.07
    g = 9.81

    nominal_mv = .25 * m * g * np.array([1, 1, 1, 1])
    # State and control variables
    x, y, z, phi, psi, theta, x_dot, y_dot, z_dot, p, q, r = ca.MX.sym('x'), ca.MX.sym('y'), ca.MX.sym('z'), ca.MX.sym(
        'phi'), ca.MX.sym('psi'), ca.MX.sym('theta'), ca.MX.sym('x_dot'), ca.MX.sym('y_dot'), ca.MX.sym(
        'z_dot'), ca.MX.sym('p'), ca.MX.sym('q'), ca.MX.sym('r')
    w1, w2, w3, w4 = ca.MX.sym('w1'), ca.MX.sym('w2'), ca.MX.sym('w3'), ca.MX.sym('w4')
    # delta_u1, delta_u2, delta_u3, delta_u4 = ca.MX.sym('delta_u1'), ca.MX.sym('delta_u2'), ca.MX.sym('delta_u3'), ca.MX.sym('delta_u4')
    # u_prev = ca.MX.sym('u_prev', 4)

    # state = ca.vertcat(x, y, z, phi, psi, theta, x_dot, y_dot, z_dot, p, q, r)
    state = ca.vertcat(x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r)
    control = ca.vertcat(w1, w2, w3, w4)
    # delta_u = ca.vertcat(delta_u1, delta_u2, delta_u3, delta_u4)

    # Dynamics equations
    # u = u_prev + delta_u
    # sigma1 = kF * (w1 + w2 + w3 + w4)

    # xdot = ca.vertcat(
    #     x_dot,
    #     y_dot,
    #     z_dot,
    #     p * ca.cos(phi) + r * ca.sin(psi),
    #     q - r * ca.cos(psi) * ca.tan(phi) + p * ca.sin(psi) * ca.tan(phi),
    #     (r * ca.cos(psi) - p * ca.sin(psi)) / ca.cos(phi),
    #     (ca.cos(theta) * ca.sin(psi) + ca.cos(psi) * ca.sin(phi) * ca.sin(theta)) * sigma1 / m,
    #     (ca.sin(psi) * ca.sin(theta) - ca.cos(psi) * ca.cos(theta) * ca.sin(phi)) * sigma1 / m,
    #     ca.cos(phi) * ca.cos(psi) * sigma1 / m - m*g,
    #     (Iyy * q * r - Izz * q * r + L * kF * w2 - L * kF * w4) / Ixx,
    #     -(Ixx * p * r - Izz * p * r + L * kF * w1 + L * kF * w3) / Iyy,
    #     (kM * w1 - kM * w2 + kM * w3 - kM * w4 + Ixx * p * q - Iyy * p * q) / Izz
    # )
    xdot = ca.vertcat(
        x_dot,
        y_dot,
        z_dot,
        p,
        r,
        q,
        (k * (ca.sin(phi) * ca.sin(psi) + ca.cos(phi) * ca.cos(psi) * ca.sin(theta)) * (
                w1 + w2 + w3 + w4)) / m,
        -(k * (ca.cos(psi) * ca.sin(phi) - ca.cos(phi) * ca.sin(psi) * ca.sin(theta)) * (
                w1 + w2 + w3 + w4)) / m,
        (k * ca.cos(phi) * ca.cos(theta) * (w1 + w2 + w3 + w4)) / m - g,
        (ca.sin(2 * phi) * ca.sin(theta) * (Iyy / 2 - Izz / 2) * (
                k * L * w1 - k * L * w3 - (Ixx * q ** 2 * ca.sin(2 * theta)) / 2 + (
                Iyy * q ** 2 * ca.sin(2 * theta)) / 2 + Ixx * p * q * ca.cos(
            theta) - Iyy * p * q * ca.cos(theta) + Izz * p * q * ca.cos(
            theta) - Iyy * p * r * ca.sin(2 * phi) + Izz * p * r * ca.sin(
            2 * phi) - Iyy * q ** 2 * ca.cos(phi) ** 2 * ca.cos(theta) * ca.sin(
            theta) + Izz * q ** 2 * ca.cos(phi) ** 2 * ca.cos(theta) * ca.sin(
            theta) + 2 * Iyy * p * q * ca.cos(phi) ** 2 * ca.cos(
            theta) - 2 * Izz * p * q * ca.cos(phi) ** 2 * ca.cos(theta))) / (
                Iyy * Izz * ca.cos(theta)) - ((Iyy * Izz + Ixx * Iyy * ca.sin(theta) ** 2 - Iyy * Izz * ca.sin(
            theta) ** 2 - Ixx * Iyy * ca.sin(phi) ** 2 * ca.sin(theta) ** 2 + Ixx * Izz * ca.sin(
            phi) ** 2 * ca.sin(theta) ** 2) * (k * L * w4 - k * L * w2 - (
                Iyy * r ** 2 * ca.sin(2 * phi)) / 2 + (Izz * r ** 2 * ca.sin(
            2 * phi)) / 2 + Ixx * q * r * ca.cos(theta) - Iyy * q * r * ca.cos(
            theta) + Izz * q * r * ca.cos(theta) + Iyy * q ** 2 * ca.cos(phi) * ca.cos(
            theta) ** 2 * ca.sin(phi) - Izz * q ** 2 * ca.cos(phi) * ca.cos(theta) ** 2 * ca.sin(
            phi) + 2 * Iyy * q * r * ca.cos(phi) ** 2 * ca.cos(
            theta) - 2 * Izz * q * r * ca.cos(phi) ** 2 * ca.cos(theta))) / (
                Ixx * Iyy * Izz * (ca.sin(theta) ** 2 - 1)) - (
                ca.sin(theta) * (Iyy - Iyy * ca.sin(phi) ** 2 + Izz * ca.sin(phi) ** 2) * (
                Ixx * p * r * ca.cos(theta) - Ixx * q * r * ca.sin(
            2 * theta) + Iyy * r ** 2 * ca.cos(phi) * ca.sin(phi) * ca.sin(
            theta) - Izz * r ** 2 * ca.cos(phi) * ca.sin(phi) * ca.sin(
            theta) - Iyy * p * r * ca.cos(phi) ** 2 * ca.cos(
            theta) + Izz * p * r * ca.cos(phi) ** 2 * ca.cos(
            theta) + Iyy * p * r * ca.cos(theta) * ca.sin(
            phi) ** 2 - Izz * p * r * ca.cos(theta) * ca.sin(
            phi) ** 2 - 2 * Iyy * p * q * ca.cos(phi) * ca.cos(theta) ** 2 * ca.sin(
            phi) + 2 * Izz * p * q * ca.cos(phi) * ca.cos(theta) ** 2 * ca.sin(
            phi) + 2 * Izz * q * r * ca.cos(phi) ** 2 * ca.cos(theta) * ca.sin(
            theta) + 2 * Iyy * q * r * ca.cos(theta) * ca.sin(phi) ** 2 * ca.sin(
            theta))) / (Iyy * Izz * (ca.sin(theta) ** 2 - 1)),
        -(Iyy ** 2 * q ** 2 * ca.cos(theta) ** 2 * ca.sin(
            theta) - Iyy ** 2 * p * q * ca.cos(
            theta) ** 2 + Ixx * Iyy * p * q * ca.cos(
            theta) ** 2 + Iyy * Izz * p * q * ca.cos(
            theta) ** 2 - Iyy ** 2 * q ** 2 * ca.cos(phi) ** 2 * ca.cos(theta) ** 2 * ca.sin(
            theta) + Izz ** 2 * q ** 2 * ca.cos(phi) ** 2 * ca.cos(theta) ** 2 * ca.sin(
            theta) + Iyy ** 2 * p * q * ca.cos(phi) ** 2 * ca.cos(
            theta) ** 2 - Izz ** 2 * p * q * ca.cos(phi) ** 2 * ca.cos(
            theta) ** 2 + Iyy * k * L * w1 * ca.cos(theta) - Iyy * k * L * w3 * ca.cos(
            theta) - Ixx * Iyy * q ** 2 * ca.cos(theta) ** 2 * ca.sin(theta) - Iyy * k * L * w1 * ca.cos(
            phi) ** 2 * ca.cos(theta) + Iyy * k * L * w3 * ca.cos(phi) ** 2 * ca.cos(
            theta) + Izz * k * L * w1 * ca.cos(phi) ** 2 * ca.cos(theta) - Izz * k * L * w3 * ca.cos(
            phi) ** 2 * ca.cos(theta) + Ixx * Iyy * q ** 2 * ca.cos(phi) ** 2 * ca.cos(
            theta) ** 2 * ca.sin(theta) - Ixx * Izz * q ** 2 * ca.cos(phi) ** 2 * ca.cos(
            theta) ** 2 * ca.sin(theta) - Ixx * Iyy * p * q * ca.cos(phi) ** 2 * ca.cos(
            theta) ** 2 + Ixx * Izz * p * q * ca.cos(phi) ** 2 * ca.cos(
            theta) ** 2 - Iyy ** 2 * p * r * ca.cos(phi) * ca.cos(theta) * ca.sin(
            phi) + Izz ** 2 * p * r * ca.cos(phi) * ca.cos(theta) * ca.sin(
            phi) + Iyy ** 2 * q * r * ca.cos(phi) * ca.cos(theta) * ca.sin(phi) * ca.sin(
            theta) - Izz ** 2 * q * r * ca.cos(phi) * ca.cos(theta) * ca.sin(phi) * ca.sin(
            theta) + Ixx * Iyy * p * r * ca.cos(phi) * ca.cos(theta) * ca.sin(
            phi) - Ixx * Izz * p * r * ca.cos(phi) * ca.cos(theta) * ca.sin(
            phi) - Iyy * k * L * w2 * ca.cos(phi) * ca.sin(phi) * ca.sin(theta) + Iyy * k * L * w4 * ca.cos(
            phi) * ca.sin(phi) * ca.sin(theta) + Izz * k * L * w2 * ca.cos(phi) * ca.sin(phi) * ca.sin(
            theta) - Izz * k * L * w4 * ca.cos(phi) * ca.sin(phi) * ca.sin(
            theta) - Ixx * Iyy * q * r * ca.cos(phi) * ca.cos(theta) * ca.sin(phi) * ca.sin(
            theta) + Ixx * Izz * q * r * ca.cos(phi) * ca.cos(theta) * ca.sin(phi) * ca.sin(
            theta)) / (Iyy * Izz * ca.cos(theta)),
        ((Izz ** 2 * q * r * ca.sin(2 * theta)) / 2 - Izz ** 2 * p * r * ca.cos(
            theta) - (Ixx * Izz * q * r * ca.sin(2 * theta)) / 2 + (
                 Iyy * Izz * q * r * ca.sin(
             2 * theta)) / 2 + Ixx * Izz * p * r * ca.cos(
            theta) + Iyy * Izz * p * r * ca.cos(theta) - Izz * k * L * w2 * ca.sin(
            theta) + Izz * k * L * w4 * ca.sin(theta) - Iyy ** 2 * p * r * ca.cos(
            phi) ** 2 * ca.cos(theta) + Izz ** 2 * p * r * ca.cos(phi) ** 2 * ca.cos(
            theta) - Iyy * k * L * w2 * ca.cos(phi) ** 2 * ca.sin(theta) + Iyy * k * L * w4 * ca.cos(
            phi) ** 2 * ca.sin(theta) + Izz * k * L * w2 * ca.cos(phi) ** 2 * ca.sin(
            theta) - Izz * k * L * w4 * ca.cos(phi) ** 2 * ca.sin(theta) + Iyy ** 2 * q ** 2 * ca.cos(
            phi) * ca.cos(theta) ** 2 * ca.sin(phi) * ca.sin(theta) - Izz ** 2 * q ** 2 * ca.cos(
            phi) * ca.cos(theta) ** 2 * ca.sin(phi) * ca.sin(theta) - Iyy ** 2 * p * q * ca.cos(
            phi) * ca.cos(theta) ** 2 * ca.sin(phi) + Izz ** 2 * p * q * ca.cos(phi) * ca.cos(
            theta) ** 2 * ca.sin(phi) + Iyy ** 2 * q * r * ca.cos(phi) ** 2 * ca.cos(
            theta) * ca.sin(theta) - Izz ** 2 * q * r * ca.cos(phi) ** 2 * ca.cos(
            theta) * ca.sin(theta) + Ixx * Iyy * p * r * ca.cos(phi) ** 2 * ca.cos(
            theta) - Ixx * Izz * p * r * ca.cos(phi) ** 2 * ca.cos(
            theta) + Iyy * k * L * w1 * ca.cos(phi) * ca.cos(theta) * ca.sin(phi) - Iyy * k * L * w3 * ca.cos(
            phi) * ca.cos(theta) * ca.sin(phi) - Izz * k * L * w1 * ca.cos(phi) * ca.cos(theta) * ca.sin(
            phi) + Izz * k * L * w3 * ca.cos(phi) * ca.cos(theta) * ca.sin(
            phi) - Ixx * Iyy * q ** 2 * ca.cos(phi) * ca.cos(theta) ** 2 * ca.sin(phi) * ca.sin(
            theta) + Ixx * Izz * q ** 2 * ca.cos(phi) * ca.cos(theta) ** 2 * ca.sin(phi) * ca.sin(
            theta) + Ixx * Iyy * p * q * ca.cos(phi) * ca.cos(theta) ** 2 * ca.sin(
            phi) - Ixx * Izz * p * q * ca.cos(phi) * ca.cos(theta) ** 2 * ca.sin(
            phi) - Ixx * Iyy * q * r * ca.cos(phi) ** 2 * ca.cos(theta) * ca.sin(
            theta) + Ixx * Izz * q * r * ca.cos(phi) ** 2 * ca.cos(theta) * ca.sin(theta)) / (
                Iyy * Izz * ca.cos(theta) ** 2)
    )

    # Function to compute the state derivatives
    # f = ca.Function('f', [state, delta_u, u_prev], [xdot])
    f = ca.Function('f', [state, control], [xdot])

    # Define optimization variables
    # Delta_U = ca.MX.sym('Delta_U', 4, N)
    U = ca.MX.sym('U', 4, N)
    X = ca.MX.sym('X', 12, N + 1)
    # U_prev = ca.MX.sym('U_prev', 4, N+1)

    # Initial state parameter
    state0 = ca.MX.sym('state0', 12)
    u0 = ca.MX.sym('state0', 4)

    # Define the cost function
    cost = 0
    g = []

    # Initial state constraint (use the actual initial state)
    g.append(X[:, 0] - state0)

    # Initial control input constraint
    # g.append(U_prev[:,0] - u0)
    # Define the dynamics constraints and cost
    # u_prev_k = U_prev
    for k in range(N):
        # yk = ca.vertcat(X[0, k], X[1, k], X[2, k], X[5, k])  # [x, y, z, theta]
        # yk = ca.vertcat(X[0, k], X[1, k], X[2, k], X[5, k], X[6,k], X[7,k], X[8,k])
        yk = ca.vertcat(X[0, k], X[1, k], X[2, k], X[3, k], X[4, k], X[5, k])
        rk = reference_trajectory[:, k]
        cost += ca.mtimes([(yk - rk).T, Q, (yk - rk)]) + ca.mtimes(
            [(U[:, k] - nominal_mv).T, R, (U[:, k] - nominal_mv)])
        # cost += ca.mtimes([(yk - rk).T, Q, (yk - rk)]) + ca.mtimes([Delta_U[:, k].T, R, Delta_U[:, k]])
        # cost += ca.mtimes([(U_prev[:,k] + Delta_U[:, k] - nominal_mv).T, V,(U_prev[:,k] + Delta_U[:, k] - nominal_mv)])  # Penalize deviation from nominal control target
        # cost += ca.mtimes([(X[:,k] - rk).T, Q, (X[:,k] - rk)]) + ca.mtimes([Delta_U[:, k].T, R, Delta_U[:, k]])
        # x_next = X[:, k] + T * f(X[:, k], Delta_U[:, k], U_prev[:,k])
        x_next = X[:, k] + T * f(X[:, k], U[:, k])
        g.append(X[:, k + 1] - x_next)
        # if k > 0:
        #     delta_u = ca.fabs(U[:, k] - U[:, k - 1])
        #     # delta_u = ca.norm_2(U[:, k] - U[:, k - 1])
        #     g.append(ca.fmax(ca.fmax(delta_u[0],delta_u[1]),ca.fmax(delta_u[2],delta_u[3])) <= 40)
        #     # g.append(U_prev[:,k+1] - U_prev[:,k] - Delta_U[:, k])

    # Terminal cost
    # yN = ca.vertcat(X[0, N], X[1, N], X[2, N], X[4, N])
    # yN = ca.vertcat(X[0, N], X[1, N], X[2, N], X[5, N], X[6, N], X[7, N], X[8, N])
    yN = ca.vertcat(X[0, N], X[1, N], X[2, N], X[3, N], X[4, N], X[5, N])
    rN = reference_trajectory[:, N]
    # cost += ca.mtimes([(X[:,N] - rN).T, Qn, (X[:,N] - rN)])
    cost += ca.mtimes([(yN - rN).T, Qn, (yN - rN)])

    # Create a single vector of decision variables
    # OPT_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U_prev,-1,1), ca.reshape(Delta_U, -1, 1))
    OPT_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    # Define the problem
    # nlp_prob = {'f': cost, 'x': OPT_variables, 'p': ca.vertcat(state0, u0), 'g': ca.vertcat(*g)}
    nlp_prob = {'f': cost, 'x': OPT_variables, 'p': state0, 'g': ca.vertcat(*g)}

    # Define the solver options
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 1000}

    # Create the solver
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    # Define the constraints bounds
    # lbg = [0] * (12 * (N + 1) + 4 * (N+1))
    # ubg = [0] * (12 * (N + 1) + 4 * (N+1))
    lbg = [0] * (12 * (N + 1))
    ubg = [0] * (12 * (N + 1))

    # # Add control constraints on U_prev
    lbx = -np.inf * np.ones((12 * (N + 1) + 4 * N, 1))
    ubx = np.inf * np.ones((12 * (N + 1) + 4 * N, 1))
    #
    # lbx = -np.inf * np.ones((12 * (N + 1) + 4 * N * 2 + 4, 1))
    # ubx = np.inf * np.ones((12 * (N + 1) + 4 * N * 2 + 4, 1))

    for i in range(N):
        lbx[12 * (N + 1) + 4 * i: 12 * (N + 1) + 4 * (i + 1)] = 0  # Ensure controls are non-negative
        ubx[12 * (N + 1) + 4 * i: 12 * (N + 1) + 4 * (i + 1)] = 100

    # for i in range(4 * N):
    #     lbx[12 * (N + 1) + 4 * N + i] = -20
    #     ubx[12 * (N + 1) + 4 * N + i] = 20

    # Initial guess for the optimization variables
    x0 = np.zeros((12, N + 1))
    u_guess = np.zeros((4, N))
    # delta_u0 = np.zeros((4, N))
    # init_guess = np.concatenate([x0.flatten(), u_guess.flatten(),delta_u0.flatten()])
    init_guess = np.concatenate([x0.flatten(), u_guess.flatten()])

    # Solve the problem
    # sol = solver(x0=init_guess, p=np.concatenate([initial_state, prev_u]), lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
    sol = solver(x0=init_guess, p=initial_state, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

    # Extract the solution
    sol_x = np.array(sol['x'])

    # Extract the state trajectory
    X_sol = sol_x[:12 * (N + 1)].reshape((N + 1, 12)).T
    # Extract the control inputs
    U_sol = sol_x[12 * (N + 1):16 * (N + 1)].reshape((N, 4)).T

    return X_sol, U_sol

def plot_results(X_sol, U_sol, ref_trajectory):
    # Extract x, y, z values from X_sol
    x_values = X_sol[0, :]
    y_values = X_sol[1, :]
    z_values = X_sol[2, :]
    phi_values = X_sol[3,:]
    psi_values = X_sol[4,:]
    theta_values = X_sol[5,:]
    ref_x_values = ref_trajectory[0, :]
    ref_y_values = ref_trajectory[1, :]
    ref_z_values = ref_trajectory[2, :]
    ref_phi_values = ref_trajectory[3,:]
    ref_psi_values = ref_trajectory[4,:]
    ref_theta_values = ref_trajectory[5,:]

    # Plot the x, y, and z values
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(x_values, label='x')
    plt.plot(ref_x_values, label='ref_x', linestyle='dashed')
    plt.title('X Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(y_values, label='y')
    plt.plot(ref_y_values, label='ref_y', linestyle='dashed')
    plt.title('Y Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(z_values, label='z')
    plt.plot(ref_z_values, label='ref_z', linestyle='dashed')
    plt.title('Z Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(phi_values, label='phi')
    plt.plot(ref_phi_values, label='ref_phi', linestyle='dashed')
    plt.title('Phi Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(psi_values, label='psi')
    plt.plot(ref_psi_values, label='ref_psi', linestyle='dashed')
    plt.title('Psi Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(theta_values, label='theta')
    plt.plot(ref_theta_values, label='ref_theta', linestyle='dashed')
    plt.title('Theta Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 3D Plot of the x, y, and z values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_values, y_values, z_values, label='Trajectory')
    ax.plot(ref_x_values, ref_y_values, ref_z_values, label='Reference Trajectory', linestyle='dashed')
    ax.set_title('3D Trajectory of x, y, z')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_aspect('equal')

    plt.show()

    # Plot control inputs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Control Inputs')
    plt.plot(U_sol[0], label='w1')
    plt.plot(U_sol[1], label='w2')
    plt.plot(U_sol[2], label='w3')
    plt.plot(U_sol[3], label='w4')
    plt.xlabel('Time Step')
    plt.ylabel('Control Inputs')
    ax.legend()

    plt.show()



# Function to evaluate dynamics given state and input
def get_dyn(x, u, blocks):
    # Constants
    if blocks == 0:
        m = 2.05
        Ixx = 6.622e-3
        Iyy = 6.616e-3
        Izz = 1.240e-2
    if blocks == 1:
        m = 2.15
        Ixx = 6.711e-3
        Iyy = 6.705e-3
        Izz = 1.243e-2
    if blocks == 2:
        m = 2.25
        Ixx = 6.987e-3
        Iyy = 6.981e-3
        Izz = 1.248e-2
    if blocks == 3:
        m = 2.35
        Ixx = 7.575e-3
        Iyy = 7.569e-3
        Izz = 1.252e-2
    kF = 1.0
    kM = 1.0
    k = 1.0
    L = 0.07
    g = 9.81

    # State and control variables
    phi, theta, psi = x[3], x[4], x[5]
    x_dot, y_dot, z_dot = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]

    w1, w2, w3, w4 = u

    # sigma1 = kF * (u[0] + u[1] + u[2] + u[3])
    xdot = np.array([
        x_dot,
        y_dot,
        z_dot,
        p,
        r,
        q,
        (k * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)) * (w1 + w2 + w3 + w4)) / m,
        -(k * (np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(psi) * np.sin(theta)) * (w1 + w2 + w3 + w4)) / m,
        (k * np.cos(phi) * np.cos(theta) * (w1 + w2 + w3 + w4)) / m - g,
        (np.sin(2 * phi) * np.sin(theta) * (Iyy / 2 - Izz / 2) * (k * L * w1 - k * L * w3 - (Ixx * q ** 2 * np.sin(2 * theta)) / 2 + (Iyy * q ** 2 * np.sin(2 * theta)) / 2 + Ixx * p * q * np.cos(theta) - Iyy * p * q * np.cos(theta) + Izz * p * q * np.cos(theta) - Iyy * p * r * np.sin(2 * phi) + Izz * p * r * np.sin(2 * phi) - Iyy * q ** 2 * np.cos(phi) ** 2 * np.cos(theta) * np.sin(theta) + Izz * q ** 2 * np.cos(phi) ** 2 * np.cos(theta) * np.sin(theta) + 2 * Iyy * p * q * np.cos(phi) ** 2 * np.cos(theta) - 2 * Izz * p * q * np.cos(phi) ** 2 * np.cos(theta))) / (Iyy * Izz * np.cos(theta)) - ((Iyy * Izz + Ixx * Iyy * np.sin(theta) ** 2 - Iyy * Izz * np.sin(theta) ** 2 - Ixx * Iyy * np.sin(phi) ** 2 * np.sin(theta) ** 2 + Ixx * Izz * np.sin(phi) ** 2 * np.sin(theta) ** 2) * (k * L * w4 - k * L * w2 - (Iyy * r ** 2 * np.sin(2 * phi)) / 2 + (Izz * r ** 2 * np.sin(2 * phi)) / 2 + Ixx * q * r * np.cos(theta) - Iyy * q * r * np.cos(theta) + Izz * q * r * np.cos(theta) + Iyy * q ** 2 * np.cos(phi) * np.cos(theta) ** 2 * np.sin(phi) - Izz * q ** 2 * np.cos(phi) * np.cos(theta) ** 2 * np.sin(phi) + 2 * Iyy * q * r * np.cos(phi) ** 2 * np.cos(theta) - 2 * Izz * q * r * np.cos(phi) ** 2 * np.cos(theta))) / (Ixx * Iyy * Izz * (np.sin(theta) ** 2 - 1)) - (np.sin(theta) * (Iyy - Iyy * np.sin(phi) ** 2 + Izz * np.sin(phi) ** 2) * (Ixx * p * r * np.cos(theta) - Ixx * q * r * np.sin(2 * theta) + Iyy * r ** 2 * np.cos(phi) * np.sin(phi) * np.sin(theta) - Izz * r ** 2 * np.cos(phi) * np.sin(phi) * np.sin(theta) - Iyy * p * r * np.cos(phi) ** 2 * np.cos(theta) + Izz * p * r * np.cos(phi) ** 2 * np.cos(theta) + Iyy * p * r * np.cos(theta) * np.sin(phi) ** 2 - Izz * p * r * np.cos(theta) * np.sin(phi) ** 2 - 2 * Iyy * p * q * np.cos(phi) * np.cos(theta) ** 2 * np.sin(phi) + 2 * Izz * p * q * np.cos(phi) * np.cos(theta) ** 2 * np.sin(phi) + 2 * Izz * q * r * np.cos(phi) ** 2 * np.cos(theta) * np.sin(theta) + 2 * Iyy * q * r * np.cos(theta) * np.sin(phi) ** 2 * np.sin(theta))) / (Iyy * Izz * (np.sin(theta) ** 2 - 1)),
        -(Iyy ** 2 * q ** 2 * np.cos(theta) ** 2 * np.sin(theta) - Iyy ** 2 * p * q * np.cos(theta) ** 2 + Ixx * Iyy * p * q * np.cos(theta) ** 2 + Iyy * Izz * p * q * np.cos(theta) ** 2 - Iyy ** 2 * q ** 2 * np.cos(phi) ** 2 * np.cos(theta) ** 2 * np.sin(theta) + Izz ** 2 * q ** 2 * np.cos(phi) ** 2 * np.cos(theta) ** 2 * np.sin(theta) + Iyy ** 2 * p * q * np.cos(phi) ** 2 * np.cos(theta) ** 2 - Izz ** 2 * p * q * np.cos(phi) ** 2 * np.cos(theta) ** 2 + Iyy * k * L * w1 * np.cos(theta) - Iyy * k * L * w3 * np.cos(theta) - Ixx * Iyy * q ** 2 * np.cos(theta) ** 2 * np.sin(theta) - Iyy * k * L * w1 * np.cos(phi) ** 2 * np.cos(theta) + Iyy * k * L * w3 * np.cos(phi) ** 2 * np.cos(theta) + Izz * k * L * w1 * np.cos(phi) ** 2 * np.cos(theta) - Izz * k * L * w3 * np.cos(phi) ** 2 * np.cos(theta) + Ixx * Iyy * q ** 2 * np.cos(phi) ** 2 * np.cos(theta) ** 2 * np.sin(theta) - Ixx * Izz * q ** 2 * np.cos(phi) ** 2 * np.cos(theta) ** 2 * np.sin(theta) - Ixx * Iyy * p * q * np.cos(phi) ** 2 * np.cos(theta) ** 2 + Ixx * Izz * p * q * np.cos(phi) ** 2 * np.cos(theta) ** 2 - Iyy ** 2 * p * r * np.cos(phi) * np.cos(theta) * np.sin(phi) + Izz ** 2 * p * r * np.cos(phi) * np.cos(theta) * np.sin(phi) + Iyy ** 2 * q * r * np.cos(phi) * np.cos(theta) * np.sin(phi) * np.sin(theta) - Izz ** 2 * q * r * np.cos(phi) * np.cos(theta) * np.sin(phi) * np.sin(theta) + Ixx * Iyy * p * r * np.cos(phi) * np.cos(theta) * np.sin(phi) - Ixx * Izz * p * r * np.cos(phi) * np.cos(theta) * np.sin(phi) - Iyy * k * L * w2 * np.cos(phi) * np.sin(phi) * np.sin(theta) + Iyy * k * L * w4 * np.cos(phi) * np.sin(phi) * np.sin(theta) + Izz * k * L * w2 * np.cos(phi) * np.sin(phi) * np.sin(theta) - Izz * k * L * w4 * np.cos(phi) * np.sin(phi) * np.sin(theta) - Ixx * Iyy * q * r * np.cos(phi) * np.cos(theta) * np.sin(phi) * np.sin(theta) + Ixx * Izz * q * r * np.cos(phi) * np.cos(theta) * np.sin(phi) * np.sin(theta)) / (Iyy * Izz * np.cos(theta)),
        ((Izz ** 2 * q * r * np.sin(2 * theta)) / 2 - Izz ** 2 * p * r * np.cos(theta) - (Ixx * Izz * q * r * np.sin(2 * theta)) / 2 + (Iyy * Izz * q * r * np.sin(2 * theta)) / 2 + Ixx * Izz * p * r * np.cos(theta) + Iyy * Izz * p * r * np.cos(theta) - Izz * k * L * w2 * np.sin(theta) + Izz * k * L * w4 * np.sin(theta) - Iyy ** 2 * p * r * np.cos(phi) ** 2 * np.cos(theta) + Izz ** 2 * p * r * np.cos(phi) ** 2 * np.cos(theta) - Iyy * k * L * w2 * np.cos(phi) ** 2 * np.sin(theta) + Iyy * k * L * w4 * np.cos(phi) ** 2 * np.sin(theta) + Izz * k * L * w2 * np.cos(phi) ** 2 * np.sin(theta) - Izz * k * L * w4 * np.cos(phi) ** 2 * np.sin(theta) + Iyy ** 2 * q ** 2 * np.cos(phi) * np.cos(theta) ** 2 * np.sin(phi) * np.sin(theta) - Izz ** 2 * q ** 2 * np.cos(phi) * np.cos(theta) ** 2 * np.sin(phi) * np.sin(theta) - Iyy ** 2 * p * q * np.cos(phi) * np.cos(theta) ** 2 * np.sin(phi) + Izz ** 2 * p * q * np.cos(phi) * np.cos(theta) ** 2 * np.sin(phi) + Iyy ** 2 * q * r * np.cos(phi) ** 2 * np.cos(theta) * np.sin(theta) - Izz ** 2 * q * r * np.cos(phi) ** 2 * np.cos(theta) * np.sin(theta) + Ixx * Iyy * p * r * np.cos(phi) ** 2 * np.cos(theta) - Ixx * Izz * p * r * np.cos(phi) ** 2 * np.cos(theta) + Iyy * k * L * w1 * np.cos(phi) * np.cos(theta) * np.sin(phi) - Iyy * k * L * w3 * np.cos(phi) * np.cos(theta) * np.sin(phi) - Izz * k * L * w1 * np.cos(phi) * np.cos(theta) * np.sin(phi) + Izz * k * L * w3 * np.cos(phi) * np.cos(theta) * np.sin(phi) - Ixx * Iyy * q ** 2 * np.cos(phi) * np.cos(theta) ** 2 * np.sin(phi) * np.sin(theta) + Ixx * Izz * q ** 2 * np.cos(phi) * np.cos(theta) ** 2 * np.sin(phi) * np.sin(theta) + Ixx * Iyy * p * q * np.cos(phi) * np.cos(theta) ** 2 * np.sin(phi) - Ixx * Izz * p * q * np.cos(phi) * np.cos(theta) ** 2 * np.sin(phi) - Ixx * Iyy * q * r * np.cos(phi) ** 2 * np.cos(theta) * np.sin(theta) + Ixx * Izz * q * r * np.cos(phi) ** 2 * np.cos(theta) * np.sin(theta)) / (Iyy * Izz * np.cos(theta) ** 2)
    ])

    return xdot

def drone_control(trajectory, initialstate, dt, blocks):

    # reference_trajectory = np.zeros([12, np.shape(trajectory)[1]])
    #
    # for i in range(0, np.shape(trajectory)[1]):
    #     if i > 0:
    #         reference_trajectory[:,i] = quad_control(trajectory[:4,i],trajectory[4:8,i],trajectory[8:12, i],trajectory[12:16,i],reference_trajectory[:,i-1], blocks)
    #     else:
    #         reference_trajectory[:,i] = quad_control(trajectory[:4, i], trajectory[4:8, i], trajectory[8:12, i],trajectory[12:16, i], initialstate, blocks)
    # ref_trajectory = trajectory[:7,:]

    ref_trajectory = np.vstack((trajectory[:3, :], np.zeros((3, np.shape(trajectory)[1]))))

    N = 18
    Q = 10 * np.diag([1, 1, 1, 1, 1, 1])
    Qn = 10 * np.diag([1, 1, 1, 1, 1, 1])
    R = 1 * np.diag([1, 1, 1, 1])
    T = np.shape(ref_trajectory)[1]
    # T =50

    # Pad the reference trajectory with the final position
    final_position = ref_trajectory[:, -1].reshape(-1, 1)
    padding = np.tile(final_position, (1, N))
    ref_trajectory_padded = np.hstack((ref_trajectory, padding))

    x_mpc = np.zeros((T, 12, N + 1))
    u_mpc = np.zeros((T, 4, N))
    x = initialstate.T
    # x = reference_trajectory[:,0]
    X_sol = x

    for t in range(T):
        print(t)
        x_mpc[t], u_mpc[t] = solve_mpc(x, ref_trajectory_padded[:, t:], Q, Qn, R, N, dt, blocks)
        x = x+dt*get_dyn(x,u_mpc[t,:,0], blocks)
        u_prev = u_mpc[t, :, 1]
        X_sol = np.vstack((X_sol, x))
        if t == 0:
            U_sol = u_mpc[t, :, 0]
        else:
            U_sol = np.vstack((U_sol, u_mpc[t, :, 0]))

    plot_results(X_sol.T, U_sol.T,ref_trajectory[:,0:T])
    # plot_results(x_mpc[0], u_mpc[0], ref_trajectory[:, 0:N])

    diff = np.linalg.norm(X_sol[:,:3].T-ref_trajectory_padded[:3,:T+1])
    print(f'Tracking error: {diff}')

    tot_u = np.linalg.norm(U_sol)
    print(f'Total control cost: {tot_u}')

    # return x_mpc, u_mpc
    return X_sol, U_sol


#Create a reference trajectory for [x, y, z, psi]
# diff = np.linspace(4,2,N+1)
# reference_trajectory = np.vstack((diff,diff,diff,np.zeros(N+1)))

# T = 30 #simulate for T seconds
# res = 10 #divide each second into res intervals
# t = np.linspace(0, T+2/res, res*T+2)
#
# circular = np.zeros([4*4, np.size(t)]) #flat outputs x, y, z, theta and their 1st, 2nd and 3rd derivatives
# cycles = 2 #go around this many times
# trajw = 2*np.pi*res/np.size(t)*cycles #frequency
# pathradius = 1
# tilt_amplitude = 0.1
#
# for i in range(0, np.size(t)):
#   circular[:, i] = np.transpose(np.array([pathradius*np.cos(trajw*t[i]), pathradius*np.sin(trajw*t[i]), tilt_amplitude*np.sin(trajw*t[i]/2), trajw*t[i], -pathradius*trajw*np.sin(trajw*t[i]), pathradius*trajw*np.cos(trajw*t[i]), tilt_amplitude*trajw/4*np.cos(trajw*t[i]/2),  trajw, -pathradius*trajw**2*np.cos(trajw*t[i]), -pathradius*trajw**2*np.sin(trajw*t[i]), -tilt_amplitude*trajw**2*np.sin(trajw*t[i]/2)/8, 0, pathradius*trajw**3*np.sin(trajw*t[i]), -pathradius*trajw**3*np.cos(trajw*t[i]), -tilt_amplitude*trajw**3*np.cos(trajw*t[i]/2)/16, 0]))
#
# reference_trajectory = np.zeros([12, np.size(t)])
# initial_state = np.array([1, 0, 0, -np.pi/7, 0, 0, 0, 0, 0, 0, 0, 0])
# u_prev = 5*np.ones(4)
#
# x,u = drone_control(circular,initial_state, 1/res, 0)
# x,u = drone_control(circular,initial_state, 1/res, 1)
# x,u = drone_control(circular,initial_state, 1/res, 2)
# x,u = drone_control(circular,initial_state, 1/res, 3)