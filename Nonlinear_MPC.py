import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def solve_mpc(initial_state, target_state, Q, Qn, R, N=50, T=0.1):
    # Constants
    m = 2.05
    kF = 1.0
    kM = 1.0
    L = 0.07
    Ixx = 6.622e-3
    Iyy = 6.616e-3
    Izz = 1.240e-2
    g = 9.81

    # State and control variables
    x, y, z, phi, psi, theta, x_dot, y_dot, z_dot, p, q, r = ca.MX.sym('x'), ca.MX.sym('y'), ca.MX.sym('z'), ca.MX.sym('phi'), ca.MX.sym('theta'), ca.MX.sym('psi'), ca.MX.sym('x_dot'), ca.MX.sym('y_dot'), ca.MX.sym('z_dot'), ca.MX.sym('p'), ca.MX.sym('q'), ca.MX.sym('r')
    w1, w2, w3, w4 = ca.MX.sym('w1'), ca.MX.sym('w2'), ca.MX.sym('w3'), ca.MX.sym('w4')

    state = ca.vertcat(x, y, z, phi, psi, theta, x_dot, y_dot, z_dot, p, q, r)
    control = ca.vertcat(w1, w2, w3, w4)

    # Dynamics equations
    sigma1 = kF * (w1 + w2 + w3 + w4)

    xdot = ca.vertcat(
        x_dot,
        y_dot,
        z_dot,
        p * ca.cos(phi) + r * ca.sin(psi),
        q - r * ca.cos(psi) * ca.tan(phi) + p * ca.sin(psi) * ca.tan(phi),
        (r * ca.cos(psi) - p * ca.sin(psi)) / ca.cos(phi),
        (ca.cos(theta) * ca.sin(psi) + ca.cos(psi) * ca.sin(phi) * ca.sin(theta)) * sigma1 / m,
        (ca.sin(psi) * ca.sin(theta) - ca.cos(psi) * ca.cos(theta) * ca.sin(phi)) * sigma1 / m,
        ca.cos(phi) * ca.cos(psi) * sigma1 / m - g,
        (Iyy * q * r - Izz * q * r + L * kF * w2 - L * kF * w4) / Ixx,
        -(Ixx * p * r - Izz * p * r + L * kF * w1 + L * kF * w3) / Iyy,
        (kM * w1 - kM * w2 + kM * w3 - kM * w4 + Ixx * p * q - Iyy * p * q) / Izz
    )

    # Function to compute the state derivatives
    f = ca.Function('f', [state, control], [xdot])

    # Define optimization variables
    U = ca.MX.sym('U', 4, N)
    X = ca.MX.sym('X', 12, N+1)

    # Initial state parameter
    state0 = ca.MX.sym('state0', 12)

    # Define the cost function
    cost = 0
    g = []

    # Initial state constraint (use the actual initial state)
    g.append(X[:, 0] - state0)

    # Define the dynamics constraints and cost
    for k in range(N):
        cost += ca.mtimes([(X[:, k] - target_state).T, Q, (X[:, k] - target_state)]) + ca.mtimes([U[:, k].T, R, U[:, k]])
        x_next = X[:, k] + T * f(X[:, k], U[:, k])
        g.append(X[:, k+1] - x_next)

    # Terminal cost
    cost += ca.mtimes([(X[:, N] - target_state).T, Qn, (X[:, N] - target_state)])

    # Create a single vector of decision variables
    OPT_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    # Define the problem
    nlp_prob = {'f': cost, 'x': OPT_variables, 'p': state0, 'g': ca.vertcat(*g)}

    # Define the solver options
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 1000}

    # Create the solver
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    # Define the constraints bounds
    lbg = [0] * (12 * (N + 1))
    ubg = [0] * (12 * (N + 1))

    # Add positive control constraints
    lbx = -np.inf * np.ones((12*(N+1) + 4*N, 1))
    ubx = np.inf * np.ones((12*(N+1) + 4*N, 1))

    for i in range(N):
        lbx[12*(N+1) + 4*i : 12*(N+1) + 4*(i+1)] = 0  # Ensure controls are non-negative

    # Initial guess for the optimization variables
    x0 = np.zeros((12, N+1))
    u0 = np.zeros((4, N))
    init_guess = np.concatenate([x0.flatten(), u0.flatten()])

    # Solve the problem
    sol = solver(x0=init_guess, p=initial_state, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

    # Extract the solution
    sol_x = np.array(sol['x'])

    # Extract the state trajectory
    X_sol = sol_x[:12*(N+1)].reshape((N+1, 12)).T
    # Extract the control inputs
    U_sol = sol_x[12*(N+1):].reshape((N, 4)).T

    return X_sol, U_sol

def plot_results(X_sol, U_sol):
    # Extract x, y, z values from X_sol
    x_values = X_sol[0, :]
    y_values = X_sol[1, :]
    z_values = X_sol[2, :]

    # Plot the x, y, and z values
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(x_values, label='x')
    plt.title('X Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(y_values, label='y')
    plt.title('Y Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(z_values, label='z')
    plt.title('Z Values')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 3D Plot of the x, y, and z values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_values, y_values, z_values, label='Trajectory')
    ax.set_title('3D Trajectory of x, y, z')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

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

# Example usage
# Define the MPC problem
N = 50
T = 0.1
Q = 10*np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
Qn = 10000*np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
R = 1*np.diag([1, 1, 1, 1])
initial_state = np.array([4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0])
target_state = np.array([2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
X_sol, U_sol = solve_mpc(initial_state, target_state,Q,Qn,R,N,T)
plot_results(X_sol, U_sol)
