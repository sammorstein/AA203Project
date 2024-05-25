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

def quad_control(sdes, dsdes, ddsdes, dddsdes, x):

    m = 2.05
    kF = 1.0
    kM = 1.0
    L = 0.07
    Ixx = 6.622e-3
    Iyy = 6.616e-3
    Izz = 1.240e-2
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


def solve_mpc(initial_state, reference_trajectory, prev_u, Q, Qn, R, N=50, T=0.1):
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
    x, y, z, phi, psi, theta, x_dot, y_dot, z_dot, p, q, r = ca.MX.sym('x'), ca.MX.sym('y'), ca.MX.sym('z'), ca.MX.sym('phi'), ca.MX.sym('psi'), ca.MX.sym('theta'), ca.MX.sym('x_dot'), ca.MX.sym('y_dot'), ca.MX.sym('z_dot'), ca.MX.sym('p'), ca.MX.sym('q'), ca.MX.sym('r')
    delta_u1, delta_u2, delta_u3, delta_u4 = ca.MX.sym('delta_u1'), ca.MX.sym('delta_u2'), ca.MX.sym('delta_u3'), ca.MX.sym('delta_u4')
    u_prev = ca.MX.sym('u_prev', 4)

    state = ca.vertcat(x, y, z, phi, psi, theta, x_dot, y_dot, z_dot, p, q, r)
    delta_u = ca.vertcat(delta_u1, delta_u2, delta_u3, delta_u4)

    # Dynamics equations
    u = u_prev + delta_u
    sigma1 = kF * (u[0] + u[1] + u[2] + u[3])

    xdot = ca.vertcat(
        x_dot,
        y_dot,
        z_dot,
        p * ca.cos(phi) + r * ca.sin(psi),
        q - r * ca.cos(psi) * ca.tan(phi) + p * ca.sin(psi) * ca.tan(phi),
        (r * ca.cos(psi) - p * ca.sin(psi)) / ca.cos(phi),
        (ca.cos(theta) * ca.sin(psi) + ca.cos(psi) * ca.sin(phi) * ca.sin(theta)) * sigma1 / m,
        (ca.sin(psi) * ca.sin(theta) - ca.cos(psi) * ca.cos(theta) * ca.sin(phi)) * sigma1 / m,
        ca.cos(phi) * ca.cos(psi) * sigma1 / m - m*g,
        (Iyy * q * r - Izz * q * r + L * kF * u[1] - L * kF * u[3]) / Ixx,
        -(Ixx * p * r - Izz * p * r + L * kF * u[0] + L * kF * u[2]) / Iyy,
        (kM * u[0] - kM * u[1] + kM * u[2] - kM * u[3] + Ixx * p * q - Iyy * p * q) / Izz
    )

    # Function to compute the state derivatives
    f = ca.Function('f', [state, delta_u, u_prev], [xdot])

    # Define optimization variables
    Delta_U = ca.MX.sym('Delta_U', 4, N)
    X = ca.MX.sym('X', 12, N+1)
    U_prev = ca.MX.sym('U_prev', 4, N+1)

    # Initial state parameter
    state0 = ca.MX.sym('state0', 12)
    u0 = ca.MX.sym('state0', 4)

    # Define the cost function
    cost = 0
    g = []

    # Initial state constraint (use the actual initial state)
    g.append(X[:, 0] - state0)

    # Initial control input constraint
    g.append(U_prev[:,0] - u0)
    # Define the dynamics constraints and cost
    # u_prev_k = U_prev
    for k in range(N):
        # yk = ca.vertcat(X[0, k], X[1, k], X[2, k], X[5, k])  # [x, y, z, theta]
        yk = ca.vertcat(X[0, k], X[1, k], X[2, k], X[5, k], X[6,k], X[7,k], X[8,k])
        rk = reference_trajectory[:, k]
        cost += ca.mtimes([(yk - rk).T, Q, (yk - rk)]) + ca.mtimes([Delta_U[:, k].T, R, Delta_U[:, k]])
        # cost += ca.mtimes([(X[:,k] - rk).T, Q, (X[:,k] - rk)]) + ca.mtimes([Delta_U[:, k].T, R, Delta_U[:, k]])
        x_next = X[:, k] + T * f(X[:, k], Delta_U[:, k], U_prev[:,k])
        g.append(X[:, k + 1] - x_next)
        g.append(U_prev[:,k+1] - U_prev[:,k] + Delta_U[:, k])

        # if k > 0:
        # g.append(U_prev[0,k+1])
        # g.append(U_prev[1,k+1])
        # g.append(U_prev[2,k+1])
        # g.append(U_prev[3,k+1])

    # Terminal cost
    # yN = ca.vertcat(X[0, N], X[1, N], X[2, N], X[4, N])
    yN = ca.vertcat(X[0, N], X[1, N], X[2, N], X[5, N], X[6, N], X[7, N], X[8, N])
    rN = reference_trajectory[:, N]
    # cost += ca.mtimes([(X[:,N] - rN).T, Qn, (X[:,N] - rN)])
    cost += ca.mtimes([(yN - rN).T, Qn, (yN - rN)])

    # Create a single vector of decision variables
    OPT_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U_prev,-1,1), ca.reshape(Delta_U, -1, 1))

    # Define the problem
    nlp_prob = {'f': cost, 'x': OPT_variables, 'p': ca.vertcat(state0, u0), 'g': ca.vertcat(*g)}

    # Define the solver options
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 1000}

    # Create the solver
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    # Define the constraints bounds
    lbg = [0] * (12 * (N + 1) + 4 * (N+1))
    ubg = [0] * (12 * (N + 1) + 4 * (N+1))

    # # Add control constraints on U_prev
    lbx = -np.inf * np.ones((12 * (N + 1) + 4 * N * 2 + 4, 1))
    ubx = np.inf * np.ones((12 * (N + 1) + 4 * N * 2 + 4, 1))
    #
    for i in range(N):
        lbx[12 * (N + 1) + 4 * i: 12 * (N + 1) + 4 * (i + 1)] = 0  # Ensure controls are non-negative
        ubx[12 * (N + 1) + 4 * i: 12 * (N + 1) + 4 * (i + 1)] = 100

        # Initial guess for the optimization variables
    x0 = np.zeros((12, N + 1))
    u_guess = np.zeros((4,N+1))
    delta_u0 = np.zeros((4, N))
    init_guess = np.concatenate([x0.flatten(), u_guess.flatten(),delta_u0.flatten()])

    # Solve the problem
    sol = solver(x0=init_guess, p=np.concatenate([initial_state, prev_u]), lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

    # Extract the solution
    sol_x = np.array(sol['x'])

    # Extract the state trajectory
    X_sol = sol_x[:12 * (N + 1)].reshape((N + 1, 12)).T
    # Extract the control inputs
    U_sol = sol_x[12 * (N + 1):16 * (N + 1)].reshape((N+1, 4)).T

    # Compute actual control inputs from delta u
    # U_sol = np.zeros_like(Delta_U_sol)
    # u_prev = prev_u
    # for k in range(N):
    #     U_sol[:, k] = u_prev + Delta_U_sol[:, k]
    #     u_prev = U_sol[:, k]

    return X_sol, U_sol

def plot_results(X_sol, U_sol, ref_trajectory):
    # Extract x, y, z values from X_sol
    x_values = X_sol[0, :]
    y_values = X_sol[1, :]
    z_values = X_sol[2, :]
    ref_x_values = ref_trajectory[0, :]
    ref_y_values = ref_trajectory[1, :]
    ref_z_values = ref_trajectory[2, :]

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
def get_dyn(x, u):
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
    phi, psi, theta = x[3], x[4], x[5]
    x_dot, y_dot, z_dot = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]

    sigma1 = kF * (u[0] + u[1] + u[2] + u[3])

    xdot = np.array([
        x_dot,
        y_dot,
        z_dot,
        p * np.cos(phi) + r * np.sin(psi),
        q - r * np.cos(psi) * np.tan(phi) + p * np.sin(psi) * np.tan(phi),
        (r * np.cos(psi) - p * np.sin(psi)) / np.cos(phi),
        (np.cos(theta) * np.sin(psi) + np.cos(psi) * np.sin(phi) * np.sin(theta)) * sigma1 / m,
        (np.sin(psi) * np.sin(theta) - np.cos(psi) * np.cos(theta) * np.sin(phi)) * sigma1 / m,
        np.cos(phi) * np.cos(psi) * sigma1 / m - m*g,
        (Iyy * q * r - Izz * q * r + L * kF * u[1] - L * kF * u[3]) / Ixx,
        -(Ixx * p * r - Izz * p * r + L * kF * u[0] + L * kF * u[2]) / Iyy,
        (kM * u[0] - kM * u[1] + kM * u[2] - kM * u[3] + Ixx * p * q - Iyy * p * q) / Izz
    ])

    return xdot

# Create a reference trajectory for [x, y, z, psi]
# diff = np.linspace(4,2,N+1)
# reference_trajectory = np.vstack((diff,diff,diff,np.zeros(N+1)))

T = 30 #simulate for T seconds
res = 100 #divide each second into res intervals
t = np.linspace(0, T+2/res, res*T+2)

circular = np.zeros([4*4, np.size(t)]) #flat outputs x, y, z, theta and their 1st, 2nd and 3rd derivatives
cycles = 2 #go around this many times
trajw = 2*np.pi*res/np.size(t)*cycles #frequency
pathradius = 1
tilt_amplitude = 0.1

reference_trajectory = np.zeros([12, np.size(t)])
initial_state = np.array([1, 0, 0, -np.pi/7, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(0, np.size(t)):
  circular[:, i] = np.transpose(np.array([pathradius*np.cos(trajw*t[i]), pathradius*np.sin(trajw*t[i]), tilt_amplitude*np.sin(trajw*t[i]/2), trajw*t[i], -pathradius*trajw*np.sin(trajw*t[i]), pathradius*trajw*np.cos(trajw*t[i]), tilt_amplitude*trajw/4*np.cos(trajw*t[i]/2),  trajw, -pathradius*trajw**2*np.cos(trajw*t[i]), -pathradius*trajw**2*np.sin(trajw*t[i]), -tilt_amplitude*trajw**2*np.sin(trajw*t[i]/2)/8, 0, pathradius*trajw**3*np.sin(trajw*t[i]), -pathradius*trajw**3*np.cos(trajw*t[i]), -tilt_amplitude*trajw**3*np.cos(trajw*t[i]/2)/16, 0]))
  if i >0:
    reference_trajectory[:,i] = quad_control(circular[0:4, i], circular[4:8, i], circular[8:12, i], circular[12:16, i],reference_trajectory[:,i-1])
  else:
    reference_trajectory[:, i] = quad_control(circular[0:4, i], circular[4:8, i], circular[8:12, i], circular[12:16, i], initial_state)



ref_trajectory = circular[:7, :]  # Use position and orientation (x, y, z, theta)

N = 500
dt = 1 / res

# Q = 20 * np.diag([1, 1, 1, .1, .1, 1, 1, 1, 1, .1, .1, .1])
# Qn = 20 * np.diag([1, 1, 1, .1, .1, 1, 1, 1, 1, .1, .1, .1])
Q = 20*np.diag([1, 1, 1, 1, 1, 1, 1])
Qn = 20*np.diag([1, 1, 1, 1, 1, 1, 1])
# Q = 20*np.diag([1, 1, 1, 1]) work well for OL
# Qn = 20*np.diag([1, 1, 1, 1])
R = 1*np.diag([1, 1, 1, 1])
# R = np.zeros((4,4))
T = 1
u_prev = 5*np.ones(4)
x_mpc = np.zeros((T, 12, N+1))
u_mpc = np.zeros((T, 4, N+1))
x = initial_state.T
X_sol = x
for t in range(T):
    print(t)
    x_mpc[t], u_mpc[t] = solve_mpc(x, ref_trajectory[:,t:], u_prev, Q, Qn, R, N, dt)
    # x = x+dt*get_dyn(x,u_mpc[t,:,1])
    u_prev = u_mpc[t,:,1]
    X_sol = np.vstack((X_sol,x))
    if t == 0:
        U_sol = u_mpc[t,:,1]
    else:
        U_sol = np.vstack((U_sol,u_mpc[t,:,1]))

    # x_mpc[t], u_mpc[t] = solve_mpc(x, ref_trajectory[:, t + t * (N - 1):], u_prev, Q, Qn, R, N, dt)
    # for i in range(N):
    #     x = x + dt * get_dyn(x, u_mpc[t, :, i + 1])
    #     u_prev = u_mpc[t, :, -1]
    #     X_sol = np.vstack((X_sol, x))
    #     if t == 0 and i == 0:
    #         U_sol = u_mpc[t, :, i]
    #     else:
    #         U_sol = np.vstack((U_sol, u_mpc[t, :, i]))

# u_test = np.array([5.0276,5.0276,5.0276,5.0276])
# x = np.zeros(12)
# X_sol = x
# dt = 0.001
# for t in range(T):
#     x = x + dt * get_dyn(x, u_test)
#     X_sol = np.vstack((X_sol, x))
# print('done')
# plot_results(X_sol.T, U_sol.T,ref_trajectory[:,0:T])
plot_results(x_mpc[0],u_mpc[0],ref_trajectory[:,0:N])
# plot_results(X_sol.T, u_test.T,ref_trajectory[:,0:T])