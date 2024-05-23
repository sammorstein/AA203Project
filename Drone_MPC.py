import numpy as np
import cvxpy as cp
from sympy import symbols, Matrix, sin, cos, tan
# Plot the results
import matplotlib.pyplot as plt

# Define the state and input symbols
x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r = symbols('x y z phi theta psi x_dot y_dots z_dot p q r')
w1, w2, w3, w4 = symbols('w1 w2 w3 w4')
m, kF, kM, L, Ixx, Iyy, Izz, g, dt = symbols('m kF kM L Ixx Iyy Izz g dt')

state = Matrix([x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r])
control = Matrix([w1, w2, w3, w4])

# Define the nonlinear dynamics
sigma1 = kF * (w1 + w2 + w3 + w4)

xdot = Matrix([
    x_dot,
    y_dot,
    *z_dot,
    (p * cos(phi) + r * sin(psi)),
    (q - r * cos(psi) * tan(phi) + p * sin(psi) * tan(phi)),
    ((r * cos(psi) - p * sin(psi)) / cos(phi)),
    ((cos(theta) * sin(psi) + cos(psi) * sin(phi) * sin(theta)) * sigma1 / m),
    ((sin(psi) * sin(theta) - cos(psi) * cos(theta) * sin(phi)) * sigma1 / m),
    (cos(phi) * cos(psi) * sigma1 / m - g),
    ((Iyy * q * r - Izz * q * r + L * kF * w2 - L * kF * w4) / Ixx),
    (-(Ixx * p * r - Izz * p * r + L * kF * w1 + L * kF * w3) / Iyy),
    ((kM * w1 - kM * w2 + kM * w3 - kM * w4 + Ixx * p * q - Iyy * p * q) / Izz)
])

# # Compute Jacobians
A_sym = xdot.jacobian(state)
B_sym = xdot.jacobian(control)

print(A_sym)
print(B_sym)
#
# # Function to evaluate the Jacobians at a specific state and input
def compute_jacobians(state_val, control_val, params):
    # print(A_sym)
    subs = {x: state_val[0], y: state_val[1], z: state_val[2],
            phi: state_val[3], theta: state_val[4], psi: state_val[5],
            x_dot: state_val[6], y_dot: state_val[7], z_dot: state_val[8],
            p: state_val[9], q: state_val[10], r: state_val[11],
            w1: control_val[0], w2: control_val[1], w3: control_val[2], w4: control_val[3],
            m: params['m'], kF: params['kF'], kM: params['kM'], L: params['L'],
            Ixx: params['Ixx'], Iyy: params['Iyy'], Izz: params['Izz'], g: params['g'], dt: params['dt']}
    A_num = np.array(A_sym.subs(subs)).astype(np.float64)
    B_num = np.array(B_sym.subs(subs)).astype(np.float64)
    return A_num, B_num

# circular trajectory
T = 30 #simulate for T seconds
res = 100 #divide each second into res intervals
t = np.linspace(0, T+2/res, res*T+2)

circular = np.zeros([4*4, np.size(t)]) #flat outputs x, y, z, theta and their 1st, 2nd and 3rd derivatives
cycles = 2 #go around this many times
trajw = 2*np.pi*res/np.size(t)*cycles #frequency
pathradius = 1
tilt_amplitude = 0.1

for i in range(0, np.size(t)):
  circular[:, i] = np.transpose(np.array([pathradius*np.cos(trajw*t[i]), pathradius*np.sin(trajw*t[i]), tilt_amplitude*np.sin(trajw*t[i]/2), trajw*t[i], -pathradius*trajw*np.sin(trajw*t[i]), pathradius*trajw*np.cos(trajw*t[i]), tilt_amplitude*trajw/4*np.cos(trajw*t[i]/2),  trajw, -pathradius*trajw**2*np.cos(trajw*t[i]), -pathradius*trajw**2*np.sin(trajw*t[i]), -tilt_amplitude*trajw**2*np.sin(trajw*t[i]/2)/8, 0, pathradius*trajw**3*np.sin(trajw*t[i]), -pathradius*trajw**3*np.cos(trajw*t[i]), -tilt_amplitude*trajw**3*np.cos(trajw*t[i]/2)/16, 0]))

ref_trajectory = circular[:4, :]  # Use position and orientation (x, y, z, theta)

# System parameters
params = {
    'm': 2.05,
    'kF': 1.0,
    'kM': 1.0,
    'L': 0.07,
    'Ixx': 6.622e-3,
    'Iyy': 6.616e-3,
    'Izz': 1.240e-2,
    'g': 9.81,
    'dt': 1 / res
}

def do_mpc(
    x_init: np.ndarray,
    u_prev: np.ndarray,
    ref: np.ndarray,
    t_start: int,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    N: int,
    rx: float,
    ru: float,
    rf: float,
) -> tuple[np.ndarray, np.ndarray, str]:

    ref = ref[:,t_start:t_start+N+1]

    #C matrix
    C = np.zeros((4,12))
    C[0,0] = 1
    C[1,1] = 1
    C[2,2] = 1
    C[3,4] = 1

    # Optimization variables
    x_cvx = cp.Variable((N+1, nx))
    # y_cvx = cp.Variable((N+1, 4))
    y_cvx = np.zeros((N+1,4))
    u_cvx = cp.Variable((N, nu))
    # u_cvx = np.zeros((N,nu))
    # u_cvx[0] = u0
    du = cp.Variable((N,nu))

    # u_init = np.ones((nu, N))
    # u_cvx.value = u_init
    # x_cvx.value = np.zeros((nx,N+1))
    # Cost function and constraints
    cost = 0
    constraints = []
    constraints += [x_cvx[0] == x_init]
    # constraints += [u_cvx[:, 0] == u0]
    constraints += [cp.norm_inf(x_cvx[N]) <= rf]
    for k in range(N):
        # if t == 0:
        #     A, B = compute_jacobians(x0, u[:, t], params)
        # else:
        y_cvx = C @ x_cvx[k]
        # constraints += [y_cvx[k] == (C @ x_cvx[k])]
        cost += cp.quad_form(y_cvx - ref[:,k],Q) + cp.quad_form(du[k],R)
        cost += cp.quad_form(C@x_cvx[k] - ref[:, k], Q) + cp.quad_form(du[k], R)
        # cost += cp.sum_squares(x_var[0:3, t+1] - ref_trajectory[t, 0:3])
        # cost += cp.sum_squares(u[:, t])

        if k > 0:
            constraints += [u_cvx[k] == u_cvx[k - 1] + du[k]]
        else:
            constraints += [u_cvx[0] == u_prev + du[k]]
        #     u_cvx[k] == u_cvx[k - 1] + du[k]
        # else:
        #     u_cvx[0] == u0 + du[k]

        # if k > 0:
        #     constraints += [u_cvx[k] == du[k]]

        # constraints += [cp.norm(u[:, t], 'inf') <= 1]
        constraints += [x_cvx[k + 1] == A @ x_cvx[k] + B @ u_cvx[k]]
        constraints += [cp.norm_inf(x_cvx[k]) <= rx]
        constraints += [cp.norm_inf(u_cvx[k]) <= ru]
        for j in range(nu):
            constraints += [u_cvx[k, j] >= 0]



    # Solve the optimization problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    # prob.solve(verbose=True)
    prob.solve()
    # Extract the optimal control inputs
    u_opt = u_cvx.value
    # u_opt = u_cvx
    x_opt = x_cvx.value
    status = prob.status

    return x_opt, u_opt, status

# Initial state
x0 = np.zeros(12)
x0[0]=1.0
u0 = np.ones(4)
# MPC parameters
N = 3
# dt = 0.1
nx = 12
nu = 4
rx = np.inf
# ru = 0.5
ru = 10.0
rf = np.inf
Q = np.eye(np.shape(ref_trajectory)[0])
R = np.eye(nu)
# T = np.shape(ref_trajectory)[1]
T = 100

x_mpc = np.zeros((T, N + 1, nx))
u_mpc = np.zeros((T, N, nu))

x_cur = x0
u = u0
A, B = compute_jacobians(x_cur, u, params)
states = [x0]
for t in range(T):
    x_mpc[t], u_mpc[t], status = do_mpc(x_cur,u,ref_trajectory,t,A,B,Q,R,N,rx,ru,rf)
    if status == "infeasible":
        x_mpc = x_mpc[:t]
        u_mpc = u_mpc[:t]
        break
    A, B = compute_jacobians(x_cur, u_mpc[t,0,: ], params)
    # print(A,B)
    x_cur = A @ x_cur + B @ u_mpc[t, 0, :]
    print(x_cur)
    u = u_mpc[t,0,:]
    print(u)
    print(t)
    states.append(x_cur)

#     ax[0, i].plot(x_mpc[t, :, 0], x_mpc[t, :, 1], "--*", color="k")
# ax[0, i].plot(x_mpc[:, 0, 0], x_mpc[:, 0, 1], "-o")
# ax[1, i].plot(u_mpc[:, 0], "-o")

# Simulate the response
# states = [x0]
# for t in range(N):
#     next_state = x_opt[:, t+1]
#     states.append(next_state)

states = np.array(states)


plt.figure()
plt.plot(ref_trajectory[0,:], ref_trajectory[1, :], 'r--', label='Reference Trajectory')
plt.plot(states[:, 0], states[:, 1], 'b-', label='Drone Trajectory')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.show()
