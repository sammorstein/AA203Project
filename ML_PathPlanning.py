'''
Created May 27: RL Path Planning 
- script to do 
'''
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

class Drone:
    def __init__(self, position, fuel, payload):
        self.position = np.array(position)
        self.fuel = fuel
        self.payload = np.sum(payload)
    
    def move(self, move):
        newpos = np.add(self.position, np.array(move))
        self.position =newpos
        self.fuel -= np.linalg.norm(move) * self.payload

    def state(self):
        return tuple(self.position), self.fuel, self.payload

class DroneEnvironment:
    def __init__(self, gridsize, warehouse_location, init, houses_location, houses_delivered, obstacles):
        self.grid = gridsize
        self.warehouse = np.array(warehouse_location)
        self.init = np.array(init)
        self.houses = [np.array(house) for house in houses_location]
        self.drone = Drone(self.warehouse, 100, np.array([0.1,0.1,0.1]))
        self.obstacles = [np.array(obstacle) for obstacle in obstacles]
        self.delivered = np.array(houses_delivered)
        self.delivery = self.houses.copy()
        
        self.utility = np.zeros(gridsize)
        self.utility[(0,0,0)] = -100 # ideally should be a function of how much battery left
        for i, h in enumerate(self.delivery):
            if self.delivered[i]:
                self.utility[tuple(h)] = -100
            else: 
                self.utility[tuple(h)] = 90000
        for obstacle in self.obstacles:
            self.utility[tuple(obstacle)] = -100
        # print(self.utility)

    def isdone(self):
        return np.all(self.delivered)

    def reset(self):
        self.drone = Drone(self.warehouse, 10, 0.1)
        self.delivered = np.zeros(len(self.houses), dtype=bool)
        self.delivery = self.houses.copy()
        return self.get_state()
    
    def get_state(self):
        return self.drone.position, self.drone.fuel, self.delivery

    def step(self, action):
        prev_fuel = self.drone.fuel
        action = np.array(action).astype(int)
        self.drone.move(action)
        cost = -(self.drone.fuel - prev_fuel)
        # print('state',self.get_state()[0])
        future_state = self.get_state()
        
        return self.get_state(), self.utility[tuple(future_state[0])]

    def possible_actions(self, pos):
        possible_moves = np.array([[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1] ])
        valid_moves = []
        if pos == None:
            pos = self.drone.position
        for move in possible_moves:
            new_position = np.array(pos) + np.array(move)
            if (0 <= new_position[0] < self.grid[0] and
                0 <= new_position[1] < self.grid[1] and
                0 <= new_position[2] < self.grid[2]):
                    valid_moves.append(move)
        return np.array(valid_moves)

    def check_house(self, house):
        for i in range(len(self.delivery)):

            if tuple(house) == tuple(self.delivery[i]):
                return i
        return 100
    
    def reset_house(self, house):
        house_idx = self.check_house(house)
        if house_idx == 100:
            return "house is not valid. stopped at the wrong location"
        removed_house = self.delivery[house_idx]
        self.delivered[house_idx] = True
        # print(house_idx)
        # self.drone.payload[house_idx] = 0
        print('payload weight has been reduced', self.drone.payload)
        print('now new delivery locations',self.delivered)
        
        self.utility[tuple(removed_house)] = -100




def value_iteration(k_max, drone_mdp, gamma):
    dim = np.array(drone_mdp.grid+(27,4))
    Qvalues = np.empty(dim)
    optimal_utility = np.copy(drone_mdp.utility)
    print(drone_mdp.utility[(0,0,0)])
    alpha = 0.05
    gamma = 0.5
    for k_iter in range(k_max):
        for i in range(dim[0]):
            for j in range(dim[1]):
                for k in range(dim[2]):
                    poss_a = drone_mdp.possible_actions((i,j,k))
                    current_cost = optimal_utility[i,j,k]
                    for a in range(len(poss_a)):
                        next_step_cost = optimal_utility[i+poss_a[a][0], j+poss_a[a][1], k+poss_a[a][2]] #- 0.000001*env.drone.payload*max(np.linalg.norm(poss_a[a]), 0)
                        if a == (0,0,0):
                            next_step_cost+= -10000000
                        Qvalues[i, j, k, a] = np.append(poss_a[a], Qvalues[i, j, k, a, 3] + alpha*(current_cost+gamma*next_step_cost))
                        q__ = Qvalues[i,j,k]
                        optimal_utility[i,j,k] = np.max(q__[:,3])
    return Qvalues, optimal_utility

# Initialize the environment
grid_size = (20, 20, 5)
warehouse_pos = (0, 0, 0)
houses_location = [(0, 19, 0), (15,15,0), (19,0,0)]
deliv = np.array([False, False, False])

obstacles = []

for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        for k in range(grid_size[2]):

            if (5<i<15 and 5<j<15 and k < 3):
                obstacles.append((i, j, k))
            
env = DroneEnvironment(grid_size, warehouse_pos,warehouse_pos, houses_location, deliv, obstacles)

Qvalues, opt_util = value_iteration(50, env, 0.1)
# Assuming Qvalues and grid_size are already defined
x, y, z, q_values = [], [], [], []
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        for k in range(grid_size[2]):
            Q = Qvalues[i, j, k]
            Q = max(Q[:, 3])
            x.append(i)
            y.append(j)
            z.append(k)
            q_values.append(Q)

# Convert lists to numpy arrays
x = np.array(x)
y = np.array(y)
z = np.array(z)
q_values = np.array(q_values)

# Create 3D Isosurface plot
fig = go.Figure(data=go.Isosurface(
    x=x, y=y, z=z,
    value=q_values,
    isomin=min(q_values),
    isomax=max(q_values),
    surface_count=10,  # Number of isosurfaces
    colorscale='Viridis',
    caps=dict(x_show=False, y_show=False, z_show=False)
))

fig.update_layout(
    scene=dict(
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        zaxis_title='Z Coordinate',
        zaxis=dict(range=[0, 10]),
    ),
    title='3D Isosurface Plot of Q-Values'
)

fig.show()


def simulate_optimal_path(Qvalues, warehouse_position, drone):
    current_position = warehouse_position
    path = [current_position]

    for _ in range(100):
        
        Q = Qvalues[current_position]
        max_idx_q = np.argmax(Q[:, 3])
        optimal_action = Q[max_idx_q, :3]
        if tuple(optimal_action) == (0,0,0):
            optimal_action = (0,0,1)
        next_position = tuple(int(x) for x in np.array(current_position) + optimal_action)
        path.append(next_position)

        if env.check_house(next_position) < 10:
            print('this the position???', next_position)
            break
        current_position = next_position
    return drone.payload, path

def monte_carlo_rollout(Qvalues, env):
    mc = 10
    initial_position = env.warehouse
    optimal_path = []
    optimal_utility = -np.inf
    
    for rollout in range(mc):
        env = DroneEnvironment(grid_size, initial_position, houses_location, obstacles)
        utility = 0
        path = [initial_position]
        current_position = initial_position
        for _ in range(100):
            i,j,k = current_position
            Q = Qvalues[i,j,k]
            # optimal actions, but balanced with exploration
            if np.random.random() < 0.8:
                action_idx = np.argmax(Q[:,3])
                Q[action_idx][:3].astype(int)
                if math.isnan(Q[np.argmax(Q[:,3])][3]):
                    print('NaN detected in Q-value during exploitation')
                    print(f'Current Position: {current_position}, Action: {action}')
                    break
                utility += Q[action_idx, 3]
            else:
                choices = env.possible_actions(tuple(current_position))
                dim = np.shape(choices)
                action_idx = np.random.randint(0,dim[1]-1)
                action = choices[action_idx]
                if math.isnan(Q[action_idx, 3]):
                    print('NaN detected in Q-value during exploration')
                    break
                utility += Q[action_idx,3]
            
            state, Q = env.step(action)
            current_position = state[0].astype(int)
            path.append(tuple(current_position))
            # check if position is a house.
            house_idx = env.check_house(tuple(current_position))
            if house_idx < 10:
                env.reset_house(current_position)
                Qvalues, opt_util = value_iteration(20, DroneEnvironment(grid_size, current_position, houses_location, obstacles), 0.1)
                print('reached house: ', current_position)
        # print('iteration done \n utility: ', utility)
        if optimal_utility < utility:
            optimal_utility = utility
            optimal_path = path
            print('optimal changed', optimal_path)
    return optimal_path, optimal_utility
# Simulate the optimal path starting from the warehouse position
optimal_path = []
current_pos = warehouse_pos
# env.reset_house(env.delivery[j])
while not env.delivered.all():

    curr_payload, path_segment = simulate_optimal_path(Qvalues, current_pos, env.drone)
    optimal_path.extend((curr_payload,path_segment))
    current_pos = path_segment[-1]
    if env.check_house(current_pos) < 10:
        '''
        only need to do this if the current end position is a house '''
        env.reset_house(current_pos)
        deliv = env.delivered

        new_env = DroneEnvironment(grid_size, warehouse_pos, current_pos, houses_location, deliv, obstacles)
        Qvalues, opt_util = value_iteration(50,new_env,0.1)
    if iter > 5:
        break
'''
fuel/time cost: 
- sort of works, i guess we need to calculate total fuel cost and time cost
visualize buildings/stay away zones. black out
- done as well. 
path segments into dictionary to take along

some uncertainty => in MPC in xdot Gaussian noises
best way to deliver 5 packages when constrained to 3 each trip. (5 3)

'''

# Print the optimal path
print("Optimal Path:", optimal_path)
# for i, position in enumerate(optimal_path):
#     print(f"Step {i + 1}: {position}")

# Extract x, y, z coordinates from the path
for segment in optimal_path(2):
    x_path = [position[0] for position in optimal_path]
    y_path = [position[1] for position in optimal_path]
    z_path = [position[2] for position in optimal_path]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Normalize the color range to the number of segments
norm = plt.Normalize(0, len(x_path) - 1)
cmap = plt.get_cmap('viridis')  # You can choose any colormap


# Extract x, y, z coordinates from the path
x_path = [position[0] for position in optimal_path]
y_path = [position[1] for position in optimal_path]
z_path = [position[2] for position in optimal_path]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Normalize the color range to the number of segments
norm = plt.Normalize(0, len(x_path) - 1)
cmap = plt.get_cmap('viridis')  # You can choose any colormap

# Plot each segment with a different color
for i in range(len(x_path) - 1):
    ax.plot(
        x_path[i:i+2],
        y_path[i:i+2],
        z_path[i:i+2],
        color=cmap(norm(i)),
        marker='o',
        markersize=5
    )

# Function to create a block
def plot_block(ax, x_start, x_end, y_start, y_end, z_start, z_end, color='black'):
    xx, yy = np.meshgrid([x_start, x_end], [y_start, y_end])
    zz_start = np.full_like(xx, z_start)
    zz_end = np.full_like(xx, z_end)
    
    ax.plot_surface(xx, yy, zz_start, color=color, alpha=0.5)
    ax.plot_surface(xx, yy, zz_end, color=color, alpha=0.5)
    
    yy, zz = np.meshgrid([y_start, y_end], [z_start, z_end])
    xx_start = np.full_like(yy, x_start)
    xx_end = np.full_like(yy, x_end)
    
    ax.plot_surface(xx_start, yy, zz, color=color, alpha=0.5)
    ax.plot_surface(xx_end, yy, zz, color=color, alpha=0.5)
    
    xx, zz = np.meshgrid([x_start, x_end], [z_start, z_end])
    yy_start = np.full_like(xx, y_start)
    yy_end = np.full_like(xx, y_end)
    
    ax.plot_surface(xx, yy_start, zz, color=color, alpha=0.5)
    ax.plot_surface(xx, yy_end, zz, color=color, alpha=0.5)

# Plot the obstacle block
plot_block(ax, 7, 13, 7, 13, 0, 2, color='black')
# plot_block(ax, 0,3,5,15,0,2,color='black')
# plot_block(ax, 17,20,5,15,0,2,color='black')
# plot_block(ax, 5,15,0,3,0,2,color='black')
# plot_block(ax, 5,15,17,20,0,2,color='black')

# Set axis limits
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_zlim(0, 5)

# Set labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
plt.title('Optimal Path in 3D')

# Show the plot
plt.show()
