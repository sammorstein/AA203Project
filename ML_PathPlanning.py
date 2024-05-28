'''
Created May 27: RL Path Planning 
- script to do 
'''

import numpy as np
import scipy.optimize
import random


class Drone:
    def __init__(self, position, fuel, payload):
        self.position = position
        self.fuel = fuel
        self.payload = payload
    
    def move(self, move):
        # deterministic, cost scales with the payload and distance desired
        self.position += np.array(move)

        # # potentially have less maneuverability if the payload is heavy/ low on fuel
        # var = 0.1*self.payload/self.fuel
        # self.position += perturb(move, var)
        self.fuel -= np.linalg.norm(move)*self.payload

    def state(self):
        return self.position, self.fuel, self.payload



class DroneEnvironment:
    def __init__(self, gridsize, warehouse_location, houses_location, obstacles):
        '''
        constructor just makes a map and a drone that'll move around in it
        also the warehouse, house location, etc. 
        '''
        self.grid = gridsize
        self.warehouse = warehouse_location
        self.houses = houses_location
        self.drone = Drone(self.warehouse, 10, 0)
        self.obstacles = obstacles
        self.delivered = np.zeros(len(self.houses), dtype = bool)
        self.delivery = self.houses.copy()

    def isdone(self):
        return np.all(self.delivered)

    def reset(self):
        self.drone = Drone(self.warehouse, 10, 1)  # Reset fuel and payload
        self.delivered = np.zeros(len(self.houses), dtype=bool)
        self.delivery = self.houses.copy()
        return self.get_state()
    
    def get_state(self):
        return (tuple(self.drone.position), self.drone.fuel, tuple(self.delivery))

    def step(self, action):
        self.drone.move(action)
        reward = -1  # Cost for moving

        for i, house in enumerate(self.delivery):
            if not self.delivered[i] and np.array_equal(self.drone.position, house):
                self.delivered[i] = True
                reward += 10  # Reward for successful delivery

        return self.get_state(), reward, self.isdone()

    def get_possible_actions(self):
        possible_moves = [
            (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)
        ]  # Up, Down, Left, Right, Down, Up in 3D
        valid_moves = []
        for move in possible_moves:
            new_position = self.drone.position + np.array(move)
            if (0 <= new_position[0] < self.grid[0] and
                0 <= new_position[1] < self.grid[1] and
                0 <= new_position[2] < self.grid[2] and
                not any(np.array_equal(new_position, obs) for obs in self.obstacles)):
                valid_moves.append(move)
        return valid_moves


def perturb(array, variance):
    for i in range(len(array)):
        array[i]*= np.random.normal(1, variance)


# Monte Carlo Reinforcement Learning Agent
class MonteCarloAgent:
    def __init__(self, env, num_episodes, gamma=0.9, epsilon=0.1):
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.returns = {}

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.get_possible_actions())
        else:
            q_values = self.q_table.get(state, {})
            if not q_values:
                return random.choice(self.env.get_possible_actions())
            max_q_value = max(q_values.values())
            actions_with_max_q_value = [action for action, q_value in q_values.items() if q_value == max_q_value]
            return random.choice(actions_with_max_q_value)

    def update_q_table(self, episode):
        G = 0
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            if not self.returns.get((state, action)):
                self.returns[(state, action)] = []
            self.returns[(state, action)].append(G)
            self.q_table.setdefault(state, {})
            self.q_table[state][action] = np.mean(self.returns[(state, action)])

    def train(self):
        for episode_num in range(self.num_episodes):
            episode = []
            state = self.env.reset()
            while not self.env.isdone():
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state
            self.update_q_table(episode)

# Initialize environment
grid_size = (5, 5, 5)  # 5x5x5 grid
warehouse_pos = (0, 0, 0)
houses_location = [(2, 2, 2), (4, 4, 4), (1, 3, 1)]
obstacles = [(2, 2, 1), (3, 3, 3)]  # Example obstacles
env = DroneEnvironment(grid_size, warehouse_pos, houses_location, obstacles)

# Initialize and train agent
num_episodes = 1000
agent = MonteCarloAgent(env, num_episodes)
agent.train()

# Example usage
state = env.reset()
done = False
while not done:
    action = agent.choose_action(state)
    state, reward, done = env.step(action)
    print(f"Drone moved to {state[0]}, Remaining deliveries: {state[2]}, Reward: {reward}")