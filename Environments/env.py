import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FootballCoordinatorEnv(gym.Env):
    def __init__(self):
        super(FootballCoordinatorEnv, self).__init__()

        # Action space: 0 - run play, 1 - pass play
        self.action_space = spaces.Discrete(2)

        # Observation space: down (1-4), distance to go (1-10)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(4),
            spaces.Discrete(10)
        ))

        self.first_down_num = 0

        self.reset()

    def step(self, action):
        # Simulate the result of the play
        yards_gained = self._simulate_play(action)

        # Update the state
        self.distance_to_go -= yards_gained
        if self.distance_to_go <= 0:
            self.down = 1
            self.distance_to_go = 10
            self.first_down_num += 1
        else:
            self.down += 1

        # Calculate the reward
        reward = yards_gained

        # Check if the episode has ended
        done = self.down > 4
        if done:
            self.down = 1

        return (self.down, self.distance_to_go), reward, done, {}

    def reset(self):
        self.down = 1
        self.distance_to_go = 10
        self.first_down_num = 0
        return (self.down, self.distance_to_go)

    def render(self, mode='human'):
        if mode == 'human':
            field = ['[ ]'] * 10
            field[self.distance_to_go - 1] = '[*]'
            field_str = ''.join(field)

            print(f"Down: {self.down} | Distance to go: {self.distance_to_go}")
            print(field_str)
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")

    def _simulate_play(self, action):
        if action == 0:  # run play
            return np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        else:  # pass play
            return np.random.choice([0, 5, 10], p=[0.5, 0.3, 0.2])

class MultiCoordinaterEnv(gym.Env):
    def __init__(self):
        super(MultiCoordinaterEnv, self).__init__()

        # Action space: 0 - run play, 1 - pass play
        self.offense_action_space = spaces.Discrete(2)
        # Action space: 0 - vs run play, 1 - medium play, 2 - vs pass play
        self.defense_action_space = spaces.Discrete(4)

        # Observation space: down (1-4), distance to go (1-10)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(4),
            spaces.Discrete(10)
        ))

        self.first_down_num = 0

        # Yard(Reward) table for the offense agent
        self.yard_table = np.array([[2, 3, 4], [6, 3, 0]])

        self.reset()

    def step(self, offense_action, defense_action):
        # Simulate the result of the play
        yards_gained = self.yard_table[offense_action][defense_action]

        # Update the state
        self.distance_to_go -= yards_gained
        if self.distance_to_go <= 0:
            self.down = 1
            self.distance_to_go = 10
            self.first_down_num += 1
        else:
            self.down += 1

        # Get the reward based on the offense and defense actions
        offense_reward = yards_gained
        defense_reward = -offense_reward
        reward = [offense_reward, defense_reward]

        # Check if the episode has ended
        done = self.down > 4
        if done:
            self.down = 1

        return (self.down, self.distance_to_go), reward, done, {}

    def reset(self):
        self.down = 1
        self.distance_to_go = 10
        self.first_down_num = 0
        return (self.down, self.distance_to_go)

    def render(self, mode='human'):
        if mode == 'human':
            field = ['[ ]'] * 10
            field[self.distance_to_go - 1] = '[*]'
            field_str = ''.join(field)

            print(f"Down: {self.down} | Distance to go: {self.distance_to_go}")
            print(field_str)
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")