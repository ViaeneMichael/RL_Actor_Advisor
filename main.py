import gym
import numpy as np

from wrappers import *
from agent import Agent
from agent import PPOTrainer
from gym import envs

EPISODES = 100000
EPSILON = 0.1
LEARNING_RATE = 0.001
GAMMA = 0.9
"""
(screen_width, screen_height) = self.ale.getScreenDims() # width: 160, height: 210
"""



def main():
    # Seaquest environment
    env_id = "Seaquest-v0"
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    """
    # 0 - standing still
            # 1 - standing still & fire
            # 2 - move up
            # 3 - move right
            # 4 - move left
            # 5 - move down
            # 6 - move right up
            # 7 - move left up
            # 8 - move right down
            # 9 - move left down
            #10 - move up & fire
            #11 - move right & fire
            #12 - move left & fire
            #13 - move down & fire
            #14 - move right up & fire
            #15 - move left up & fire
            #16 - move right down & fire
            #17 - move left down & fire
    """
    output_size = env.action_space.n # 18 actions
    # Will need to resize the image -> computing power and square images (padding vs resizing best practice)
    #  print(env.ale.getScreenDims())
    # Resize the image of the Atari game to a 86*86 image with grayscale instead of RGB
    # and add a stack of frames of 3 frames (in order to see the way that objects are moving
    agent = Agent(env, EPSILON, GAMMA, 0.0001, PPOTrainer, 4)
    avg_cum_reward = 0
    for episode in range(100000):
        score = agent.do_episode()
        avg_cum_reward += score
        print("Avg cum reward of episode: " + str(episode) + "....." + str(avg_cum_reward / (episode + 1)))


main()