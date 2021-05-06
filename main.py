import gym
from neuralnet import SeaquestNet

# Neural net


# Agent
    # Build networks
    # PPO LEARNING
        # ..
    # Save networks

#Advisor


EPISODES = 100000
EPSILON = 0.1
GAMMA = 0.9
LEARNING_RATE = 0.001
"""
(screen_width, screen_height) = self.ale.getScreenDims()
"""
def main():
    # Seaquest environment
    env = gym.make('Seaquest-v0')
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
    print(env.ale.getScreenDims()) # width: 160, height: 210
    # env.getimage.resize(84,84)


    # For loop episodes
        # loop steps episode


    # Save scores

main()