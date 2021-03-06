from wrappers import *
from agent import Agent
from agent import PPOTrainer

EPISODES = 100000
LEARNING_RATE = 0.0001
GAMMA = 0.9
LAMBDA = 0.95
C1 = 0.5
MEM_SIZE = 12

"""
(screen_width, screen_height) = self.ale.getScreenDims() # width: 160, height: 210
"""
def make_mean_ofrewards():
    output = np.array([])
    for i in range(16, 18):
        averages = [0]*4000
        for j in range(10):
            f = open("rewards" + str(i) + "_" + str(j) + ".txt", "r")
            line = 0
            for reward in f:
                print(line)
                averages[line]+=float(reward)
                line += 1
        outfile = open("AverageRewardsPPO"+str(i)+".txt","a")
        for av in range(2000):
            outfile.write(str(averages[av]/10.0)+"\n")
    print("done")



def hyperparameters():
    mem_sizes = [12, 24, 48]
    learning_rates = [0.001, 0.0001, 0.00001]
    lambdas = [0.9, 0.95, 0.99]
    gammas = [0.9, 0.95, 0.99]
    c1s = [0.5, 1]
    for i in range(16, len(lambdas) * len(gammas) * len(c1s)):
        lambd = lambdas[i % 3]
        gamma = gammas[(i // 3) % 3]
        c1 = c1s[i // 9]
        for j in range(10):
            # Seaquest environment
            env_id = "Seaquest-v0"
            env = make_atari(env_id)
            env = wrap_deepmind(env)
            env = wrap_pytorch(env)
            agent = Agent(env, learning_rates[1], mem_sizes[0], gamma, lambd, c1, PPOTrainer)
            avg_cum_reward = 0
            f = open("rewards" + str(i) + "_" + str(j) + ".txt", "a")
            for episode in range(2000):
                score = agent.do_episode()
                avg_cum_reward += score
                f.write(str(avg_cum_reward / (episode + 1)) + "\n")
                print("Avg cum reward of episode: " + str(episode) + "....." + str(avg_cum_reward / (episode + 1)))
            f.close()

def learn_seaquest():
    for i in range(5):
        env_id = "Seaquest-v0"
        env = make_atari(env_id)
        env = wrap_deepmind(env)
        env = wrap_pytorch(env)
        agent = Agent(env, LEARNING_RATE, MEM_SIZE, GAMMA, LAMBDA, C1, PPOTrainer)
        avg_cum_reward = 0
        f = open("ActualAdvisorrewards" + str(i) + ".txt", "a")
        for episode in range(20000):
            score = agent.do_episode()
            avg_cum_reward += score
            f.write(str(avg_cum_reward / (episode + 1)) + "\n")
            print("Avg cum reward of episode: " + str(episode) + "....." + str(avg_cum_reward / (episode + 1)))
        f.close()


def main():

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

    used as reference: https://github.com/safe-rl/safe-rl-shielding/blob/master/envs/seaquest/dqn/seaquest_shielded_agent.py
    """
    # output_size = env.action_space.n # 18 actions
    # Will need to resize the image -> computing power and square images (padding vs resizing best practice)
    #  print(env.ale.getScreenDims())
    # Resize the image of the Atari game to a 86*86 image with grayscale instead of RGB
    # and add a stack of frames of 3 frames (in order to see the way that objects are moving

    learn_seaquest()


if __name__ == "__main__":
    main()