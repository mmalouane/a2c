import numpy as np
import gym
from a2c import A2CAgent
import matplotlib.pyplot as plt
from gym import wrappers
from environment import Environment

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    agent = A2CAgent(lr=0.0001, input_dims=[4], gamma=0.99, n_actions=4)

    env = Environment('BreakoutNoFrameskip-v4', [], atari_wrapper=True)
    score_history = []
    num_episodes = 70000

    file_loss = open("loss.csv", "a")
    file_loss.write("episode,reward,loss\n")

    for episode in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            loss = agent.learn(observation, reward, new_observation, done)
            observation = new_observation
            score += reward
        
        file_loss.write("%d,%d,%.6f\n" % (episode, score, loss))
        file_loss.flush()

        if(episode % 50 == 0):
            agent.save("checkpoints/model_%d.pth" % episode)