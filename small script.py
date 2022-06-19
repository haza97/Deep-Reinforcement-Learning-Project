from stable_baselines3 import PPO
import gym
import stable_baselines3
import os
import 

filepath = '/home/ubuntu/roundabout_models_zipped/-0.30.250.2.zip'
files = os.listdir(filepath)

env = gym.make('roundabout-v0')

model = PPO.load(filepath)
