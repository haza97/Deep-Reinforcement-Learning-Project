import numpy as np
import os
import matplotlib.pyplot as plt
filepath = r"C:\Users\harol\Desktop\DRL\roundabout_output"

crash_count = np.load(os.path.join(filepath, "crash count.npy"))
max_reward = np.load(os.path.join(filepath, "max reward.npy"))
mean_reward = np.load(os.path.join(filepath, "mean reward.npy"))
min_reward = np.load(os.path.join(filepath, "min reward.npy"))
mean_speed = np.load(os.path.join(filepath, "mean speed.npy"))
name_list = np.load(os.path.join(filepath, "name list.npy"))

filepath = r"C:\Users\harol\Desktop\DRL\graphs"

plt.figure()
plt.hist(crash_count, density=False, facecolor='g', alpha=0.75)
plt.title("Number of Crashes per episode")
plt.ylabel("Count")
plt.xlabel("Number of Crashes")
plt.savefig(os.path.join(filepath, "crashcount"))

plt.figure()
plt.title("Maximum reward per episode")
plt.hist(max_reward, density=False, facecolor='g', alpha=0.75)
plt.ylabel("Count")
plt.xlabel("Reward")
plt.savefig(os.path.join(filepath, "maxreward"))

plt.figure()
plt.title("Mean Reward per episode")
plt.hist(mean_reward, density=False, facecolor='g', alpha=0.75)
plt.savefig(os.path.join(filepath, "meanreward"))

plt.figure()
plt.title("Minimum Reward per episode")
plt.hist(min_reward, density=False, facecolor='g', alpha=0.75)
plt.xlabel("Reward")
plt.ylabel("Count")
plt.savefig(os.path.join(filepath, "minreward"))

plt.figure()
plt.title("Mean Speed per episode")
plt.hist(mean_speed, density=False, facecolor='g', alpha=0.75)
plt.ylabel("Count")
plt.xlabel("Speed")
plt.savefig(os.path.join(filepath, "meanspeed"))
