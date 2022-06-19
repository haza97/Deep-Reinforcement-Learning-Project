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


sorted_indices_speed = np.argsort(mean_speed)
sorted_indices_mean = np.argsort(mean_reward)
sorted_indices_min = np.argsort(min_reward)
sorted_indices_crash = np.argsort(crash_count)
#print(sorted_indices_speed)
#print(sorted_indices_mean)
print(sorted_indices_min)
print(sorted_indices_crash)

