import numpy as np
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=DeprecationWarning)
	from stable_baselines3 import PPO
	from stable_baselines3.common.utils import set_random_seed
	from stable_baselines3.common.env_util import make_vec_env
	from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
	import gym
	import stable_baselines3
	import sys
	import highway_env 

env_id = 'roundabout-v0'


train_env = make_vec_env(env_id, n_envs=10, vec_env_cls=SubprocVecEnv,
                         vec_env_kwargs=dict(start_method='fork'))

collision_rewards = [-0.8]
high_speed_rewards = [0, 0.25, 0.5, 0.75, 1]
lane_change_rewards =  [0.2, 0.4]

for c_reward in collision_rewards:
	for h_reward in high_speed_rewards:
		for l_reward in lane_change_rewards:
			config_list = [{'action': {'target_speeds': [0, 8, 16], 'type': 'DiscreteMetaAction'},
			'centering_position': [0.5, 0.6],
                        'collision_reward': c_reward,
                        'duration': 11,
                        'high_speed_reward': h_reward,
                        'incoming_vehicle_destination': None,
                        'lane_change_reward': l_reward,
                        'manual_control': False,
                        'observation': {'absolute': True,
                        'features_range': {'vx': [-15, 15],
                        'vy': [-15, 15],
                        'x': [-100, 100],
                        'y': [-100, 100]},
                        'type': 'Kinematics'},
                        'offscreen_rendering': False,
                        'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
                        'policy_frequency': 1,
                        'real_time_rendering': False,
                        'render_agent': True,
                        'right_lane_reward': 0,
                        'scaling': 5.5,
                        'screen_height': 600,
                        'screen_width': 600,
                        'show_trajectories': False,
                        'simulation_frequency': 15}]

			train_env.set_attr("config", config_list)

            		# Model training and saving
			model_name = f"{c_reward}{h_reward}{l_reward}"
			model_filepath = f"/home/ubuntu/roundabout_models/{model_name}"
			model = stable_baselines3.PPO("MlpPolicy", train_env, verbose=1)
			train_env.reset()
			print("Training Start!")
			model.learn(total_timesteps=50000)
			model.save(model_filepath)

           		# model evaluation
			n_episodes = 100
			counter = 0
			crashcount = 0

			env = gym.make("roundabout-v0")
			while counter < n_episodes:
				obs = env.reset()
				done = False
				while not done:
					action, states = model.predict(obs)
					obs, reward, done, info = env.step(action)
					if env.vehicle.crashed:
						crashcount += 1
				env.close()
				counter += 1

			crashrate = np.array(crashcount/n_episodes)
			outfile = '/home/ubuntu/roundabout_models/'
			np.save(outfile, crashrate)
