from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN
import tensorflow as tf

print("##### Test tensorflow device allocation #####\n")
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print("##### Train a DQN agent #####\n")
env = make_atari('BreakoutNoFrameskip-v4')

model = DQN(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=100)
model.save("deepq_breakout")

obs = env.reset()

print("##### Test the agent - should have 0 reward because it's bad #####\n")
total_reward = 0

for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, dones, info = env.step(action)
    total_reward += reward

print(f"Total reward {total_reward}")
