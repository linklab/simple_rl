from gym import envs
for idx, env_name in enumerate(envs.registry.all()):
    print(idx, env_name)

import gym
print(gym.__version__)