# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
import gym

# Environment
# Slippery environment (stochastic policy, move left probability = 1/3) comes by default!
# If we want deterministic policy, we need to create new environment
# Make environment No Slippery (deterministic policy, move left = 100% left)
MAX_EPISODE_STEPS = 250

gym.envs.register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={
        'map_name': '4x4', 'is_slippery': False
    },
    max_episode_steps=MAX_EPISODE_STEPS,
)

# You can only register once
# To delete any new environment
# del gym.envs.registry.env_specs['FrozenLakeNotSlippery-v0']

# Make the environment based on deterministic policy
env = gym.make('FrozenLakeNotSlippery-v0')
# env = gym.make('FrozenLake-v0')


def main_env_info():
    #####################
    # observation space #
    #####################
    print("*" * 80)
    print(env.observation_space)
    print(env.observation_space.n)
    # We should expect to see 15 possible grids from 0 to 15 when
    # we uniformly randomly sample from our observation space
    for i in range(10):
        print(env.observation_space.sample(), end=" ")
    print()

    print("*" * 80)
    ################
    # action space #
    ################
    print(env.action_space)
    print(env.action_space.n)
    # We should expect to see 4 actions when
    # we uniformly randomly sample:
    #     1. LEFT: 0
    #     2. DOWN: 1
    #     3. RIGHT: 2
    #     4. UP: 3
    for i in range(10):
        print(env.action_space.sample(), end=" ")
    print()

    print("*" * 80)
    # This sets the initial state at S, our starting point
    # We can render the environment to see where we are on the 4x4 frozenlake gridworld
    observation = env.reset()
    print(observation, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    env.render()

    action = 2  # RIGHT
    observation, reward, done, info = env.step(action)
    env.render()

    # Observation = 1: move to grid number 1 (unchanged)
    # Prob = 1: deterministic policy, if we choose to go right, we'll go right
    print("Observation: {0}, Action: {1}, Reward: {2}, Done: {3}, Info: {4}".format(
        observation, action, reward, done, info
    ))

    action = 1  # DOWN
    observation, reward, done, info = env.step(action)
    env.render()

    # Observation = 5: move to grid number 5 (unchanged)
    print("Observation: {0}, Action: {1}, Reward: {2}, Done: {3}, Info: {4}".format(
        observation, action, reward, done, info
    ))

    print("*" * 80)
    # This sets the initial state at S, our starting point
    # We can render the environment to see where we are on the 4x4 frozenlake gridworld
    env.reset()
    env.render()

    actions = [2, 2, 1, 1, 1, 2]
    for action in actions:
        observation, reward, done, info = env.step(action)
        env.render()
        print("Observation: {0}, Action: {1}, Reward: {2}, Done: {3}, Info: {4}".format(
            observation, action, reward, done, info
        ))


if __name__ == "__main__":
    main_env_info()
