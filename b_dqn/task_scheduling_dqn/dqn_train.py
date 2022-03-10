import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import wandb

from b_dqn.common import ReplayBuffer, Transition
from b_dqn.task_scheduling_dqn.task_scheduling_env import EnvironmentTaskScheduling, DoneReasonType

# DQN Parameters
GAMMA = 0.999
BUFFER_LIMIT = 10_000
BATCH_SIZE = 128
MIN_BUFFER_SIZE_FOR_TRAIN = BATCH_SIZE * 10
Q_NET_SYNC_INTERVAL = 200
LEARNING_RATE = 0.001

# Epsilon Decaying Parameters
EPSILON_START = 0.9
EPSILON_END = 0.005
EPSILON_LAST_EPISODES_RATIO = 0.75

NUM_TASKS = 10
INITIAL_RESOURCES_CAPACITY = [70, 80]  # task resource limits
LOW_DEMAND_RESOURCE_AT_TASK = [1, 1]
HIGH_DEMAND_RESOURCE_AT_TASK = [20, 20]
INITIAL_TASK_DISTRIBUTION_FIXED = True

# General Parameters
PRINT_INTERVAL = 20
TEST_INTERVAL_EPISODE = 200

VERBOSE = False
USE_WANDB = True

NUM_EPISODES = 5_000


if USE_WANDB:
    wandb_obj = wandb.init(
        entity="link-koreatech", project="DQN_TASK_SCHEDULING"
    )


class QNet(nn.Module):
    def __init__(self, num_input, num_output):
        super(QNet, self).__init__()
        self.num_output = num_output
        self.fc1 = nn.Linear(num_input, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x

    def get_action(self, obs, epsilon=0.1):
        # random.random(): 0.0과 1.0사이의 임의의 값을 반환
        if random.random() < epsilon:
            action = random.randrange(0, self.num_output)
        else:
            out = self.forward(obs)
            action = torch.argmax(out, dim=-1)
            action = action.item()
        return action  # argmax: 가장 큰 값에 대응되는 인덱스 반환


class TASK_SCHEDULING_DQN:
    def __init__(self, train_env, test_env, q_net, q_net_target):
        self.train_env = train_env
        self.test_env = test_env
        self.q_net = q_net
        self.q_net_target = q_net_target

        self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.memory = ReplayBuffer(BUFFER_LIMIT)

        self.optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)

        self.last_loss_value = 0.0

        self.train_score_list = []
        self.train_avg_score_list = []

        self.test_avg_score_list = []
        self.test_std_score_list = []

        self.test_avg_resource_allocation_list = []
        self.test_episode_list = []

        test_avg_score, test_std_score, test_avg_resource_allocated = self.test_main()
        self.test_avg_score_list.append(test_avg_score)
        self.test_std_score_list.append(test_std_score)
        self.test_avg_resource_allocation_list.append(test_avg_resource_allocated)
        self.test_episode_list.append(0)

    def epsilon_scheduled(self, current_episode):
        epsilon_scheduled_last_episode = NUM_EPISODES * EPSILON_LAST_EPISODES_RATIO
        fraction = min(current_episode / epsilon_scheduled_last_episode, 1.0)
        epsilon = min(
            EPSILON_START + fraction * (EPSILON_END - EPSILON_START), EPSILON_START
        )
        return epsilon

    def train_loop(self):
        test_avg_score = 0.0
        test_std_score = 0.0
        test_avg_resource_allocated = 0.0
        last_loss_value = 0.0

        total_step = 0

        for n_episode in range(NUM_EPISODES):
            epsilon = self.epsilon_scheduled(n_episode)

            observation = self.train_env.reset()  # Initialize task_scheduling state

            done = False
            info = None
            episode_step = 0
            score = 0.0
            if VERBOSE:
                print("\n========================================================================================")

            while not done:
                if VERBOSE:
                    print("[Episode: {0}/{1}, Step: {2}] observation: {3} ".format(
                        n_episode, NUM_EPISODES, episode_step, observation
                    ), end="")

                action = q_net.get_action(torch.from_numpy(observation).float(), epsilon)

                new_observation, reward, done, info = train_env.step(action)

                if VERBOSE:
                    action_str = "{0}".format(action)

                    print("action: {0}, next_observation:\n{1}, reward: {2}, done: {3}".format(
                        action_str, info["INTERNAL_STATE"], reward, done
                    ))

                transition = Transition(observation, action, new_observation, reward, done)

                self.memory.append(transition)
                episode_step = episode_step + 1
                total_step = total_step + 1
                score += reward

                observation = new_observation

                # TRAIN SHOULD BE DONE EVERY STEP
                if self.memory.size() > MIN_BUFFER_SIZE_FOR_TRAIN:
                    last_loss_value = self.train_step()

                if total_step % Q_NET_SYNC_INTERVAL == 0 and total_step != 0:
                    q_net_target.load_state_dict(q_net.state_dict())

            self.train_score_list.append(score)
            self.train_avg_score_list.append(np.mean(self.train_score_list[-100:]))

            if n_episode % TEST_INTERVAL_EPISODE == 0 and n_episode != 0:
                test_avg_score, test_std_score, test_avg_resource_allocated = self.test_main()
                self.test_avg_score_list.append(test_avg_score)
                self.test_std_score_list.append(test_std_score)
                self.test_avg_resource_allocation_list.append(test_avg_resource_allocated)
                self.test_episode_list.append(n_episode)

            done_info = self.get_done_info(info=info)
            resource_allocation_info = self.get_resource_allocation_info(info=info)

            if USE_WANDB:
                wandb_obj.log({
                    "[TEST] Average Episode Reward": test_avg_score,
                    "[TEST] Std. Episode Reward": test_std_score,
                    "[TEST] Resource Utilization": test_avg_resource_allocated,
                    "Average Episode Reward": score,
                    "epsilon": epsilon,
                    "Resource Utilization": info["RESOURCE_ALLOCATED"],
                    "Loss": last_loss_value,
                    "episode": n_episode
                })

            if n_episode % PRINT_INTERVAL == 0 and n_episode != 0:
                print("Epi.: {0:5,}/{1:,}, Score: {2:5.2f}, Mean Score: {3:5.2f}, Total Step: {4:6,}, "
                      "Episode Step: {5:2}, Epsilon: {6:4.1f}%, Last Loss: {7:5.3f}, {8} {9}".format(
                    n_episode, NUM_EPISODES, score, self.train_avg_score_list[-1], total_step,
                    episode_step, epsilon * 100, last_loss_value,
                    done_info, resource_allocation_info
                ))

            if VERBOSE:
                print("[Epi.: {0:5,}/{1:,}, Step: {2:,}] Episode Score {3}, {4} {5}".format(
                    n_episode, NUM_EPISODES, episode_step, score, done_info, resource_allocation_info
                ))
                print("========================================================================================")

        if INITIAL_TASK_DISTRIBUTION_FIXED:
            test_avg_score, test_std_score, test_avg_resource_allocated = self.test_main()
        else:
            test_avg_score, test_std_score, test_avg_resource_allocated = self.test_main()

    def train_step(self):
        observations, actions, next_observations, rewards, dones = self.memory.sample(BATCH_SIZE)

        # state_action_values.shape: torch.Size([32, 1])
        q_out = self.q_net(observations)
        q_values = q_out.gather(dim=1, index=actions)

        with torch.no_grad():
            q_prime_out = self.q_net_target(next_observations)
            # next_state_values.shape: torch.Size([32, 1])
            max_q_prime = q_prime_out.max(dim=1, keepdim=True).values
            max_q_prime[dones] = 0.0
            max_q_prime = max_q_prime.detach()

            # target_state_action_values.shape: torch.Size([32, 1])
            targets = rewards + GAMMA * max_q_prime

        # loss is just scalar torch value
        loss = F.mse_loss(targets, q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_done_info(self, info):
        if info["DoneReasonType"] == DoneReasonType.TYPE_3:
            done_info = "Done: {0}".format(info["DoneReasonType"].name + " [BEST!]")
        elif info["DoneReasonType"] == DoneReasonType.TYPE_4:
            done_info = "Done: {0}".format(info["DoneReasonType"].name + " [GOOD!]")
        elif info["DoneReasonType"] == DoneReasonType.TYPE_5:
            done_info = "Done: {0}".format(info["DoneReasonType"].name + " [ ALL!]")
        else:
            done_info = "Done: {0}".format(info["DoneReasonType"].name + " -------")

        return done_info

    def get_resource_allocation_info(self, info, tasks_selected_per_server=None):
        resource_allocation_info = "Alloc.: {0:3}/{1:3} ({2:3}),  Limit: {3:3}/{4:3} ({5:3}), Util.: {6:5.2f}%/{7:5.2f}%, Rasks: {8}".format(
            info["CPU_RESOURCE_ALLOCATED"],
            info["RAM_RESOURCE_ALLOCATED"],
            info["CPU_RESOURCE_ALLOCATED"] + info["RAM_RESOURCE_ALLOCATED"],
            info["CPU_LIMIT"],
            info["RAM_LIMIT"],
            info["CPU_LIMIT"] + info["RAM_LIMIT"],
            100 * (info["CPU_RESOURCE_ALLOCATED"] / info["CPU_LIMIT"]),
            100 * (info["RAM_RESOURCE_ALLOCATED"] / info["RAM_LIMIT"]),
            info["ACTIONS_SELECTED"]
        )

        return resource_allocation_info

    def test_main(self):
        score = 0.0
        NUM_EPISODES = 3

        test_score_list = []
        test_resource_allocation_list = []

        for n_episode in range(NUM_EPISODES):
            observation = self.test_env.reset()
            done = False
            info = None

            episode_step = 0
            while not done:
                action = self.q_net.get_action(torch.from_numpy(observation).float(), epsilon=0.0) # Only Greedy Action Selection
                new_observation, reward, done, info = test_env.step(action)
                episode_step = episode_step + 1
                score += reward
                observation = new_observation

            test_score_list.append(score)
            test_resource_allocation_list.append(info["RESOURCE_ALLOCATED"])

        return np.average(test_score_list), np.std(test_score_list), np.average(test_resource_allocation_list)


if __name__ == '__main__':
    q_net = QNet(num_input=(NUM_TASKS + 1) * 3, num_output=NUM_TASKS)
    q_net_target = QNet(num_input=(NUM_TASKS + 1) * 3, num_output=NUM_TASKS)

    train_env = EnvironmentTaskScheduling(
        INITIAL_TASK_DISTRIBUTION_FIXED,
        NUM_TASKS,
        LOW_DEMAND_RESOURCE_AT_TASK,
        HIGH_DEMAND_RESOURCE_AT_TASK,
        INITIAL_RESOURCES_CAPACITY
    )
    if INITIAL_TASK_DISTRIBUTION_FIXED:
        test_env = train_env
    else:
        test_env = EnvironmentTaskScheduling(
            INITIAL_TASK_DISTRIBUTION_FIXED,
            NUM_TASKS,
            LOW_DEMAND_RESOURCE_AT_TASK,
            HIGH_DEMAND_RESOURCE_AT_TASK,
            INITIAL_RESOURCES_CAPACITY
        )

    task_scheduling_dqn = TASK_SCHEDULING_DQN(
        train_env=train_env,
        test_env=test_env,
        q_net=q_net,
        q_net_target=q_net_target
    )

    task_scheduling_dqn.train_loop()

    if INITIAL_TASK_DISTRIBUTION_FIXED:
        print("Initial Internal State\n", train_env.fixed_initial_internal_state)
        print("min_task_CPU_demand:", train_env.min_task_cpu_demand)
        print("min_task_RAM_demand:", train_env.min_task_ram_demand)
        print("Final Internal State\n", train_env.internal_state)
        print("###########################################################")
