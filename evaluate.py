from tqdm import tqdm
import torch
import numpy as np
from common import make_env
import TD3
import pandas as pd
import argparse

default_timesteps = {'InvertedPendulum-v2': 0.02, 'Hopper-v2': 0.002, 'Walker2d-v2': 0.002,
                     'InvertedDoublePendulum-v2': 0.01}
default_frame_skips = {'InvertedPendulum-v2': 2, 'Hopper-v2': 4, 'Walker2d-v2': 4, 'InvertedDoublePendulum-v2': 5}


def get_results(response_rate, parent_response_rate, env_name, threshold,
                with_parent_action, penalty, double_action):
    dataframe = pd.DataFrame(columns=["env", "result", "seed", "actions", "threshold"])
    delayed_env = True
    discount = 0.99
    tau = 0.005
    policy_freq = 2
    policy_noise = 2
    noise_clip = 0.5

    default_timestep = default_timesteps[env_name]
    default_frame_skip = default_frame_skips[env_name]

    for seed in tqdm(range(5)):
        torch.manual_seed(seed)
        np.random.seed(seed)

        timestep = default_timestep if default_timestep <= response_rate else response_rate
        frame_skip = response_rate / timestep
        parent_steps = int(parent_response_rate / response_rate)  # Number children steps in on parent step
        time_change_factor = (default_timestep * default_frame_skip) / (timestep * frame_skip)
        env = make_env(env_name, seed, time_change_factor, timestep, frame_skip, delayed_env)
        state_dim = sum(s.shape[0] for s in env.observation_space)
        action_dim = env.action_space.shape[0]
        parent_max_action = float(env.action_space.high[0])
        child_max_action = 2 * parent_max_action
        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": parent_max_action,
            "discount": discount,
            "tau": tau,
            "observation_space": env.observation_space,
            "delayed_env": delayed_env,
        }
        # Initialize policy
        kwargs["policy_noise"] = policy_noise * parent_max_action
        kwargs["noise_clip"] = noise_clip * parent_max_action
        kwargs["policy_freq"] = policy_freq
        parent_policy = TD3.TD3(**kwargs)
        if with_parent_action:
            kwargs["state_dim"] = state_dim + action_dim
        if double_action:
            kwargs["max_action"] = child_max_action
            kwargs["policy_noise"] = policy_noise * child_max_action
            kwargs["noise_clip"] = noise_clip * child_max_action
        policy = TD3.TD3(**kwargs)
        arguments = [env_name, seed, parent_response_rate, 'best']
        parent_file_name = '_'.join([str(x) for x in arguments])
        parent_policy.load(f"./models/{parent_file_name}")
        augment_type = "fast"
        if penalty:
            augment_type += '_penalty'
        if with_parent_action:
            augment_type += '_with_parent_action'
        if double_action:
            augment_type += '_double_action'

        arguments = [augment_type, env_name, seed, response_rate, parent_response_rate, 'best']
        file_name = '_'.join([str(x) for x in arguments])
        policy.load(f"./models/{file_name}")
        eval_env = make_env(env_name, seed, time_change_factor, timestep, frame_skip, delayed_env)
        rewards = 0
        actions = 0
        for _ in range(10):
            eval_state, eval_done = eval_env.reset(), False
            eval_parent_action = eval_env.previous_action
            eval_episode_timesteps = 0
            eval_next_parent_action = parent_policy.select_action(eval_state)
            eval_child_state = np.concatenate([eval_state, eval_parent_action], 0) if with_parent_action else eval_state
            while not eval_done:

                eval_child_action = policy.select_action(eval_child_state)
                eval_action = (eval_parent_action + eval_child_action).clip(-parent_max_action, parent_max_action)
                if np.all(abs(eval_action - eval_parent_action) <= threshold):
                    eval_action = eval_parent_action
                    eval_child_action = 0
                    if (eval_episode_timesteps + 1) % parent_steps == 0:
                        actions += 1
                else:
                    actions += 1

                eval_next_state, eval_reward, eval_done, _ = eval_env.step(eval_action)
                eval_state = eval_next_state
                eval_episode_timesteps += 1

                if eval_episode_timesteps % parent_steps == 0:
                    eval_next_parent_action = parent_policy.select_action(eval_state)
                elif (eval_episode_timesteps + 1) % parent_steps == 0:
                    eval_parent_action = eval_next_parent_action

                eval_child_state = np.concatenate([eval_state, eval_parent_action],
                                                  0) if with_parent_action else eval_state
                rewards += eval_reward
        avg_reward = rewards / 10
        avg_actions = actions / 10
        dataframe.loc[len(dataframe)] = [env_name, avg_reward, seed, avg_actions, threshold]
    dataframe.to_csv('dataframe.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="InvertedPendulum-v2", help="Environment name")
    parser.add_argument("--response_rate", default=0.04, type=float,
                        help="Response time of the fast network in seconds")
    parser.add_argument("--parent_response_rate", default=0.08, type=float,
                        help="Response time of the slow network in seconds")
    parser.add_argument("--threshold", default=0.3, type=float, help="Threshold for gating the fast action")
    parser.add_argument("--penalty", action="store_true", help="add penalty to reward for action magnitude")
    parser.add_argument("--with_parent_action", action="store_true", help="add parent action to state")
    parser.add_argument("--double_action", action="store_true", help="max double action for policy")

    args = parser.parse_args()
    args = vars(args)
    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    get_results(**args)
