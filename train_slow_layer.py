import TD3
import utils
import numpy as np
import torch
import argparse
from common import make_env, create_folders

default_timesteps = {'InvertedPendulum-v2': 0.02, 'Hopper-v2': 0.002, 'Walker2d-v2': 0.002,
                     'InvertedDoublePendulum-v2': 0.01}
default_frame_skips = {'InvertedPendulum-v2': 2, 'Hopper-v2': 4, 'Walker2d-v2': 4, 'InvertedDoublePendulum-v2': 5}


# Main function of the policy. Model is trained and evaluated inside.
def train(seed=0, start_timesteps=25e3, eval_freq=5e3, max_timesteps=1e5,
          expl_noise=0.1, batch_size=256, discount=0.99, tau=0.005, policy_freq=2, policy_noise=2, noise_clip=0.5,
          response_rate=0.04, env_name='InvertedPendulum-v2'):
    arguments = [env_name, seed, response_rate]
    file_name = '_'.join([str(x) for x in arguments])

    default_timestep = default_timesteps[env_name]
    default_frame_skip = default_frame_skips[env_name]

    print("---------------------------------------")
    print(f"Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    create_folders()

    timestep = default_timestep if default_timestep <= response_rate else response_rate
    frame_skip = response_rate / timestep

    print('timestep:', timestep)  # How long does it take before two frames
    # How many frames to skip before return the state
    print('frameskip:', frame_skip)

    # The ratio of the default time consumption between two states returned and reset version.
    # Used to reset the max episode number to guarantee the actual episode max time is always the same.
    time_change_factor = (default_timestep * default_frame_skip) / (timestep * frame_skip)
    env = make_env(env_name, seed, time_change_factor, timestep, frame_skip, delayed_env=True)

    print('time change factor', time_change_factor)
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = sum(s.shape[0] for s in env.observation_space)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "tau": tau,
        "observation_space": env.observation_space,
        "delayed_env": True,
    }

    # Initialize policy
    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = policy_noise * max_action
    kwargs["noise_clip"] = noise_clip * max_action
    kwargs["policy_freq"] = policy_freq
    policy = TD3.TD3(**kwargs)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    evaluations = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    max_episode_timestep = env.env.env._max_episode_steps
    best_performance = -100

    for t in range(int(max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < max_episode_timestep else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            policy.train(replay_buffer, batch_size)

        if done:
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % eval_freq == 0:

            eval_env = make_env(env_name, seed, time_change_factor, timestep, frame_skip, delayed_env=True)
            rewards = 0
            for _ in range(10):
                eval_state, eval_done = eval_env.reset(), False
                while not eval_done:
                    eval_action = policy.select_action(np.array(eval_state))
                    eval_next_state, eval_reward, eval_done, _ = eval_env.step(eval_action)
                    eval_state = eval_next_state

                    rewards += eval_reward
            avg_reward = rewards / 10
            evaluations.append(avg_reward)
            print(f" --------------- Evaluation reward {avg_reward:.3f}")
            np.save(f"./results/{file_name}", evaluations)

            if best_performance < avg_reward:
                best_performance = avg_reward
                policy.save(f"./models/{file_name}_best")

    policy.save(f"./models/{file_name}_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="InvertedPendulum-v2", help="Environment name")
    parser.add_argument("--seed", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument("--start_timesteps", default=1000, type=int, help="Time steps initial random policy is used")
    parser.add_argument("--eval_freq", default=5000, type=int, help="How often (time steps) we evaluate")
    parser.add_argument("--max_timesteps", default=1000000, type=int, help="Max time steps to run environment")
    parser.add_argument("--expl_noise", default=0.1, help="Std of Gaussian exploration noise")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for both actor and critic")
    parser.add_argument("--discount", default=0.99, help="Discount factor")
    parser.add_argument("--tau", default=0.005, help="Target network update rate")
    parser.add_argument("--policy_noise", default=0.2, help="Noise added to target policy during critic update")
    parser.add_argument("--noise_clip", default=0.5, help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", default=2, type=int, help="Frequency of delayed policy updates")
    parser.add_argument("--response_rate", default=0.04, type=float, help="Response time of the agent in seconds")

    args = parser.parse_args()
    args = vars(args)
    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    train(**args)
