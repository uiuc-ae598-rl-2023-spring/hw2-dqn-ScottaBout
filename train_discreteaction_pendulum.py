from DQN import DQN
import discreteaction_pendulum


def main():
    env = discreteaction_pendulum.Pendulum()
    model = DQN(in_dim=env.num_states, hid_dim=64, out_dim=env.num_actions)
    model.learn(env=env, episode_length=env.max_num_steps, num_episodes=500)
    model.plot()
    model.generate_gif(env=env)


if __name__ == '__main__':
    main()
