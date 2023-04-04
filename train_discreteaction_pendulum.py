from DQN import DQN
import discreteaction_pendulum


def main():
    env = discreteaction_pendulum.Pendulum()
    agent = DQN(env=env, hid_dim=64)
    agent.learn(episode_length=env.max_num_steps, num_episodes=1000, replay=True, target=False)
    agent.plot_return(name="trained_learning_curve_4")
    agent.generate_gif(env=env, name="trained_policy_4")
    agent.plot_trajectory(name='trained_trajectory_4')
    agent.plot_fcn(pol=True, name='Policy_4')
    agent.plot_fcn(pol=False, name='Value_4')


if __name__ == '__main__':
    main()
