import gym

env = gym.make('CartPole-v1', render_mode='human')

env.reset()
print("State: ",env.observation_space.shape[0])
print("Action: ",env.action_space.n)
input()
while True:
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, info, _ = env.step(action)
    print(next_state)
    if done:
        env.reset()