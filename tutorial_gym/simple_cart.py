import gym.spaces

# Choose environment
env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')
#env = gym.make('LunarLander-v2')
#env = gym.make('BipedalWalker-v2')
#env = gym.make('CarRacing-v0')
#env = gym.make('Pong-v0')

print(env.action_space)
print(env.observation_space)

for i_episode in range(5):
    # Reset environment
    state = env.reset()
    for t in range(1000):
        # Display results on screen
        env.render()
        print('State:')
        print(state)
        # Take random action
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        # Episode end
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

# Close environment
env.close()