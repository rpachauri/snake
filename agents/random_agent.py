# Gotta import gym!
import gym
import snake
import random

# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('snake-v0')

# Reset the environment to default beginning
# Default observation variable
print("Initial Observation")
observation = env.reset()
print(observation)

# Using _ as temp placeholder variable
for i in range(1000):
    print(i)
    # Render the env
    env.render()

    # Still a lot more explanation to come for this line!
    obs, reward, done, info = env.step(random.choice(env.action_space)) # take a random action
    if done:
        break