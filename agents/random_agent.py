# Gotta import gym!
import gym
import snake

# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make('snake-v0')

# Reset the environment to default beginning
# Default observation variable
print("Initial Observation")
observation = env.reset()
print(observation)
