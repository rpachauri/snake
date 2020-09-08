from gym.envs.registration import register

register(
    id='snake-v0',
    entry_point='snake.envs:SnakeEnv',
)

register(
    id='snake-v1',
    entry_point='snake.envs:SnakeEnvV1',
)


register(
    id='multiplayer-snake-v0',
    entry_point='snake.envs:MultiplayerSnakeEnv',
)