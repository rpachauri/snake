# agents

## Random Agent

This agent just picks an action at random.

Run with:

    $ python random_agent.py

## Hamiltonian Cycle

This agent follows a hardcoded Hamiltonian Cycle so that it will never crash into itself. Unfortunately, it's not very efficient so it will take a long time to finish, but it is guaranteed to win the game at some point.

Run with:

    $ python hamiltonian_cycle_agent.py

This is not an actual Reinforcement Learning algorithm but it takes advantage of the structure of the Snake game to execute a policy that will always win (I think a constraint on this specific implementation is that the grid must have an even number of rows and columns).

## Monte Carlo

In the Monte Carlo algorithm, the agent performs a number of "rollouts" at each state. This is the equivalent of a human thinking through possible scenarios at each state in a game (note that each rollout plays until the end of the game). It uses these rollouts to *estimate* the "action-value" (i.e. the expected total reward after selecting an action) for each possible action. It then selects the action with the highest estimated action-value. Since the action-values are *estimates*, the agent is not guaranteed to select the best action.

Run with:

    $ python monte_carlo.py

## Monte Carlo Tree Search

Note that in the Monte Carlo algorithm, we collect rollouts for each action at a particular state, but throw away those rollouts once we move to the next state. This is wasteful because some of those rollouts were for the action we ended up choosing. Monte Carlo Tree Search solves for this by organizing the rollouts into a tree with each node's children being the state that would result from a possible action. Whenever we select an action, we move to that child node.

Run with:

    $ python mcts.py

## Upper Confidence Bounds for Trees

Note that in Monte Carlo Tree Search, each action during each rollout is selected at random. In reinforcement learning, there is something known as the exploitation-exploration tradeoff: do we continue to exploit the action we know does well or explore to see if there is a better action?
    *   An exploitative policy would be something like trying each action once and then only selecting the action that has the highest reward. The problem with something like this is that we may miss out on action has a higher value.
    *   An explorative policy would be something like Breadth-First Search. There just isn't enough time to exhaustively try all options.

The Upper Confidence Bound algorithm is a neat way to balance this tradeoff. For a more thorough understanding, check out [A Deep Dive into Monte Carlo Tree Search](https://www.moderndescartes.com/essays/deep_dive_mcts/)!

UCT applies Upper Confidence Bounds for action selection at each state during each rollout. This means that while we're doing rollouts, we balance between exploiting actions that we know do well and exploring for actions that could do better.

Run with:

    $ python uct.py

Note that depending on your environment, you'll have to change EXPLORATION_CONSTANT based on the reward function.

# Summary

This was a fun project to understand RL algorithms. Starting with Monte Carlo, each algorithm builds upon the previous one. The agents may not be optimal but hopefully you should be able to see some sort of progression in ability (each agent should seem smarter than the one before it).