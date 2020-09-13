# snake
OpenAI Gym Environment for Snake

## Problem Statement

I started this project as a response to the Slitherin' problem under [OpenAI's Request for Research 2.0](https://openai.com/blog/requests-for-research-2/).

## My Response
### [Singleplayer Snake](#single-player-snake)
*  Environment: The snake starts off with a length of two facing a random direction. There only exists one fruit in the environment at a time. Each time the snake consumes the fruit, the snake grows by a length of one and the fruit's next location is randomly generated. The snake dies when it collides with itself or the wall (this is considered "losing" the game). The snake "wins" the game when its body length equals the grid's width * length (i.e. covers all available squares).
*  Agent: I implemented the Monte Carlo Algorithm, MCTS, UCT, and DQN in that order. UCT is the most advanced planning algorithm that I implemented and I compared it with an implementation of DQN that I found from [Machine Learning with Phil](https://www.youtube.com/watch?v=SMZfgeHFFcA). Unfortunately, the DQN was not able to get better than random results even after extensive hyperparameter tuning. This is why I decided to choose UCT for my Multiplayer Snake environment.

### [Multiplayer Snake](#multi-player-snake)
*  Environment: Each snake starts off with a length of one facing a random direction. There only exists one fruit in the environment at a time. Each time a snake consumes the fruit, the snake grows by a length of one and the fruit's next location is randomly generated. Like the problem statement says: "a snake dies when colliding with another snake, itself, or the wall; and the game ends when all snakes die." The last snake to die wins the game.
*  Agent: The problem statement told us to solve the environment using self-play with an RL algorithm of our choice. As stated in the [Singleplayer Snake section](#single-player-snake), UCT was the best-performing agent in the single-player environment so I evaluated how it performed in the multi-player environment.
*  Observed behavior: In a moderately sized environment (~20x20), the snakes are pretty competent in pursuing the food and avoiding other snakes. If we try to increase the size of the environment beyond ~40x40, the snakes aren't really able to find the fruit through planning. I document below some of the behaviors that I found interesting:

    1. If Snake A is sandwiched between Snake B and the wall, Snake B recognizes that only turning once it reaches the wall will trap Snake A. You can see this one the left below with the top two snakes. Note however, that this strategy must be found through planning and agents can miss it, as can be seen on the right below.
    <div class="row" align="center">
      <div class="column">
        <img src="https://github.com/rpachauri/snake/blob/master/imgs/block_till_edge.gif" width="250">
      </div>
      <div class="column">
        <img src="https://github.com/rpachauri/snake/blob/master/imgs/missed_block.gif" width="250">
      </div>
    </div>
    
    2. Snakes are able to take advantage of the confined space to trap other snakes; however, this is not a regular occurrence, so I'm not sure if I would qualify this as "ganging up" on each other. You can see this below where three snakes are killed consecutively because other agents are able to trap them.
    <div class="row" align="center">
      <div class="column">
        <img src="https://github.com/rpachauri/snake/blob/master/imgs/triple_kill.gif" width="250">
      </div>
    </div>
    
    3. We defined "winning" for an agent to be "the last snake to die." If a snake was the last snake alive, it would try to find the fastest way to die because anything it did would be considered a win. In the future, a different definition of winning might be to be the longest snake, which could encourage more aggressive tactics.
    <div class="row" align="center">
      <div class="column">
        <img src="https://github.com/rpachauri/snake/blob/master/imgs/suicide.png" width="250">
      </div>
    </div>

## Installation

I created this project using the anaconda distribution. You can install it [here](https://docs.anaconda.com/anaconda/install/). If you prefer a lightweight version, you can [install Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) instead.


Create the conda environment with:

    $ conda env create -f snake.yml

Activate the conda environment with:

    $ conda activate snake

Install the Snake environment with (note that the term "environment" is overloaded and we mean the RL environment in this case):

    $ pip install -e .

Go into the agents directory and run one of the agents:

    $ cd agents
    $ python monte_carlo.py

## References
1. [2048 - Solving 2048 with AI 🤖](https://towardsdatascience.com/2048-solving-2048-with-monte-carlo-tree-search-ai-2dbe76894bab)
    1. This article showed me how to implement MCTS. I followed its logic, but the code in this repo was written by me.
2. [A Deep Dive into Monte Carlo Tree Search](https://www.moderndescartes.com/essays/deep_dive_mcts/)
    1. This blog post was incredibly helpful as it took MCTS a step further and showed the implementation for UCT. I did my best to write the code in my own style, but there will likely be similarities.
3. [AI FOR A MULTIPLAYER SNAKE GAME](https://sds-dubois.github.io/2017/01/03/Multiplayer-Snake-AI.html)
    1. I found this blog post upon finishing my project. They do a very thorough analysis of the problem, comparing Minimax agents with RL agents (finding that Minimax agents tended to do slightly better). I implemented a Multi-Agent version of UCT and I'm interested to know how that would compare against their Minimax agent.
