# snake
OpenAI Gym Environment for Snake

## Problem Statement

I started this project as a response to the Slitherin' problem under [OpenAI's Request for Research 2.0](https://openai.com/blog/requests-for-research-2/).

### My Response


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