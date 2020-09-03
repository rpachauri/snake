# snake
OpenAI Gym Environment for Snake

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
    1. This article was the inspiration for this project. I followed its logic, but the code in this repo was written by me.
2. [A Deep Dive into Monte Carlo Tree Search](https://www.moderndescartes.com/essays/deep_dive_mcts/)
    1. This blog post was incredibly helpful. I did my best to write the code in my own style, but there will likely be similarities.