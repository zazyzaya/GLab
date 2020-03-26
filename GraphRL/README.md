Using the Relative Novelty policy from [Using Relative Novelty to Identify Useful Temporal Abstractions in Reinforcement Learning. Simesk & Barto, 2004](https://dl.acm.org/doi/pdf/10.1145/1015330.1015353) to guide random walks in the Node2Vec algorithm for node classification in graphs.

Here's the output of the tests on the Zachary's Karate Club graph:

    Default N2V:
    Avg accuracy from 100 runs: 0.637059
    Elapsed time: 65.7086250782013


    RL N2V:
    Avg accuracy from 100 runs: 0.700882
    Elapsed time: 56.61166334152222
    
It appears to run faster, and have higher accuracy on small graphs. More research needs to be done on larger graph inputs
