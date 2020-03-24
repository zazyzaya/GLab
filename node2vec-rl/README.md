## Node2Vec-RL 
This project attempts to use RL in the random walk used by Node2Vec

So far, I have only implemented Q-learning which greatly reduces the dimensions required in order to represent each node as a vector

Karate Club with 2D vector reprs using traditional Node2Vec  |  Karate Club with 2D vector reprs. using Q-learning Node2Vec
:-------------------------:|:-------------------------:
|<sub><sup>Circles are true classes</sub></sup>| <sub><sup>X's are predicted</sub></sup>
![Without RL](img/Without_RL.png) | ![With RL](img/With_RL.png)
55% accuracy  | 88% accuracy 

