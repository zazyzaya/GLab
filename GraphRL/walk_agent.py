import networkx as nx
import random
import queue

class WalkAgent:
    def __init__(self, g, num_walks=128, walk_length=8, novelty_lag=4, 
                learning_rate=0.1):
        '''
        Parameters:
            g:              An nx graph 
            encode_state:   a function f(node) -> V in R^d where d is max node cardinality
        '''
        
        self.g = g
        self.num_walks = num_walks
        self.walk_len = walk_length
        self.d = sorted([d for n,d in g.degree()], reverse=True)[0]
        self.novelty_dict = {}
        self.learning_rate = learning_rate
        
        self.nl = novelty_lag
        self.rn_queue = queue.Queue(maxsize=int(novelty_lag/2))
        self.rn_dict = {}
        self.rn_buffer = NoveltyQueue(novelty_lag, self.novelty_dict)
        self.cur_state = None
        self.default_rn = 2
    
    def clear_rn_queue(self):
        while not self.rn_queue.empty():
            self.rn_queue.get()
        
        self.rn_buffer.clear()

    def reset(self):
        '''
        Empties all settings. Should be called between random walks
        (not between episodes, however. Between switching which node we 
        generate walks from)
        '''
        self.novelty_dict.clear()
        self.rn_dict.clear()
        self.rn_buffer.clear()
        self.clear_rn_queue()
        

    def policy(self, state):
        ''' 
        Current policy: pick next state based on 
            Highest relative novelty score (if known)
        '''
        neighbors = list(self.g[state])

        # Uses RN score as pdf for random walk
        # Improves as it progresses
        return random.choices(
            neighbors, [self.rn_dict.get(k, self.default_rn) for k in neighbors]
        )[0]


    def state_transition(self, action, state=None, change_state=True):
        ''' 
        Goes to next state, and updates novelty/relative novelty scores
        as needed

        Assumes action is the neighbor's index we are going to 
        ''' 
        if state == None:
            state = self.cur_state

        if state in self.novelty_dict:
            self.novelty_dict[state] += 1
        else:
            self.novelty_dict[state] = 1

        # Remove value from q and assign it relative novelty
        if self.rn_queue.full():
            rns = self.rn_queue.get()
            rn = self.rn_buffer.get_relative_novelty()
            
            # Have to wait twice as long to get relative novelty
            # When we have it, update based on gradient
            if rn:
                expected = self.rn_dict.get(rns, self.default_rn)
                self.rn_dict[rns] = expected + self.learning_rate*(rn - expected)

        # Add this state to the respective queues for processing later
        self.rn_queue.put(state)
        self.rn_buffer.add(state)

        # Finally, advance to new state
        if change_state:
            self.cur_state = action

        return action 


    def generate_random_walks(self, start_node):
        '''
        Generates random walks for a given start node
        Resets so relative novelty continues to be variable
        '''

        walks = []
        for _ in range(self.num_walks):
            walk = [start_node]

            for __ in range(self.walk_len):
                s = walk[-1]
                walk.append(
                    self.state_transition(
                        self.policy(s), state=s, change_state=False
                    )
                )

            walks.append([str(w) for w in walk])
            self.clear_rn_queue()
        
        self.reset()
        return walks

    
    ''' I don't think we need to use NNs if we use Relative Novelty. I want
        to try that approach first
    '''
    def supervised_policy_learning(self):
        '''
        Teach the agent to successfully find neighbors (not necessarilly 
        to find the best neighbors. Just to be able to assign actions based
        on state vectors)
        ''' 

        # Note, this part we train using TDL as the only reward is 
        # immidiate, so E[pi(s_t+1) | pi(s_t)] = E[pi(s_t+1)]
        pass


class NoveltyQueue():
    def __init__(self, length, novelty_map, default_n=1):
        self.nl = length 
        self.ptr = 0
        self.isfull = False
        self.queue = [None] * self.nl
        self.novelty_map = novelty_map
        self.default_n = default_n

    def clear(self):
        self.ptr = 0
        self.isfull = False

    def avg(self, arr):
        return sum(
            [self.novelty_map.get(val, self.default_n)**(-0.5) for val in arr] 
        )/len(arr)

    def add(self, item):
        # Keep filling from left to right until buffer full
        if not self.isfull:
            self.queue[self.ptr] = item
            self.ptr += 1
            
            if self.ptr == self.nl:
                self.isfull = True
                self.ptr = 0

            return 

        # Othewise, boot whatever is at ptr and advance it
        self.queue[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.nl
    

    def get_relative_novelty(self):
        if not self.isfull:
            return False

        old = []
        new = []

        # There may be a faster way of doing this, but this works for now
        isOld = lambda x : self.ptr <= x and x < self.ptr+(int(self.nl/2))

        for i in range(self.nl):
            if isOld(i) or isOld(i+self.nl):
                old.append(self.queue[i])
            else:
                new.append(self.queue[i])

        return self.avg(new)/self.avg(old)