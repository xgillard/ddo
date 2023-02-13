import ddo
import os

class KnapsackState:
    '''
    A state of your dynamic programme. 
    I know it is a bit annoying, but this class *MUST* implement the methods
    __eq__ and __hash__ at the very least
    '''
    def __init__(self, depth, capa):
        self.depth = depth
        self.capa  = capa
    
    def __eq__(self, other):
        if not isinstance(other, KnapsackState):
            return False
        else:
            return self.depth == other.depth and self.capa == other.capa
    
    def __hash__(self):
        return (self.depth * 47) + self.capa

class Knapsack:
    '''
    The problem definition
    '''
    def __init__(self, capa, profit, weight):
        self.capa   = capa
        self.profit = profit
        self.weight = weight
    
    def nb_variables(self):
        return len(self.profit)
    
    def initial_state(self):
        return KnapsackState(0, self.capa)
    
    def initial_value(self):
        return 0
    
    def transition(self, state, variable, value):
        if value == 1:
            return KnapsackState(state.depth + 1, state.capa - self.weight[variable])
        else:
            return KnapsackState(state.depth + 1, state.capa)
    
    def transition_cost(self, state, variable, value):
        return self.profit[variable] * value
    
    def domain(self, variable, state):
        if state.capa >= self.weight[variable]:
            return [1, 0]
        else:
            return [0]
    
    def next_variable(self, depth, next_layer):
        if len(next_layer) == 0:
            return None
        else:
            depth = next_layer[0].depth
            if depth < len(self.profit):
                return depth
            else:
                return None

class KnapsackRelax:
    '''
    This is the problem relaxation 
    '''
    def __init__(self, problem):
        self.problem = problem

    def merge(self, states):
        depth = 0
        capa  = 0
        for s in states:
            depth = max(depth, s.depth)
            capa  = max(capa,  s.capa)
        return KnapsackState(depth, capa)

    def relax(self, edge):
        return edge["cost"]

    def fast_upper_bound(self, state):
        '''
        Implementing this method is not mandatory but it has the potential
        to greatly speed up the resolution of your problem.
        
        If you implement it wrong, it also has the potential to make the 
        resolution of your problem seem impossible when it is not actually
        the case (try returning -9223372036854775808 if you want to see).
        '''
        #return -9223372036854775808
        tot = 0
        for i in range(state.depth, self.problem.nb_variables()):
            tot += self.problem.profit[i]
        return tot


class KnapsackRanking:
    '''
    An heuristic to discriminate the better states from the worse ones
    '''
    def compare(self, a, b):
        return a.capa - b.capa


#### UTILS #######################
def read_instance(fname):
    '''
    returns a tuple (capa, [profits], [weights])
    '''
    with open(fname) as f:
        is_first = True
        n        = 0
        capa     = 0
        count    = 0
        profits  = []
        weights  = []
        for line in f.readlines():
            if is_first:
                is_first = False
                a,b  = line.split(" ")
                n    = int(a)
                capa = int(b)
            else:
                if count >= n:
                    break
                count += 1
                a,b = line.split(" ")
                profits.append(int(a))
                weights.append(int(b))
        return (capa, profits, weights)

def get_path(instance):
    dirname = os.path.dirname(__file__)
    return os.path.join(dirname, "..", "..", "..", "resources", "knapsack", instance)

if __name__ == "__main__":
    instance  = "f8_l-d_kp_23_10000"
    (c, p, w) = read_instance(get_path(instance))
    problem   = Knapsack(c, p, w)
    relax     = KnapsackRelax(problem)
    ranking   = KnapsackRanking()
    result    = ddo.maximize(problem, relax, ranking, True, True, True, 100, 15)
    print("Duration:   {:.3f} seconds \nObjective:  {}\nUpper Bnd:  {}\nLower Bnd:  {}\nGap:        {}\nAborted:    {}\nSolution:   {}"
        .format(
            result.duration, 
            result.objective, 
            result.upper_bound, 
            result.lower_bound, 
            result.gap, 
            result.aborted, 
            result.assignment))