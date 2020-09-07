# Author: Martin Kristiansen

from itertools import combinations
import itertools
import math


class CostBenefit(object):
    '''
    Experimental class for cost-benefit calculations.

    Currently including allocation schemes based on "cooperative game theory"
    '''


    def __init__(self):
        """Collect value-functions for each player in the expansion-game

        Parameters
        ----------


        Creat CostBenefit object:
        """
        self.players = None
        self.coalitions = None
        self.valueFunction = None
        self.payoff = None

    def power_set(self,List):
        """
        function to return the powerset of a list, i.e. all possible subsets ranging
        from length of one, to the length of the larger list
        """

        subs = [list(j) for i in range(len(List)) for j in combinations(List, i+1)]
        return subs

    def nCr(self,n,r):
        '''
        calculate the binomial coefficient, i.e. how many different possible
        subsets can be made from the larger set n
        '''
        f = math.factorial
        return f(n) / f(r) / f(n-r)


    def gameSetup(self, grid_data):
        self.players = grid_data.node.area.unique().tolist()
        self.coalitions = self.power_set(self.players)


    def getBinaryCombinations(self,num):
        '''
        Returns a sequence of different combinations 1/0 for a number of
        decision variables. E.g. three cable investments;
        (0,0,0), (1,0,0), (0,1,0), and so on.
        '''
        combinations = list(itertools.product([0,1], repeat=num))
        return combinations


    def gameShapleyValue(self,player_list, values):
        '''compute the Shapley Value from cooperative game theory

       Parameters:
        ===========
        player_list: list of all players in the game
        values: characteristic function for each subset of N players, i.e.
                possible coaltions/cooperations among players.

        Returns the Shapley value, i.e. a fair cost/benefit allocation based
        on the average marginal contribution from each player.

        '''

        if type(values) is not dict:
            raise TypeError("characteristic function must be a dictionary")
        for key in values:
            if len(str(key)) == 1 and type(key) is not tuple:
                values[(key,)] = values.pop(key)
            elif type(key) is not tuple:
                raise TypeError("key must be a tuple")
        for key in values:
            sortedkey = tuple(sorted(list(key)))
            values[sortedkey] = values.pop(key)

        player_list = max(values.keys(), key=lambda key: len(key))
        for coalition in self.power_set(player_list):
            if tuple(sorted(list(coalition))) not in sorted(values.keys()):
                raise ValueError("characteristic function must be the power set")

        payoff_vector = {}
        n = len(player_list)
        for player in player_list:
            weighted_contribution = 0
            for coalition in self.power_set(player_list):
                if coalition:  # If non-empty
                    k = len(coalition)
                    weight = 1/(self.nCr(n,k)*k)
                    t = tuple(p for p in coalition if p != player)
                    weighted_contribution += weight * (values[tuple(coalition)]
                                                       - values[t])
            payoff_vector[player] = weighted_contribution

        return payoff_vector

    def gameIsMonotone(self, values):
        '''
        Returns true if the game/valueFunction is monotonic.
        A game G = (N, v) is monotonic if it satisfies the value function
        of a subset is less or equal then the value function from its
        union set:
        v(C_2) \geq v(C_1) for all C_1 \subseteq C_2
        '''
        return not any([set(p1) <= set(p2) and values[p1] > values[p2]
            for p1, p2 in itertools.permutations(values.keys(), 2)])


    def gameIsSuperadditive(self, values):
        '''
        Returns true if the game/valueFunction is superadditive.
        A characteristic function game G = (N, v) is superadditive
        if it the sum of two coalitions/subsets gives a larger value than the
        individual sum:
        v(C_1 \cup C_2) \geq v(C_1) +  v(C_2) for
        all C_1, C_2 \subseteq 2^{\Omega} such that C_1 \cap C_2
        = \emptyset.
        '''
        sets = values.keys()
        for p1, p2 in itertools.combinations(sets, 2):
            if not (set(p1) & set(p2)):
                union = tuple(sorted(set(p1) | set(p2)))
                if values[union] < values[p1] + values[p2]:
                    return False
        return True

    def gamePayoffIsEfficient(self, player_list, values, payoff_vector):
        '''
        Return `true if the payoff vector is efficient. A payoff vector v is
        efficient if the sum of payments equal the total value provided by a
        set of players.
        \sum_{i=1}^N \lambda_i = v(\Omega);
        '''
        pl = tuple(sorted(list(player_list)))
        return sum(payoff_vector.values()) == values[pl]

    def gamePayoffHasNullplayer(self, player_list, values, payoff_vector):
        '''
        Return true if the payoff vector possesses the nullplayer property.
        A payoff vector v has the nullplayer property if there exists
        an i such that v(C \cup i) = v(C) for all C \in 2^{\Omega}
        then, \lambda_i = 0. In other words: if a player does not
        contribute to any coalition then that player should receive no payoff.
        '''
        for player in player_list:
            results = []
            for coalit in values:
                if player in coalit:
                    t = tuple(sorted(set(coalit) - {player}))
                    results.append(values[coalit] == values[t])
            if all(results) and payoff_vector[player] != 0:
                return False
        return True

    def gamePayoffIsSymmetric(self, values, payoff_vector):
        '''
        Returns true if the resulting payoff vector possesses the symmetry property.
        A payoff vector possesses the symmetry property if players with equal
        marginal contribution receives the same payoff:
        v(C \cup i) = v(C \cup j) for all
        C \in 2^{\Omega} \setminus \{i,j\}, then x_i = x_j.
        '''
        sets = values.keys()
        element = [i for i in sets if len(i) == 1]
        for c1, c2 in itertools.combinations(element, 2):
            results = []
            for m in sets:
                junion = tuple(sorted(set(c1) | set(m)))
                kunion = tuple(sorted(set(c2) | set(m)))
                results.append(values[junion] == values[kunion])
            if all(results) and payoff_vector[c1[0]] != payoff_vector[c2[0]]:
                return False
        return True
