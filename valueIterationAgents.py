# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        allstates = self.mdp.getStates()
        for k in range(0, self.iterations):
          vi1 = self.values.copy()
          for square in allstates:
            act = self.computeActionFromValues(square)
            if act != None:
              vi1[square] = self.computeQValueFromValues(square, act)
          self.values = vi1


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        total = 0
        for pair in self.mdp.getTransitionStatesAndProbs(state, action):
          total += pair[1] * (self.mdp.getReward(state, action, pair[0]) + (self.discount * self.getValue(pair[0])))
        return total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
          return None
        action = None
        Q = float("-inf")
        for act in self.mdp.getPossibleActions(state):
          q = self.computeQValueFromValues(state, act)
          if q > Q:
            action = act
            Q = q
        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        allstates = self.mdp.getStates()
        k = 0
        while k in range(0, self.iterations):
          vi1 = self.values.copy()
          square = allstates[k % len(allstates)]
          act = ValueIterationAgent.computeActionFromValues(self, square)
          if act != None:
            vi1[square] = ValueIterationAgent.computeQValueFromValues(self, square, act)
          self.values = vi1
          k += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        preds = []
        allstates = self.mdp.getStates()
        i = 0
        for s in  allstates:
          preds.append(set())
          i += 0
        for state in allstates:
          acts = self.mdp.getPossibleActions(state)
          for act in acts:
            nextstates = self.mdp.getTransitionStatesAndProbs(state, act)
            for ns in nextstates:
              index = 0
              for i in range(0, len(allstates)):
                if allstates[i] == ns[0]:
                  index = i
                  break
              if ns[1] > 0:
                preds[i].add(state)

        pq = util.PriorityQueue()

        for s in allstates:
          if not self.mdp.isTerminal(s):
            diff = 0
            sval = self.values[s]
            qval = float("-inf")
            for act in self.mdp.getPossibleActions(s):
              qval = max(qval, ValueIterationAgent.computeQValueFromValues(self, s, act))
            diff = abs(sval - qval)
            pq.push(s, -diff)
        for it in range(0, self.iterations):
          if pq.isEmpty():
            break
          else:
            s = pq.pop()
            index = 0
            for i in range(0, len(allstates)):
              if allstates[i] == s:
                index = i
                break
            if not self.mdp.isTerminal(s):
              a = ValueIterationAgent.computeActionFromValues(self, s)
              if act != None:
                self.values[s] = self.computeQValueFromValues(s, a)
              else:
                self.values[s] = 0
              for pre in preds[i]:
                pval = self.values[pre]
                qvalp = float("-inf")
                for actp in self.mdp.getPossibleActions(pre):
                  qvalp = max(qvalp, ValueIterationAgent.computeQValueFromValues(self, pre, actp))
                diffp = abs(pval - qvalp)
                if diffp > self.theta:
                  pq.update(pre, -diffp)
