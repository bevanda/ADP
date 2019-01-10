# here is the main for running the algorithm
from typing import Tuple
import numpy as np
from ADP.environments import *
from ADP.algorithms import *
from Tkinter import *
from ADP.utilities import *
# to load the absolute path of the .txt Maze into the
import sys


def value_iteration(env, theta=0.001, discount_factor=0.9):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):

        A = np.zeros(env.num_actions)
        for a in range(env.num_actions):
            for prob, next_state, cost in env.P_g[state][a]:
                A[a] += prob * (cost + discount_factor * V[next_state])
        return A

    V = np.zeros(env.num_states)
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.num_states):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.min(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            V[s] = best_action_value
            # Check if we can stop
        if delta < theta:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.num_states, env.num_actions])
    for s in range(env.num_states):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmin(A)
        # Always take the best action
        policy[s, best_action] = 1.0

    return policy, V




if __name__ == "__main__":

    env = Maze()

    print value_iteration(env)
    #print policy_improvement(env)








