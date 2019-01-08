# here is the main for running the algorithm
from typing import Tuple
import numpy as np
from ADP.environments import *
from ADP.algorithms import *
from Tkinter import *
from ADP.utilities import *
# to load the absolute path of the .txt Maze into the
import sys

if __name__ == "__main__":
    env = Maze()

    print env.shape

    #print env.num_states
    # print env.subs2action(0)
    #print env.subs2idx(env.idx2subs((3, 1)))
    print 'Init state:{}'.format(env.current_state)
    action = 'right'
    print 'Action{}'.format(action)

    #env.action_execution(action)
    #print 'New state:{}'.format(env.current_state)
    # env.current_state = 2, 7
    # no_drift = 0
    # u_drift = 0
    # d_drift = 0
    # for i in range(1000):
    #     env.action_execution(action)
    #     if (2,8) == env.current_state:
    #         no_drift +=1
    #     elif (1,8) == env.current_state:
    #         u_drift += 1
    #     elif (3,8) == env.current_state:
    #         d_drift += 1
    #     env.current_state = 2, 7
    #
    #
    # print no_drift
    # print u_drift
    # print d_drift



    V = np.zeros(env.num_states)
    policy = env.rand_policy()













    def value_iteration(env, theta=0.0001, discount_factor=1.0):
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
            """
            Helper function to calculate the value for all action in a given state.

            Args:
                state: The state to consider (int)
                V: The value to use as an estimator, Vector of length env.nS

            Returns:
                A vector of length env.nA containing the expected value of each action.
            """
            A = np.zeros(env.num_states)
            for a in range(env.num_actions):
                for prob, next_state, reward, done in env.P[state][a]:
                    A[a] += prob * (reward + discount_factor * V[next_state])
            return A

        V = np.zeros(env.num_states)
        while True:
            # Stopping condition
            delta = 0
            # Update each state...
            for s in range(env.num_states):
                # Do a one-step lookahead to find the best action
                A = one_step_lookahead(s, V)
                best_action_value = np.max(A)
                # Calculate delta across all states seen so far
                delta = max(delta, np.abs(best_action_value - V[s]))
                # Update the value function. Ref: Sutton book eq. 4.10.
                V[s] = best_action_value
                # Check if we can stop
            if delta < theta:
                break

        # Create a deterministic policy using the optimal value function
        policy = np.zeros([env.nS, env.nA])
        for s in range(env.nS):
            # One step lookahead to find the best action for this state
            A = one_step_lookahead(s, V)
            best_action = np.argmax(A)
            # Always take the best action
            policy[s, best_action] = 1.0

        return policy, V

