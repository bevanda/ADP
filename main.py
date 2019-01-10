# here is the main for running the algorithm
from typing import Tuple
import numpy as np
from ADP.environments import *
from ADP.algorithms import *
from Tkinter import *
from ADP.utilities import *
# to load the absolute path of the .txt Maze into the
import sys


def value_iteration(env, theta=0.01, discount_factor=0.9):
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
            delta = min(delta, np.abs(best_action_value - V[s]))
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


def policy_eval(policy, env, discount_factor=0.9, theta=0.01):

    # Start with a random (all 0) value function
    V = np.zeros(env.num_states)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.num_states):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, cost in env.P_g[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (cost + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=0.9):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.num_states

        Returns:
            A vector of length env.num_actions containing the expected value of each action.
        """
        A = np.zeros(env.num_actions)
        for a in range(env.num_actions):
            for prob, next_state, cost in env.P_g[state][a]:
                A[a] += prob * (cost + discount_factor * V[next_state])
        return A

    # Start with a random policy
    policy = np.ones([env.num_states, env.num_actions]) / env.num_actions

    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)

        # Will be set to false if we make any changes to the policy
        policy_stable = True

        # For each state...
        for s in range(env.num_states):
            # The best action we would take under the correct policy
            chosen_a = np.argmin(policy[s])

            # Find the best action by one-step lookahead
            # Ties are resolved arbitrarily
            action_values = one_step_lookahead(s, V)
            best_a = np.argmin(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.num_actions)[best_a]

        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V


if __name__ == "__main__":

    env = Maze()

    #print value_iteration(env)
    print policy_improvement(env)








