# here is the main for running the algorithm
from typing import Tuple
import numpy as np
from ADP.environments import *
from ADP.algorithms import *
from Tkinter import *
from ADP.utilities import *
# to load the absolute path of the .txt Maze into the
import sys
import matplotlib.pyplot as plt


def value_iteration(env, epsilon=0.0001, alpha=0.9):
    """
    Value Iteration Algorithm.

    Args:
        env: Maze environment for ADPRL course WS2018  env. env.P_g represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, nxt_state, reward, done).
            env.num_states is a number of states in the environment.
            env.num_actions is a number of actions in the environment.
        epsilon: We stop evaluation once our value function change is less than epsilon for all states.
        alpha: Gamma discount factor.

    Returns:
        A tuple (policy, J) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, J):

        A = np.zeros(env.num_actions)
        for a, act in enumerate(env.possible_actions(state)):
            an = env.action_list.index(act)
            for prob, nxt_state, cost in env.P_g[state][an]:
                A[an] += prob * (cost + alpha * J[nxt_state])
        return A

    J = np.zeros(env.num_states)
    j_plot = []
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.num_states):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, J)
            best_action_value = np.min(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - J[s]))
            # Update the value function. Ref: Sutton book eq. 4.10.
            J[s] = best_action_value
            # Check if we can stop
        j_plot.append(np.linalg.norm(J))

        if delta < epsilon:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.num_states, env.num_actions])
    for s in range(env.num_states):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, J)
        best_action = np.argmin(A)
        # Always take the best action
        policy[s, best_action] = 1.0

    return policy, J, j_plot



# Taken from Policy Evaluation Exercise!

def policy_eval(policy, env, alpha=0.9, epsilon=0.0001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: Maze environment for ADPRL course WS2018 env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, nxt_state, reward, done).
            env.num_states is a number of states in the environment.
            env.num_actions is a number of actions in the environment.
        epsilon: We stop evaluation once our value function change is less than epsilon for all states.
        alpha: Gamma discount factor.

    Returns:
        Vector of length env.num_states representing the value function.
    """
    # Start with a random (all 0) value function
    J = np.zeros(env.num_states)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.num_states):
            v = 0
            # Look at the possible next actions
            for a, act in enumerate(env.possible_actions(s)):
            # for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                an = env.action_list.index(act)
                action_prob = policy[s][an]
                for prob, nxt_state, cost in env.P_g[s][an]:
                    # Calculate the expected value
                    v += action_prob * prob * (cost + alpha * J[nxt_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - J[s]))
            J[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < epsilon:
            break
    return np.array(J)


def policy_improvement(env, policy_eval_fn=policy_eval, alpha=0.9):

    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The Maze envrionment for ADPRL course WS2018.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, alpha.
        alpha: alpha discount factor.

    Returns:
        A tuple (policy, J).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        J is the value function for the optimal policy.

    """

    def one_step_lookahead(state, J):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            J: The value to use as an estimator, Vector of length env.num_states

        Returns:
            A vector of length env.num_actions containing the expected value of each action.
        """

        A = np.zeros(env.num_actions)
        for a, act in enumerate(env.possible_actions(state)):
            an = env.action_list.index(act)
            for prob, nxt_state, cost in env.P_g[state][an]:
                A[an] += prob * (cost + alpha * J[nxt_state])
        return A

    # Start with a random policy
    policy = np.ones([env.num_states, env.num_actions]) / env.num_actions
    j_plot = []
    while True:
        # Evaluate the current policy
        J = policy_eval_fn(policy, env, alpha)
        j_plot.append(np.linalg.norm(J))
        # Will be set to false if we make any changes to the policy
        policy_stable = True

        # For each state...
        for s in range(env.num_states):
            # The best action we would take under the current policy
            chosen_a = np.argmax(policy[s])

            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s, J)
            best_a = np.argmin(action_values)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False

            policy[s] = np.eye(env.num_actions)[best_a]

        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, J, j_plot


if __name__ == "__main__":

    env = Maze()

    policyVI, value_function, VI_plot = value_iteration(env)
    print policyVI
    plt.plot(VI_plot)
    plt.show()

    heatmap(env, value_function)

    policyPI, value_function, PI_plot = policy_improvement(env)
    print policyPI
    plt.plot(PI_plot)
    plt.show()
    print policy_improvement(env)


    heatmap(env, value_function)

    print np.array_equal(policyPI, policyVI)





