from ADP.utilities import *
from math import e


@timeit
def value_iteration(env, epsilon=e**-20, alpha=0.9):

    def lookahead(state, J):
        # INITIALISE ACTION DICTIONARY
        A = {}
        for _, a in enumerate(env.possible_actions(state)):
            a_idx = env.action_list.index(a)
            A[a_idx] = 0.0
        for _, act in enumerate(env.possible_actions(state)):
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
            A = lookahead(s, J)
            # min_action_cost = np.min(A)
            min_action_cost = A[min(A, key=A.get)]
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(min_action_cost - J[s]))

            # Update the cost function
            J[s] = min_action_cost
            # Check if we can stop
        j_plot.append(np.linalg.norm(J))

        if delta < epsilon:
            break

    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.num_states, env.num_actions])
    for s in range(env.num_states):
        # One step lookahead to find the best action for this state
        A = lookahead(s, J)
        # best_action = np.argmin(A)
        best_action = min(A, key=A.get)
        # Always take the best action
        policy[s, best_action] = 1.0

    return policy, J, j_plot


def policy_eval(policy, env, alpha=0.9, epsilon=e**-20):

    # Start with a random (e.g. all 0) value function
    J = np.zeros(env.num_states)
    while True:
        delta = 0

        for s in range(env.num_states):
            v = 0
            # Look at the possible next actions
            for _, act in enumerate(env.possible_actions(s)):
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


@timeit
def policy_iteration(env, policy_eval_fn=policy_eval, epsilon=e**-20, alpha=0.9):

    def lookahead(state, J):

        # INITIALISE ACTION DICTIONARY
        A = {}
        for _, a in enumerate(env.possible_actions(state)):
            a_idx = env.action_list.index(a)
            A[a_idx] = 0.0
        for _, act in enumerate(env.possible_actions(state)):
            an = env.action_list.index(act)
            for prob, nxt_state, cost in env.P_g[state][an]:
                A[an] += prob * (cost + alpha * J[nxt_state])
        return A


    # Start with empty policy
    policy = np.zeros([env.num_states, env.num_actions])
    j_plot = []
    while True:
        # Evaluate the current policy
        J = policy_eval_fn(policy, env, alpha, epsilon=epsilon)
        j_plot.append(np.linalg.norm(J))
        # Will be set to false if we make any changes to the policy
        policy_stable = True

        for s in range(env.num_states):
            # The best action we would take under the current policy
            chosen_a = np.argmax(policy[s])

            # Find the best action by one-step lookahead
            action_values = lookahead(s, J)
            best_a = min(action_values, key=action_values.get)

            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False

            policy[s] = np.eye(env.num_actions)[best_a]

        # If the policy is stable an optimal policy is found
        if policy_stable:
            return policy, J, j_plot







