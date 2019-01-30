from ADP.environments import *
from ADP.algorithms import value_iteration, policy_iteration
import matplotlib.pyplot as plt
import matplotlib
from numpy.random import *


if __name__ == "__main__":
    #******Runung alg to obtain ground truth*****

    env = Maze(g=1)

    smallest_num = np.nextafter(0, 1)

    policyVI_ground_truth_g1, VI_cost_function_ground_truth_g1, VI_J2_ground_truth_g1 = \
        value_iteration(env, np.zeros(env.num_states),
                        epsilon=smallest_num,
                        alpha=0.99)

    policyPI_ground_truth_g1, PI_cost_function_ground_truth_g1, PI_J2_ground_truth_g1 = \
        policy_iteration(env, np.zeros(env.num_states),
                         epsilon=smallest_num,
                         alpha=0.99)

    plot = visualise(env, VI_cost_function_ground_truth_g1, policyVI_ground_truth_g1)
    plot.title("Value Iteration: $g_1$, $\\alpha=0.99$ - 'ground truth'")
    plot.show()

    plot = visualise(env, PI_cost_function_ground_truth_g1, policyPI_ground_truth_g1)
    plot.title("Policy Iteration: $g_1$, $\\alpha=0.99$ - 'ground truth'")
    plot.show()

    env = Maze(g=2)

    policyVI_ground_truth_g2, VI_cost_function_ground_truth_g2, VI_J2_ground_truth_g2 = \
        value_iteration(env, np.zeros(env.num_states),
                        epsilon=smallest_num,
                        alpha=0.99)

    policyPI_ground_truth_g2, PI_cost_function_ground_truth_g2, PI_J2_ground_truth_g2 = \
        policy_iteration(env, np.zeros(env.num_states),
                         epsilon=smallest_num,
                         alpha=0.99)

    plot = visualise(env, VI_cost_function_ground_truth_g2, policyVI_ground_truth_g2)
    plot.title("Value Iteration: $g_2$, $\\alpha=0.99$ - 'ground truth'")
    plot.show()

    plot = visualise(env, PI_cost_function_ground_truth_g2, policyPI_ground_truth_g2)
    plot.title("Policy Iteration: $g_2$, $\\alpha=0.99$ - 'ground truth'")
    plot.show()

    # ******Visualising for J and /mu for  $g_1$************
    env = Maze(g=1)

    policyVI_1, VI_cost_function_1, VI_J2_1 = value_iteration(env, VI_cost_function_ground_truth_g1,
                                                              epsilon=smallest_num,
                                                              alpha=0.9)

    policyPI_1, PI_cost_function_1, PI_J2_1 = policy_iteration(env, PI_cost_function_ground_truth_g1,
                                                               epsilon=smallest_num,
                                                               alpha=0.9)

    print 'Ground truth policies equal 1)?\n{}'.format(np.array_equal(policyPI_1, policyVI_1))
    print 'Cost functions equal 1)?\n{}'.format(np.array_equal(VI_cost_function_1, PI_cost_function_1))

    plot = visualise(env, VI_cost_function_1, policyVI_1)
    plot.title("Value Iteration: $g_1$, $\\alpha=0.9$  ")
    plot.show()
    plot = visualise(env, PI_cost_function_1, policyPI_1)
    plot.title("Policy Iteration: $g_1$, $\\alpha=0.9$  ")
    plot.show()

    plt.plot(VI_J2_1)
    plt.title("Squared distance to ground truth VI $g_1$, $\\alpha=0.9$")
    plt.show()
    plt.plot(PI_J2_1)
    plt.title("Squared distance to ground truth PI $g_1$, $\\alpha=0.9$")
    plt.show()

    env = Maze(g=1)

    policyVI_2, VI_cost_function_2, VI_J2_2 = value_iteration(env, VI_cost_function_ground_truth_g1,
                                                              epsilon=smallest_num,
                                                              alpha=0.5)

    policyPI_2, PI_cost_function_2, PI_J2_2 = policy_iteration(env, PI_cost_function_ground_truth_g1,
                                                               epsilon=smallest_num,
                                                               alpha=0.5)

    print 'Policies equal 2)?\n{}'.format(np.array_equal(policyPI_2, policyVI_2))
    print 'Cost functions equal2)?\n{}'.format(np.array_equal(VI_cost_function_2, PI_cost_function_2))
    plot = visualise(env, VI_cost_function_2, policyVI_2)
    plot.title("Value Iteration: $g_1$, $\\alpha=0.5$  ")
    plot.show()
    plot = visualise(env, PI_cost_function_2, policyPI_2)
    plot.title("Policy Iteration: $g_1$, $\\alpha=0.5$  ")
    plot.show()

    plt.plot(VI_J2_2)
    plt.title("Squared distance to ground truth VI $g_1$, $\\alpha=0.5$")
    plt.show()
    plt.plot(PI_J2_2)
    plt.title("Squared distance to ground truth PI $g_1$, $\\alpha=0.5$")
    plt.show()

    env = Maze(g=1)

    policyVI_3, VI_cost_function_3, VI_J2_3 = value_iteration(env, VI_cost_function_ground_truth_g1,
                                                              epsilon=smallest_num,
                                                              alpha=0.01)

    policyPI_3, PI_cost_function_3, PI_J2_3 = policy_iteration(env, PI_cost_function_ground_truth_g1,
                                                               epsilon=smallest_num,
                                                               alpha=0.01)

    print 'Policies equal 3)?\n{}'.format(np.array_equal(policyPI_3, policyVI_3))
    print 'Cost functions equal 3)?\n{}'.format(np.array_equal(VI_cost_function_3, PI_cost_function_3))

    plot = visualise(env, VI_cost_function_3, policyVI_3)
    plot.title("Value Iteration: $g_1$, $\\alpha=0.01$  ")
    plot.show()
    plot = visualise(env, PI_cost_function_3, policyPI_3)
    plot.title("Policy Iteration: $g_1$, $\\alpha=0.01$  ")
    plot.show()

    plt.plot(VI_J2_3)
    plt.title("Squared distance to ground truth VI $g_1$, $\\alpha=0.01$")
    plt.show()
    plt.plot(PI_J2_3)
    plt.title("Squared distance to ground truth PI $g_1$, $\\alpha=0.01$")
    plt.show()

    # ******Visualising for J and /mu for $g_2$********

    env = Maze(g=2)

    policyVI_1, VI_cost_function_1, VI_J2_1 = value_iteration(env, VI_cost_function_ground_truth_g2,
                                                              epsilon=smallest_num,
                                                              alpha=0.9)

    policyPI_1, PI_cost_function_1, PI_J2_1 = policy_iteration(env, PI_cost_function_ground_truth_g2,
                                                               epsilon=smallest_num,
                                                               alpha=0.9)

    print 'Ground truth policies equal 1)?\n{}'.format(np.array_equal(policyPI_1, policyVI_1))
    print 'Cost functions equal 1)?\n{}'.format(np.array_equal(VI_cost_function_1, PI_cost_function_1))

    plot = visualise(env, VI_cost_function_1, policyVI_1)
    plot.title("Value Iteration: $g_2$, $\\alpha=0.9$  ")
    plot.show()
    plot = visualise(env, PI_cost_function_1, policyPI_1)
    plot.title("Policy Iteration: $g_2$, $\\alpha=0.9$  ")
    plot.show()

    plt.plot(VI_J2_1)
    plt.title("Squared distance to ground truth VI $g_2$, $\\alpha=0.9$")
    plt.show()
    plt.plot(PI_J2_1)
    plt.title("Squared distance to ground truth PI $g_2$, $\\alpha=0.9$")
    plt.show()

    env = Maze(g=2)

    policyVI_2, VI_cost_function_2, VI_J2_2 = value_iteration(env, VI_cost_function_ground_truth_g2,
                                                              epsilon=smallest_num,
                                                              alpha=0.5)

    policyPI_2, PI_cost_function_2, PI_J2_2 = policy_iteration(env, PI_cost_function_ground_truth_g2,
                                                               epsilon=smallest_num,
                                                               alpha=0.5)

    print 'Policies equal 2)?\n{}'.format(np.array_equal(policyPI_2, policyVI_2))
    print 'Cost functions equal2)?\n{}'.format(np.array_equal(VI_cost_function_2, PI_cost_function_2))
    plot = visualise(env, VI_cost_function_2, policyVI_2)
    plot.title("Value Iteration: $g_2$, $\\alpha=0.5$  ")
    plot.show()
    plot = visualise(env, PI_cost_function_2, policyPI_2)
    plot.title("Policy Iteration: $g_2$, $\\alpha=0.5$  ")
    plot.show()

    plt.plot(VI_J2_2)
    plt.title("Squared distance to ground truth VI $g_2$, $\\alpha=0.5$")
    plt.show()
    plt.plot(PI_J2_2)
    plt.title("Squared distance to ground truth PI $g_2$, $\\alpha=0.5$")
    plt.show()

    env = Maze(g=2)

    policyVI_3, VI_cost_function_3, VI_J2_3 = value_iteration(env, VI_cost_function_ground_truth_g2,
                                                              epsilon=smallest_num,
                                                              alpha=0.01)

    policyPI_3, PI_cost_function_3, PI_J2_3 = policy_iteration(env, PI_cost_function_ground_truth_g2,
                                                               epsilon=smallest_num,
                                                               alpha=0.01)



    print 'Policies equal 3)?\n{}'.format(np.array_equal(policyPI_3, policyVI_3))
    print 'Cost functions equal 3)?\n{}'.format(np.array_equal(VI_cost_function_3, PI_cost_function_3))

    plot = visualise(env, VI_cost_function_3, policyVI_3)
    plot.title("Value Iteration: $g_2$, $\\alpha=0.01$  ")
    plot.show()
    plot = visualise(env, PI_cost_function_3, policyPI_3)
    plot.title("Policy Iteration: $g_2$, $\\alpha=0.01$  ")
    plot.show()

    plt.plot(VI_J2_3)
    plt.title("Squared distance to ground truth VI $g_2$, $\\alpha=0.01$")
    plt.show()
    plt.plot(PI_J2_3)
    plt.title("Squared distance to ground truth PI $g_2$, $\\alpha=0.01$")
    plt.show()

    # ********Testing suitable alphas for VI PI comparison*********

    # ***Alpha=0.8****
    # ******Visualising for J and /mu for  $g_1$************
    env = Maze(g=1)

    policyVI_1, VI_cost_function_1, VI_J2_1 = value_iteration(env, VI_cost_function_ground_truth_g1,
                                                              epsilon=smallest_num,
                                                              alpha=0.8)

    policyPI_1, PI_cost_function_1, PI_J2_1 = policy_iteration(env, PI_cost_function_ground_truth_g1,
                                                               epsilon=smallest_num,
                                                               alpha=0.8)

    print 'Ground truth policies equal 1)?\n{}'.format(np.array_equal(policyPI_1, policyVI_1))
    print 'Cost functions equal 1)?\n{}'.format(np.array_equal(VI_cost_function_1, PI_cost_function_1))

    plot = visualise(env, VI_cost_function_1, policyVI_1)
    plot.title("Value Iteration: $g_1$, $\\alpha=0.8$  ")
    plot.show()
    plot = visualise(env, PI_cost_function_1, policyPI_1)
    plot.title("Policy Iteration: $g_1$, $\\alpha=0.8$  ")
    plot.show()

    plt.plot(VI_J2_1)
    plt.title("Squared distance to ground truth VI $g_1$, $\\alpha=0.8$")
    plt.show()
    plt.plot(PI_J2_1)
    plt.title("Squared distance to ground truth PI $g_1$, $\\alpha=0.8$")
    plt.show()

    # ***Alpha=0.7****
    env = Maze(g=1)

    policyVI_2, VI_cost_function_2, VI_J2_2 = value_iteration(env, VI_cost_function_ground_truth_g1,
                                                              epsilon=smallest_num,
                                                              alpha=0.7)

    policyPI_2, PI_cost_function_2, PI_J2_2 = policy_iteration(env, PI_cost_function_ground_truth_g1,
                                                               epsilon=smallest_num,
                                                               alpha=0.7)

    print 'Policies equal 2)?\n{}'.format(np.array_equal(policyPI_2, policyVI_2))
    print 'Cost functions equal2)?\n{}'.format(np.array_equal(VI_cost_function_2, PI_cost_function_2))
    plot = visualise(env, VI_cost_function_2, policyVI_2)
    plot.title("Value Iteration: $g_1$, $\\alpha=0.7$  ")
    plot.show()
    plot = visualise(env, PI_cost_function_2, policyPI_2)
    plot.title("Policy Iteration: $g_1$, $\\alpha=0.7$  ")
    plot.show()

    plt.plot(VI_J2_2)
    plt.title("Squared distance to ground truth VI $g_1$, $\\alpha=0.7$")
    plt.show()
    plt.plot(PI_J2_2)
    plt.title("Squared distance to ground truth PI $g_1$, $\\alpha=0.7$")
    plt.show()

    # ***Alpha=0.6****

    env = Maze(g=1)

    policyVI_3, VI_cost_function_3, VI_J2_3 = value_iteration(env, VI_cost_function_ground_truth_g1,
                                                              epsilon=smallest_num,
                                                              alpha=0.6)

    policyPI_3, PI_cost_function_3, PI_J2_3 = policy_iteration(env, PI_cost_function_ground_truth_g1,
                                                               epsilon=smallest_num,
                                                               alpha=0.6)

    print 'Policies equal 3)?\n{}'.format(np.array_equal(policyPI_3, policyVI_3))
    print 'Cost functions equal 3)?\n{}'.format(np.array_equal(VI_cost_function_3, PI_cost_function_3))

    plot = visualise(env, VI_cost_function_3, policyVI_3)
    plot.title("Value Iteration: $g_1$, $\\alpha=0.6$  ")
    plot.show()
    plot = visualise(env, PI_cost_function_3, policyPI_3)
    plot.title("Policy Iteration: $g_1$, $\\alpha=0.6$  ")
    plot.show()

    plt.plot(VI_J2_3)
    plt.title("Squared distance to ground truth VI $g_1$, $\\alpha=0.6$")
    plt.show()
    plt.plot(PI_J2_3)
    plt.title("Squared distance to ground truth PI $g_1$, $\\alpha=0.6$")
    plt.show()

    # ******Visualising for J and /mu for  $g_2$************

    # ***Alpha=0.8****
    env = Maze(g=2)

    policyVI_1, VI_cost_function_1, VI_J2_1 = value_iteration(env, VI_cost_function_ground_truth_g2,
                                                              epsilon=smallest_num,
                                                              alpha=0.8)

    policyPI_1, PI_cost_function_1, PI_J2_1 = policy_iteration(env, PI_cost_function_ground_truth_g2,
                                                               epsilon=smallest_num,
                                                               alpha=0.8)

    print 'Ground truth policies equal 1)?\n{}'.format(np.array_equal(policyPI_1, policyVI_1))
    print 'Cost functions equal 1)?\n{}'.format(np.array_equal(VI_cost_function_1, PI_cost_function_1))

    plot = visualise(env, VI_cost_function_1, policyVI_1)
    plot.title("Value Iteration: $g_2$, $\\alpha=0.8$  ")
    plot.show()
    plot = visualise(env, PI_cost_function_1, policyPI_1)
    plot.title("Policy Iteration: $g_2$, $\\alpha=0.8$  ")
    plot.show()

    plt.plot(VI_J2_1)
    plt.title("Squared distance to ground truth VI $g_2$, $\\alpha=0.8$")
    plt.show()
    plt.plot(PI_J2_1)
    plt.title("Squared distance to ground truth PI $g_2$, $\\alpha=0.8$")
    plt.show()

    # ***Alpha=0.7****
    env = Maze(g=2)

    policyVI_2, VI_cost_function_2, VI_J2_2 = value_iteration(env, VI_cost_function_ground_truth_g2,
                                                              epsilon=smallest_num,
                                                              alpha=0.7)

    policyPI_2, PI_cost_function_2, PI_J2_2 = policy_iteration(env, PI_cost_function_ground_truth_g2,
                                                               epsilon=smallest_num,
                                                               alpha=0.7)

    print 'Policies equal 2)?\n{}'.format(np.array_equal(policyPI_2, policyVI_2))
    print 'Cost functions equal2)?\n{}'.format(np.array_equal(VI_cost_function_2, PI_cost_function_2))
    plt.figure(1)
    plot = visualise(env, VI_cost_function_2, policyVI_2)
    plot.title("Value Iteration: $g_2$, $\\alpha=0.7$  ")
    plot.show()
    plt.figure(2)
    plot = visualise(env, PI_cost_function_2, policyPI_2)
    plot.title("Policy Iteration: $g_2$, $\\alpha=0.7$  ")
    plot.show()

    plt.figure(3)
    plt.plot(VI_J2_2)
    plt.title("Squared distance to ground truth VI $g_2$, $\\alpha=0.7$")
    plt.show()
    plt.figure(4)
    plt.plot(PI_J2_2)
    plt.title("Squared distance to ground truth PI $g_2$, $\\alpha=0.7$")
    plt.show()

    # ***Alpha=0.6****

    env = Maze(g=2)

    policyVI_3, VI_cost_function_3, VI_J2_3 = value_iteration(env, VI_cost_function_ground_truth_g2,
                                                              epsilon=smallest_num,
                                                              alpha=0.6)

    policyPI_3, PI_cost_function_3, PI_J2_3 = policy_iteration(env, PI_cost_function_ground_truth_g2,
                                                               epsilon=smallest_num,
                                                               alpha=0.6)

    print 'Policies equal 3)?\n{}'.format(np.array_equal(policyPI_3, policyVI_3))
    print 'Cost functions equal 3)?\n{}'.format(np.array_equal(VI_cost_function_3, PI_cost_function_3))

    plot = visualise(env, VI_cost_function_3, policyVI_3)
    plot.title("Value Iteration: $g_2$, $\\alpha=0.6$  ")
    plot.show()
    plot = visualise(env, PI_cost_function_3, policyPI_3)
    plot.title("Policy Iteration: $g_2$, $\\alpha=0.6$  ")
    plot.show()

    plt.plot(VI_J2_3)
    plt.title("Squared distance to ground truth VI $g_2$, $\\alpha=0.6$")
    plt.show()
    plt.plot(PI_J2_3)
    plt.title("Squared distance to ground truth PI $g_2$, $\\alpha=0.6$")
    plt.show()

    # *******Putting in the reference cost of ground truth into the 0.99 case *******

    env = Maze(g=1)

    smallest_num = np.nextafter(0, 1)

    policyVI_ground_truth_g1, VI_cost_function_ground_truth_g1, VI_J2_ground_truth_1 = \
        value_iteration(env, VI_cost_function_ground_truth_g1,
                        epsilon=smallest_num,
                        alpha=0.99)

    policyPI_ground_truth_g1, PI_cost_function_ground_truth_g1, PI_J2_ground_truth_1 = \
        policy_iteration(env, PI_cost_function_ground_truth_g1,
                         epsilon=smallest_num,
                         alpha=0.99)

    plt.plot(VI_J2_ground_truth_1)
    plt.title("Squared distance to ground truth VI $g_1$, $\\alpha=0.99$")
    plt.show()
    plt.plot(PI_J2_ground_truth_1)
    plt.title("Squared distance to ground truth PI $g_1$, $\\alpha=0.99$")
    plt.show()

    env = Maze(g=2)

    policyVI_ground_truth_g2, VI_cost_function_ground_truth_g2, VI_J2_ground_truth_2 = \
        value_iteration(env, VI_cost_function_ground_truth_g2,
                        epsilon=smallest_num,
                        alpha=0.99)

    policyPI_ground_truth_g2, PI_cost_function_ground_truth_g2, PI_J2_ground_truth_2 = \
        policy_iteration(env, PI_cost_function_ground_truth_g2,
                         epsilon=smallest_num,
                         alpha=0.99)

    plt.plot(VI_J2_ground_truth_2)
    plt.title("Squared distance to ground truth VI $g_2$, $\\alpha=0.99$")
    plt.show()
    plt.plot(PI_J2_ground_truth_2)
    plt.title("Squared distance to ground truth PI $g_2$, $\\alpha=0.99$")
    plt.show()
