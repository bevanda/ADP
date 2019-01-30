from ADP.environments import *
from ADP.algorithms import value_iteration, policy_iteration
import matplotlib.pyplot as plt
from numpy.random import *
import scipy.spatial.distance

if __name__ == "__main__":

    env = Maze(g=1)

    smallest_num = np.nextafter(0, 1)

    policyVI_ground_truth_g1, VI_cost_function_ground_truth_g1, VI_J2_ground_truth_g1 = \
                                                            value_iteration(env, np.zeros(env.num_states),
                                                          #epsilon=smallest_num,
                                                          alpha=0.99)

    policyPI_ground_truth_g1, PI_cost_function_ground_truth_g1, PI_J2_ground_truth_g1 = \
                                                            policy_iteration(env,  np.zeros(env.num_states),
                                                           #epsilon=smallest_num,
                                                           alpha=0.99)

    # print 'Ground truth policies equal 1)?\n{}'.format(np.array_equal(policyPI_ground_truth_g1, policyVI_ground_truth_g1))
    # print 'Cost functions equal 1)?\n{}'.format(np.array_equal(VI_cost_function_ground_truth_g1, PI_cost_function_ground_truth_g1))
    plot = visualise(env, VI_cost_function_ground_truth_g1, policyVI_ground_truth_g1)
    plot.title("Value Iteration: $g_1$, $\\alpha=0.99$ - 'ground truth'")
    plot.show()
    # plot1.savefig("VI_GT_g1.png")
    plot = visualise(env, PI_cost_function_ground_truth_g1, policyPI_ground_truth_g1)
    plot.title("Policy Iteration: $g_1$, $\\alpha=0.99$ - 'ground truth'")
    plot.show()
    # plot2.savefig("PI_GT_g1.png")

    env = Maze(g=2)

    smallest_num = np.nextafter(0, 1)

    policyVI_ground_truth_g2, VI_cost_function_ground_truth_g2, VI_J2_ground_truth_g2 = \
                                                        value_iteration(env, np.zeros(env.num_states),
                                                          epsilon=smallest_num,
                                                          alpha=0.99)


    policyPI_ground_truth_g2, PI_cost_function_ground_truth_g2, PI_J2_ground_truth_g2 = \
                                                            policy_iteration(env, np.zeros(env.num_states),
                                                           epsilon=smallest_num,
                                                           alpha=0.99)



    # print 'Ground truth policies equal 1)?\n{}'.format(np.array_equal(policyPI_ground_truth_g2, policyVI_ground_truth_g2))
    # print 'Cost functions equal 1)?\n{}'.format(np.array_equal(VI_cost_function_ground_truth_g2, PI_cost_function_ground_truth_g2))

    plot = visualise(env, VI_cost_function_ground_truth_g2, policyVI_ground_truth_g2)
    plot.title("Value Iteration: $g_2$, $\\alpha=0.99$ - 'ground truth'")
    plot.show()
    # plot.savefig("VI_GT_g2.png")
    plot = visualise(env, PI_cost_function_ground_truth_g2, policyPI_ground_truth_g2)
    plot.title("Policy Iteration: $g_2$, $\\alpha=0.99$ - 'ground truth'")
    plot.show()
    # plot.savefig("PI_GT_g2.png")

# ******Visualising for J and /mu for  $g_1$************
    env = Maze(g=1)

    smallest_num = np.nextafter(0, 1)

    policyVI_1, VI_cost_function_1, VI_J2_1 = value_iteration(env, VI_cost_function_ground_truth_g1,
                                                          epsilon=smallest_num,
                                                          alpha=0.9)

    policyPI_1, PI_cost_function_1, PI_J2_1 = policy_iteration(env, PI_cost_function_ground_truth_g1,
                                                           epsilon=smallest_num,
                                                           alpha=0.9)



    print 'Ground truth policies equal 1)?\n{}'.format(np.array_equal(policyPI_1, policyVI_1))
    print 'Cost functions equal 1)?\n{}'.format(np.array_equal(VI_cost_function_1, PI_cost_function_1))

    plot = visualise(env, VI_cost_function_1, policyVI_1)
    plot.title("Policy Iteration: $g_1$, $\\alpha=0.9$  ")
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

    smallest_num = np.nextafter(0, 1)

    policyVI_2, VI_cost_function_2, VI_J2_2 = value_iteration(env, VI_cost_function_ground_truth_g1,
                                                          epsilon=smallest_num,
                                                          alpha=0.5)

    policyPI_2, PI_cost_function_2, PI_J2_2 = policy_iteration(env, PI_cost_function_ground_truth_g1,
                                                           epsilon=smallest_num,
                                                           alpha=0.5)


    print 'Policies equal 2)?\n{}'.format(np.array_equal(policyPI_2, policyVI_2))
    print 'Cost functions equal2)?\n{}'.format(np.array_equal(VI_cost_function_2, PI_cost_function_2))
    plot = visualise(env, VI_cost_function_2, policyVI_2)
    plot.title("Policy Iteration: $g_1$, $\\alpha=0.5$  ")
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

    smallest_num = np.nextafter(0, 1)

    policyVI_3, VI_cost_function_3, VI_J2_3 = value_iteration(env, VI_cost_function_ground_truth_g1,
                                                          epsilon=smallest_num,
                                                         alpha=0.01)

    policyPI_3, PI_cost_function_3, PI_J2_3 = policy_iteration(env, PI_cost_function_ground_truth_g1,
                                                           epsilon=smallest_num,
                                                           alpha=0.01)


    print 'Policies equal 3)?\n{}'.format(np.array_equal(policyPI_3, policyVI_3))
    print 'Cost functions equal 3)?\n{}'.format(np.array_equal(VI_cost_function_3, PI_cost_function_3))

    plot = visualise(env, VI_cost_function_3, policyVI_3)
    plot.title("Policy Iteration: $g_1$, $\\alpha=0.01$  ")
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

    smallest_num = np.nextafter(0, 1)

    policyVI_1, VI_cost_function_1, VI_J2_1 = value_iteration(env, VI_cost_function_ground_truth_g2,
                                                          epsilon=smallest_num,
                                                          alpha=0.9)

    policyPI_1, PI_cost_function_1, PI_J2_1 = policy_iteration(env, PI_cost_function_ground_truth_g2,
                                                           epsilon=smallest_num,
                                                           alpha=0.9)



    print 'Ground truth policies equal 1)?\n{}'.format(np.array_equal(policyPI_1, policyVI_1))
    print 'Cost functions equal 1)?\n{}'.format(np.array_equal(VI_cost_function_1, PI_cost_function_1))

    plot = visualise(env, VI_cost_function_1, policyVI_1)
    plot.title("Policy Iteration: $g_2$, $\\alpha=0.9$  ")
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

    smallest_num = np.nextafter(0, 1)

    policyVI_2, VI_cost_function_2, VI_J2_2 = value_iteration(env, VI_cost_function_ground_truth_g2,
                                                          epsilon=smallest_num,
                                                          alpha=0.5)

    policyPI_2, PI_cost_function_2, PI_J2_2 = policy_iteration(env, PI_cost_function_ground_truth_g2,
                                                           epsilon=smallest_num,
                                                           alpha=0.5)


    print 'Policies equal 2)?\n{}'.format(np.array_equal(policyPI_2, policyVI_2))
    print 'Cost functions equal2)?\n{}'.format(np.array_equal(VI_cost_function_2, PI_cost_function_2))
    plot = visualise(env, VI_cost_function_2, policyVI_2)
    plot.title("Policy Iteration: $g_2$, $\\alpha=0.5$  ")
    plot.show()
    plot = visualise(env, PI_cost_function_2, policyPI_2)
    plot.title("Policy Iteration: $g_2$, $\\alpha=0.5$  ")
    plot.show()

    plt.plot(VI_J2_2)
    plt.title("Squared distance to ground truth VI $g_2$, $\\alpha=0.01$")
    plt.show()
    plt.plot(PI_J2_2)
    plt.title("Squared distance to ground truth PI $g_2$, $\\alpha=0.01$")
    plt.show()

    env = Maze(g=2)

    smallest_num = np.nextafter(0, 1)

    policyVI_3, VI_cost_function_3, VI_J2_3 = value_iteration(env, VI_cost_function_ground_truth_g2,
                                                          epsilon=smallest_num,
                                                          alpha=0.01)

    policyPI_3, PI_cost_function_3, PI_J2_3 = policy_iteration(env, PI_cost_function_ground_truth_g2,
                                                           epsilon=smallest_num,
                                                           alpha=0.01)

    plt.plot(VI_J2_3)
    plt.title("Squared distance to ground truth VI $g_2$, $\\alpha=0.01$")
    plt.show()
    plt.plot(PI_J2_3)
    plt.title("Squared distance to ground truth PI $g_2$, $\\alpha=0.01$")
    plt.show()

    print 'Policies equal 3)?\n{}'.format(np.array_equal(policyPI_3, policyVI_3))
    print 'Cost functions equal 3)?\n{}'.format(np.array_equal(VI_cost_function_3, PI_cost_function_3))

    plot = visualise(env, VI_cost_function_3, policyVI_3)
    plot.title("Policy Iteration: $g_2$, $\\alpha=0.01$  ")
    plot.show()
    plot = visualise(env, PI_cost_function_3, policyPI_3)
    plot.title("Policy Iteration: $g_2$, $\\alpha=0.01$  ")
    plot.show()

# ********Plotting the diff to the ground truth *********
#     print np.shape(VI_J2_1)
#     diff = scipy.spatial.distance.cdist([ 3 ,2, 1 ], [0 ,0 ,0], 'sqeuclidean')
#     print diff

    env = Maze(g=1)

    smallest_num = np.nextafter(0, 1)

    policyVI_ground_truth_g1, VI_cost_function_ground_truth_g1, VI_J2_ground_truth_g1 = \
                                                            value_iteration(env, VI_cost_function_ground_truth_g1,
                                                          epsilon=smallest_num,
                                                          alpha=0.99)

    policyPI_ground_truth_g1, PI_cost_function_ground_truth_g1, PI_J2_ground_truth_g1 = \
                                                            policy_iteration(env,  PI_cost_function_ground_truth_g1,
                                                           epsilon=smallest_num,
                                                           alpha=0.99)

    # print 'Ground truth policies equal 1)?\n{}'.format(np.array_equal(policyPI_ground_truth_g1, policyVI_ground_truth_g1))
    # print 'Cost functions equal 1)?\n{}'.format(np.array_equal(VI_cost_function_ground_truth_g1, PI_cost_function_ground_truth_g1))


    plot = visualise(env, VI_cost_function_ground_truth_g1, policyVI_ground_truth_g1)
    plot.title("Value Iteration: $g_2$, $\\alpha=0.99$ - 'ground truth'")
    plot.show()
    # plot.savefig("VI_GT_g2.png")
    plot = visualise(env, PI_cost_function_ground_truth_g1, policyPI_ground_truth_g1)
    plot.title("Policy Iteration: $g_2$, $\\alpha=0.99$ - 'ground truth'")
    plot.show()
    # plot.savefig("PI_GT_g2.png")

    plt.plot(VI_J2_ground_truth_g1)
    plt.title("Squared distance to ground truth VI $g_1$, $\\alpha=0.99$")
    plt.show()
    plt.plot(PI_J2_ground_truth_g1)
    plt.title("Squared distance to ground truth PI $g_1$, $\\alpha=0.99$")
    plt.show()