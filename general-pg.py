import tensorflow as tf
import numpy as np
import gym
from tensorflow.contrib.layers import fully_connected as fclayer

def select_action_from_prob(probs):
    probs = np.asarray(probs)
    cums = np.cumsum(probs)
    return (cums > np.random.rand()).argmax()


def policy_network(state, obs_dim, act_dim):
    with tf.variable_scope("policy_network"):
        fc1 = fclayer(state, num_outputs=10)
        linear = fclayer(fc1, num_outputs=act_dim, activation_fn=None)
        probabilities = tf.nn.softmax(linear)

    return probabilities


def policy_gradient(env):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    with tf.variable_scope("policy"):
        state = tf.placeholder("float",[None,obs_dim])
        actions = tf.placeholder("float",[None,act_dim])
        advantages = tf.placeholder("float",[None,1])

        probabilities = policy_network(state, obs_dim, act_dim)

        good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions),reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return probabilities, state, actions, advantages, optimizer


def value_network(state, obs_dim):
    with tf.variable_scope("value_network"):
        fc1 = fclayer(state, num_outputs=10)
        calculated = fclayer(fc1, num_outputs=1, activation_fn=None)

    return calculated
 
def value_gradient(env):
    obs_dim = env.observation_space.shape[0]

    with tf.variable_scope("value"):
        state = tf.placeholder("float",[None,obs_dim])
        newvals = tf.placeholder("float",[None,1])

        calculated = value_network(state, obs_dim)

        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss

def run_episode(env, policy_grad, value_grad, sess):
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []


    for _ in xrange(200):
        # calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calculated,feed_dict={pl_state: obs_vector})
        action = select_action_from_prob(probs)
        # record the transition
        states.append(observation)
        actionblank = np.zeros(env.action_space.n)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            break
    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in xrange(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.97
        obs_vector = np.expand_dims(obs, axis=0)
        currentval = sess.run(vl_calculated,feed_dict={vl_state: obs_vector})[0][0]

        # advantage: how much better was this action than normal
        advantages.append(future_reward - currentval)

        # update the value function towards new return
        update_vals.append(future_reward)

    # update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

    return totalreward

env_name = 'Acrobot-v0'
env = gym.make(env_name)
env.monitor.start(env_name + '_repo', force=True)
policy_grad = policy_gradient(env)
value_grad = value_gradient(env)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in xrange(2000):
    reward = run_episode(env, policy_grad, value_grad, sess)
    if reward == 200:
        print "reward 200"
        print i
        break
t = 0
for _ in xrange(1000):
    reward = run_episode(env, policy_grad, value_grad, sess)
    t += reward
print t / 1000
env.monitor.close()
