import unittest
import gym
import snake
import numpy as np
import uct_2

class TestUCT2(unittest.TestCase):

  def setUp(self):
    self.env = gym.make('snake-v0')
    snake.envs.snake_env.SnakeEnv.M = 4
    snake.envs.snake_env.SnakeEnv.N = 4

  def test_facing_up_at_top(self):
    self.env.body = [(0,3), (1,3)]
    self.env.fruit = (0,2)

    uct = uct_2.UCT()
    uct._perform_rollouts(num_rollouts=10, env=self.env)
    got_action = uct._select_action()
    self.assertEqual(got_action, 1)

  def test_best_action(self):
    # test UCTNode initialization
    node = uct_2.UCTNode(num_actions=2)
    expected_action_priors = np.array([1,1])
    self.assertIsNone(np.testing.assert_array_equal(node.action_priors, expected_action_priors))

    # test UCTNode.best_action()
    node.child_total_value = np.array([1, 10])
    node.child_num_visits = np.array([10, 1])
    got_best_action = node.best_action(current_num_visits=11)

  def test_repeated_single_rollout(self):
    self.env.body = [(3,0), (2,0)]
    self.env.fruit = (3,3)
    # obscenely long access to a variable
    wall_reward = uct_2.UCTNode.LOSING_VALUE

    # perform one rollout
    # perform_rollouts should update action prior of left action to 0
    uct = uct_2.UCT()
    uct.root.action_priors = np.array([0, 1, 0, 0, 0], dtype=np.float32)
    uct._perform_rollouts(num_rollouts=1, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_priors,
      np.array([0, 0, 0, 0, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_visits,
      np.array([0, 1, 0, 0, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_total_values,
      np.array([0, wall_reward, 0, 0, 0], dtype=np.float32)
    ))
      
    # perform a second rollout
    uct.root.action_priors = np.array([1, 0, 0, 0, 0], dtype=np.float32)
    uct._perform_rollouts(num_rollouts=1, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_priors,
      np.array([0, 0, 0, 0, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_visits,
      np.array([1, 1, 0, 0, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_total_values,
      np.array([wall_reward, wall_reward, 0, 0, 0], dtype=np.float32)
    ))
      
    # perform a third rollout
    uct.root.action_priors = np.array([0, 0, 1, 0, 0], dtype=np.float32)
    uct._perform_rollouts(num_rollouts=1, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_priors,
      np.array([0, 0, 0, 0, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_visits,
      np.array([1, 1, 1, 0, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_total_values,
      np.array([wall_reward, wall_reward, wall_reward, 0, 0], dtype=np.float32)
    ))
      
    # perform a fourth rollout
    uct.root.action_priors = np.array([0, 0, 0, 0, 1], dtype=np.float32)
    uct._perform_rollouts(num_rollouts=1, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_priors,
      np.array([0, 0, 0, 0, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_visits,
      np.array([1, 1, 1, 0, 1], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_total_values,
      np.array([wall_reward, wall_reward, wall_reward, 0, wall_reward], dtype=np.float32)
    ))
      
    # perform a fifth rollout
    uct.root.action_priors = np.array([0, 0, 0, 1, 0], dtype=np.float32)
    uct._perform_rollouts(num_rollouts=1, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_priors,
      np.array([0, 0, 0, 1, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_visits,
      np.array([1, 1, 1, 1, 1], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      np.delete(uct.root.action_total_values, 3),
      np.ones(4, dtype=np.float32) * wall_reward
    ))
    # the action_total_value for this action could be 0 or wall_reward

    # perform more rollouts
    additional_rollouts = 100
    uct._perform_rollouts(num_rollouts=additional_rollouts, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_priors,
      np.array([0, 0, 0, 1, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      np.delete(uct.root.action_total_values, 3),
      np.ones(4, dtype=np.float32) * wall_reward
    ))
    self.assertEqual(uct.root_num_visits, 6 + additional_rollouts)
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_visits,
      np.array([1, 1, 1, 1 + additional_rollouts, 1], dtype=np.float32)
    ))

  def test_no_options(self):
    """ Tests a scenario in which all of the agent's options are losing
    """
    self.env.body = [(3,3), (2,3), (2,2), (3,2)]
    self.env.fruit = (0,0)

    uct = uct_2.UCT()
    uct._perform_rollouts(num_rollouts=100, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      uct.root.action_total_values,
      np.ones(5, dtype=np.float32) * uct_2.UCTNode.LOSING_VALUE
    ))

  def test_greedy_losing(self):
    """ Tests a scenario in which the agent should not
      take a greedy action that would result in a loss
    """
    self.env.body = [(2,3), (2,2), (3,2)]
    self.env.fruit = (3,3)

    uct = uct_2.UCT()
    # perform a large number of rollouts
    # down should show that it'll lead to a losing state
    num_rollouts = 1000
    uct._perform_rollouts(num_rollouts=num_rollouts, env=self.env)
    #print(uct.root.children[2].action_total_values)
    #print(uct.root.children[2].action_visits)

    #print(uct.root.children[2].children[1].action_total_values)
    #print(uct.root.children[2].children[1].action_visits)
    #TODO actually write a test


  def test_adjust_action(self):
    self.env.body = [(2,3), (2,2), (2,1), (3,1), (3,0)]
    self.env.fruit = (3,3)

    uct = uct_2.UCT()
    # perform a large number of rollouts
    # down should show that it'll lead to a losing state
    num_rollouts = 1000
    uct._perform_rollouts(num_rollouts=num_rollouts, env=self.env)
    #print(uct.root.children)
    #print(uct.root.children[2].children[1].action_total_values)
    #print(uct.root.children[2].children[1].action_visits)
    #print(uct.root.action_total_values)
    #print(uct.root.action_visits)


if __name__ == '__main__':
  unittest.main()