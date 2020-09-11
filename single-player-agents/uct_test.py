import unittest
import gym
import snake
import numpy as np
import uct

class TestUCT(unittest.TestCase):

  def setUp(self):
    self.env = gym.make('snake-v0')
    snake.envs.snake_env.SnakeEnv.M = 4
    snake.envs.snake_env.SnakeEnv.N = 4
    self.env.reset()

  def test_facing_up_at_top_right_after_single_rollout(self):
    self.env.body = [(0,3), (1,3)]
    self.env.fruit = (0,2)

    agent = uct.UCT()
    agent._perform_rollouts(num_rollouts=1, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_visits,
      np.array([1, 0, 0, 0], dtype=np.float32)
    ))
    # print("len(agent.root.children):", len(agent.root.children))
    # print(agent.root.best_action(agent.root_num_visits))

  def test_facing_up_at_top_right(self):
    self.env.body = [(0,3), (1,3)]
    self.env.fruit = (0,2)

    agent = uct.UCT()
    agent._perform_rollouts(num_rollouts=10, env=self.env)
    
    # agent.root.children should only contain left.
    self.assertTrue(1 in agent.root.children)
    self.assertEqual(len(agent.root.children), 1)

    # root_num_visits should be 1 + num_rollouts
    self.assertEqual(agent.root_num_visits, 11)
    # up, down, and right should all result in a loss.
    # this means we should visit left 7 out of 10 times.
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_visits,
      np.array([1, 7, 1, 1], dtype=np.float32)
    ))
    # up, down, and right should all result in a LOSING_VALUE
    self.assertIsNone(np.testing.assert_array_equal(
      np.delete(agent.root.action_total_values, 1),
      np.array([uct.UCTNode.LOSING_VALUE, uct.UCTNode.LOSING_VALUE, uct.UCTNode.LOSING_VALUE], dtype=np.float32)
    ))

    got_action = agent._select_action()
    self.assertEqual(got_action, 1)
    

  def test_best_action(self):
    # test UCTNode initialization
    node = uct.UCTNode(num_actions=2)
    expected_action_priors = np.array([1,1]) / 2
    self.assertIsNone(np.testing.assert_array_equal(node.action_priors, expected_action_priors))

    # test UCTNode.best_action()
    node.child_total_value = np.array([1, 10])
    node.child_num_visits = np.array([10, 1])
    got_best_action = node.best_action(current_num_visits=11)
    self.assertEqual(got_best_action, 0)

  def test_repeated_single_rollout(self):
    self.env.body = [(3,0), (2,0)]
    self.env.fruit = (3,3)
    # obscenely long access to a variable
    wall_reward = uct.UCTNode.LOSING_VALUE

    # perform one rollout
    # perform_rollouts should update action prior of left action to 0
    agent = uct.UCT(num_actions=4)
    agent.root.action_priors = np.array([0, 1, 0, 0], dtype=np.float32)
    agent._perform_rollouts(num_rollouts=1, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_priors,
      np.array([0, 1, 0, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_visits,
      np.array([0, 1, 0, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_total_values,
      np.array([0, wall_reward, 0, 0], dtype=np.float32)
    ))
      
    # perform a second rollout
    agent.root.action_priors = np.array([1, 0, 0, 0], dtype=np.float32)
    agent._perform_rollouts(num_rollouts=1, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_priors,
      np.array([1, 0, 0, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_visits,
      np.array([1, 1, 0, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_total_values,
      np.array([wall_reward, wall_reward, 0, 0], dtype=np.float32)
    ))
      
    # perform a third rollout
    agent.root.action_priors = np.array([0, 0, 1, 0], dtype=np.float32)
    agent._perform_rollouts(num_rollouts=1, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_priors,
      np.array([0, 0, 1, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_visits,
      np.array([1, 1, 1, 0], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_total_values,
      np.array([wall_reward, wall_reward, wall_reward, 0], dtype=np.float32)
    ))
      
    # perform a fourth rollout
    agent.root.action_priors = np.array([0, 0, 0, 1], dtype=np.float32)
    agent._perform_rollouts(num_rollouts=1, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_priors,
      np.array([0, 0, 0, 1], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_visits,
      np.array([1, 1, 1, 1], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      np.delete(agent.root.action_total_values, 3),
      np.ones(3, dtype=np.float32) * wall_reward
    ))
    # the action_total_value for this action could be 0 or wall_reward

    # perform more rollouts
    additional_rollouts = 100
    agent._perform_rollouts(num_rollouts=additional_rollouts, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_priors,
      np.array([0, 0, 0, 1], dtype=np.float32)
    ))
    self.assertIsNone(np.testing.assert_array_equal(
      np.delete(agent.root.action_total_values, 3),
      np.ones(3, dtype=np.float32) * wall_reward
    ))
    self.assertEqual(agent.root_num_visits, 5 + additional_rollouts)
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_visits,
      np.array([1, 1, 1, 1 + additional_rollouts], dtype=np.float32)
    ))

  def test_no_options(self):
    """ Tests a scenario in which all of the agent's options are losing
    """
    self.env.body = [(3,3), (2,3), (2,2), (3,2), (3,1)]
    self.env.fruit = (0,0)

    agent = uct.UCT(num_actions=4)
    agent._perform_rollouts(num_rollouts=100, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      agent.root.action_total_values,
      np.ones(4, dtype=np.float32) * uct.UCTNode.LOSING_VALUE
    ))

  def test_greedy_losing(self):
    """ Tests a scenario in which the agent should not take a greedy action that would result in a loss
    """
    self.env.body = [(2,3), (2,2), (3,2), (3,1)]
    self.env.fruit = (3,3)

    agent = uct.UCT(num_actions=4)
    # perform a large number of rollouts
    # down should show that it'll lead to a losing state
    num_rollouts = 1000
    agent._perform_rollouts(num_rollouts=num_rollouts, env=self.env)
    self.assertIsNone(np.testing.assert_array_equal(
      np.delete(agent.root.action_total_values, 0),
      np.ones(3, dtype=np.float32) * uct.UCTNode.LOSING_VALUE
    ))

  def test_adjust_action(self):
    self.env.body = [(2,3), (2,2), (2,1), (3,1), (3,0)]
    self.env.fruit = (3,3)

    agent = uct.UCT()
    # perform a large number of rollouts
    # down should show that it'll lead to a losing state
    num_rollouts = 1000
    agent._perform_rollouts(num_rollouts=num_rollouts, env=self.env)
    #print(uct.root.children)
    #print(uct.root.children[2].children[1].action_total_values)
    #print(uct.root.children[2].children[1].action_visits)
    #print(uct.root.action_total_values)
    #print(uct.root.action_visits)


# If you want to run this test, make sure to comment out the bottom of uct.py,
# because that will play a game when we import the file.
if __name__ == '__main__':
  unittest.main()