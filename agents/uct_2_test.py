import unittest
import gym
import snake
import numpy as np
import uct_2

class TestUCT2(unittest.TestCase):

  def setUp(self):
    self.env = gym.make('snake-v0')
    self.env.M = 4
    self.env.N = 4

  def test_facing_up_at_top(self):
    self.env.body = [(0,3), (1,3)]
    self.env.fruit = (0,2)

    uct = uct_2.UCT()
    got_action = uct.action(1000, self.env)
    self.assertEqual(got_action, 1)

  def test_best_action(self):
    node = uct_2.UCTNode(num_actions=2)
    node.child_total_value = np.array([1, 10])
    node.child_num_visits = np.array([10, 1])
    got_best_action = node.best_action(current_num_visits=12)
    self.assertEqual(got_best_action, 1)


if __name__ == '__main__':
  unittest.main()