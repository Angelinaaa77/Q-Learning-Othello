#!/usr/bin/env python3
"""
Tests for the Q-learning Othello lab.

Run all:            python test_othello_q.py
Run one TODO:       python test_othello_q.py TestTODO2_QTarget
Verbose:            python test_othello_q.py -v
Skip slow tests:    FAST=1 python test_othello_q.py

By default this tests othello_q.py. To test another file:
    OTHELLO_MODULE=othello_q_solution python test_othello_q.py
"""
import importlib
import os
import random
import unittest

M = importlib.import_module(os.environ.get('OTHELLO_MODULE', 'othello_q'))
FAST = os.environ.get('FAST') == '1'
X, O = M.X, M.O


def board_from_string(s):
    return tuple({'x': X, 'o': O}.get(ch, 0) for ch in s)


# ====================================================================== provided engine
class TestEngine(unittest.TestCase):
    """Sanity checks on the provided code. These should pass before you write anything."""

    def test_initial_moves_8x8(self):
        g = M.Othello(8)
        self.assertEqual(sorted(g.legal(g.initial(), X)), [19, 26, 37, 44])

    def test_initial_matches_lab_format(self):
        g = M.Othello(8)
        self.assertEqual(g.initial(), board_from_string('.' * 27 + 'ox......xo' + '.' * 27))

    def test_play_flips(self):
        g = M.Othello(8)
        b = g.play(g.initial(), X, 19)
        self.assertEqual(b[19], X)
        self.assertEqual(b[27], X)
        self.assertEqual(sum(1 for v in b if v == X), 4)

    def test_eight_symmetries(self):
        for n in (4, 6, 8):
            g = M.Othello(n)
            self.assertEqual(len({tuple(p) for p in g.syms}), 8)

    def test_symmetry_classes_8x8(self):
        self.assertEqual(len(M.Othello(8).classes), 10)


# ====================================================================== TODO 1
class TestTODO1_TerminalReward(unittest.TestCase):

    def test_win_loss(self):
        b = (X, X, X, O) + (0,) * 12
        self.assertEqual(M.terminal_reward(b, X), 1.0)
        self.assertEqual(M.terminal_reward(b, O), -1.0)

    def test_draw(self):
        b = (X, O, X, O) + (0,) * 12
        self.assertEqual(M.terminal_reward(b, X), 0.0)
        self.assertEqual(M.terminal_reward(b, O), 0.0)

    def test_sign_not_margin(self):
        b = (X,) * 15 + (O,)
        self.assertEqual(M.terminal_reward(b, X), 1.0)

    def test_returns_float(self):
        self.assertIsInstance(M.terminal_reward((X, O, O, O), X), float)


# ====================================================================== TODO 2
class TestTODO2_QTarget(unittest.TestCase):

    def test_terminal_returns_reward(self):
        self.assertEqual(M.q_target(1.0, None, False, 0.9), 1.0)
        self.assertEqual(M.q_target(-1.0, None, False, 0.9), -1.0)
        self.assertEqual(M.q_target(0.0, None, True, 0.9), 0.0)

    def test_opponent_moves_next_negates(self):
        # Opponent's best option is worth +0.5 to them, so -0.5 to us.
        self.assertAlmostEqual(M.q_target(0.0, [0.1, 0.5, -0.3], False, 1.0), -0.5)

    def test_opponent_passes_no_negation(self):
        # Opponent passed, so we move again: our own best option is ours.
        self.assertAlmostEqual(M.q_target(0.0, [0.1, 0.5, -0.3], True, 1.0), 0.5)

    def test_uses_max_not_min(self):
        # Opponent picks what is best for THEM (max of their Q), not worst for us.
        self.assertAlmostEqual(M.q_target(0.0, [-0.8, -0.2], False, 1.0), 0.2)

    def test_gamma(self):
        self.assertAlmostEqual(M.q_target(0.0, [0.4], False, 0.5), -0.2)
        self.assertAlmostEqual(M.q_target(0.0, [0.4], True, 0.5), 0.2)

    def test_reward_added(self):
        self.assertAlmostEqual(M.q_target(0.25, [0.4], False, 1.0), -0.15)


# ====================================================================== TODO 3
class TestTODO3_EpsilonGreedy(unittest.TestCase):

    def setUp(self):
        random.seed(0)

    def test_greedy_picks_argmax(self):
        evals = [(0.1, 'a'), (0.9, 'b'), (-0.5, 'c'), (0.3, 'd')]
        for _ in range(50):
            self.assertEqual(M.epsilon_greedy(evals, 0.0), 1)

    def test_returns_index(self):
        i = M.epsilon_greedy([(0.0, 'a'), (1.0, 'b')], 0.0)
        self.assertIsInstance(i, int)

    def test_ties_broken_randomly(self):
        evals = [(0.0, None)] * 4
        seen = {M.epsilon_greedy(evals, 0.0) for _ in range(200)}
        self.assertEqual(seen, {0, 1, 2, 3})

    def test_ties_only_among_best(self):
        evals = [(0.5, None), (0.1, None), (0.5, None)]
        seen = {M.epsilon_greedy(evals, 0.0) for _ in range(200)}
        self.assertEqual(seen, {0, 2})

    def test_eps_one_is_uniform(self):
        evals = [(10.0, None), (0.0, None), (0.0, None), (0.0, None)]
        counts = [0] * 4
        for _ in range(4000):
            counts[M.epsilon_greedy(evals, 1.0)] += 1
        for c in counts:
            self.assertTrue(800 < c < 1200, counts)

    def test_eps_rate(self):
        # eps=0.2 with 4 actions: best chosen 0.8 + 0.2/4 = 0.85 of the time
        evals = [(1.0, None), (0.0, None), (0.0, None), (0.0, None)]
        n = 10000
        hits = sum(M.epsilon_greedy(evals, 0.2) == 0 for _ in range(n))
        self.assertAlmostEqual(hits / n, 0.85, delta=0.02)


# ====================================================================== TODO 4
class TestTODO4_TabularUpdate(unittest.TestCase):

    def setUp(self):
        self.g = M.Othello(4)
        self.agent = M.TabularQ(self.g, alpha=0.5)
        self.b = self.g.initial()
        self.moves = self.g.legal(self.b, X)

    def test_update_from_zero(self):
        q, h = self.agent.evaluate(self.b, X, self.moves)[0]
        self.assertEqual(q, 0.0)
        self.agent.update(h, q, 1.0)
        q2, _ = self.agent.evaluate(self.b, X, self.moves)[0]
        self.assertAlmostEqual(q2, 0.5)

    def test_update_rule(self):
        q, h = self.agent.evaluate(self.b, X, self.moves)[0]
        self.agent.update(h, q, 1.0)          # Q = 0.5
        q, h = self.agent.evaluate(self.b, X, self.moves)[0]
        self.agent.update(h, q, -1.0)         # Q = 0.5 + 0.5 * (-1 - 0.5) = -0.25
        q, _ = self.agent.evaluate(self.b, X, self.moves)[0]
        self.assertAlmostEqual(q, -0.25)

    def test_converges_to_fixed_target(self):
        for _ in range(60):
            q, h = self.agent.evaluate(self.b, X, self.moves)[0]
            self.agent.update(h, q, 0.7)
        q, _ = self.agent.evaluate(self.b, X, self.moves)[0]
        self.assertAlmostEqual(q, 0.7, places=6)

    def test_symmetric_moves_share_entry(self):
        # All 4 opening moves are equivalent under symmetry, so updating one updates all.
        q, h = self.agent.evaluate(self.b, X, self.moves)[0]
        self.agent.update(h, q, 1.0)
        for q, _ in self.agent.evaluate(self.b, X, self.moves):
            self.assertAlmostEqual(q, 0.5)

    def test_does_not_touch_other_states(self):
        q, h = self.agent.evaluate(self.b, X, self.moves)[0]
        self.agent.update(h, q, 1.0)
        b2, p2, m2 = self.g.step(self.b, X, self.moves[0])
        for q, _ in self.agent.evaluate(b2, p2, m2):
            self.assertEqual(q, 0.0)


# ====================================================================== TODO 5
class TestTODO5_LinearQValue(unittest.TestCase):

    def test_dot_product(self):
        a = M.LinearQ(M.Othello(8))
        a.w = [[0.0] * a.nf for _ in range(a.PHASES)]
        a.w[1][0], a.w[1][1], a.w[1][-1] = 2.0, -1.0, 0.5
        f = [0.0] * a.nf
        f[0], f[1], f[-1] = 3.0, 4.0, 1.0
        self.assertAlmostEqual(a.q_value(1, f), 2 * 3 - 4 + 0.5)

    def test_uses_correct_phase(self):
        a = M.LinearQ(M.Othello(8))
        a.w = [[float(k + 1)] * a.nf for k in range(a.PHASES)]
        f = [1.0] * a.nf
        for k in range(a.PHASES):
            self.assertAlmostEqual(a.q_value(k, f), (k + 1) * a.nf)

    def test_zero_weights(self):
        a = M.LinearQ(M.Othello(8))
        g = a.g
        for q, _ in a.evaluate(g.initial(), X, g.legal(g.initial(), X)):
            self.assertEqual(q, 0.0)


# ====================================================================== TODO 6
class TestTODO6_LinearUpdate(unittest.TestCase):

    def setUp(self):
        self.a = M.LinearQ(M.Othello(8), alpha=0.1)
        self.f = [0.5] * self.a.nf
        self.f[0] = -1.0

    def test_gradient_step(self):
        self.a.update((1, self.f), 0.0, 1.0)
        for i, fi in enumerate(self.f):
            self.assertAlmostEqual(self.a.w[1][i], 0.1 * 1.0 * fi)

    def test_only_one_phase_changes(self):
        self.a.update((2, self.f), 0.0, 1.0)
        self.assertEqual(self.a.w[0], [0.0] * self.a.nf)
        self.assertEqual(self.a.w[1], [0.0] * self.a.nf)

    def test_uses_given_q_not_recomputed(self):
        # The caller passes q; the error is (target - q).
        self.a.update((0, self.f), 0.4, 1.0)
        self.assertAlmostEqual(self.a.w[0][1], 0.1 * 0.6 * 0.5)

    def test_reduces_error(self):
        before = abs(1.0 - self.a.q_value(1, self.f))
        for _ in range(5):
            self.a.update((1, self.f), self.a.q_value(1, self.f), 1.0)
        after = abs(1.0 - self.a.q_value(1, self.f))
        self.assertLess(after, before)

    def test_zero_feature_weight_unchanged(self):
        f = [0.0] * self.a.nf
        f[3] = 1.0
        self.a.update((0, f), 0.0, 1.0)
        self.assertEqual(sum(1 for w in self.a.w[0] if w != 0.0), 1)


# ====================================================================== end-to-end learning
@unittest.skipIf(FAST, 'FAST=1')
class TestLearningTabular(unittest.TestCase):
    """Slow (about 10s). Your agent has to actually learn."""

    def test_tabular_4x4_beats_random(self):
        random.seed(1)
        g = M.Othello(4)
        a = M.TabularQ(g, 0.2)
        M.train(g, a, 20000, 0.3, 0.02, log=False)
        w, d, l = map(int, M.evaluate(g, a, M.random_player, 400).split('/'))
        # Reference solution scores 87-94% here across seeds.
        self.assertGreater(w / 400, 0.80, f'W/D/L {w}/{d}/{l}')

    def test_tabular_4x4_learns_terminal_values(self):
        # After training, a move that ends the game must have Q equal to its true result.
        random.seed(2)
        g = M.Othello(4)
        a = M.TabularQ(g, 1.0)   # alpha=1: a terminal target is exact after one visit
        M.train(g, a, 5000, 0.5, 0.2, log=False)
        checked = 0
        for _ in range(200):
            b, p = g.initial(), X
            moves = g.legal(b, p)
            while p is not None:
                for (q, h), m in zip(a.evaluate(b, p, moves), moves):
                    b2, p2, _ = g.step(b, p, m)
                    if p2 is None and h[0] in a.Q and h[1] in a.Q[h[0]]:
                        self.assertAlmostEqual(q, M.terminal_reward(b2, p), places=6)
                        checked += 1
                b, p, moves = g.step(b, p, random.choice(moves))
        self.assertGreater(checked, 20)



@unittest.skipIf(FAST, 'FAST=1')
class TestLearningLinear(unittest.TestCase):
    """Trains one 8x8 linear agent (about 10s) and checks it."""

    @classmethod
    def setUpClass(cls):
        random.seed(3)
        cls.g = M.Othello(8)
        cls.a = M.LinearQ(cls.g, 0.01)
        M.train(cls.g, cls.a, 600, 0.3, 0.05, log=False)

    def test_linear_8x8_beats_random(self):
        # Reference solution scores 70-78% here across seeds; random vs random is about 45%.
        w, d, l = map(int, M.evaluate(self.g, self.a, M.random_player, 200).split('/'))
        self.assertGreater(w / 200, 0.60, f'W/D/L {w}/{d}/{l}')

    def test_linear_learns_corners_good(self):
        corner = next(k for k, cls in enumerate(self.g.classes) if 0 in cls)
        self.assertGreater(self.a.w[1][corner], 0.0, 'corner weight should be positive mid-game')

    def test_weights_finite_and_bounded(self):
        for row in self.a.w:
            for w in row:
                self.assertTrue(abs(w) < 50, f'weight {w} exploded: check the sign of your update')

if __name__ == '__main__':
    unittest.main()
