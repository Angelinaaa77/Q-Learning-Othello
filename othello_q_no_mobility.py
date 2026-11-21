#!/usr/bin/env python3
"""
Q-learning Othello: tabular and linear agents trained from self-play.
Pure Python 3, no dependencies.

Two agents, same Q-learning update, different Q representation:
  tabular : Q stored in a dict keyed by canonical (symmetry-reduced) state.
            Converges on 4x4, learns something on 6x6, hopeless on 8x8.
  linear  : Q(s,a) = w . phi(s,a), phi = features of the board after move a.
            Works on 8x8.

Zero-sum self-play formulation (negamax Q):
  Q(s,a) is always from the perspective of the player to move in s.
  target = final result                if the game ended  (+1 win, 0 draw, -1 loss)
         = -gamma * max_a' Q(s',a')    if the opponent moves next
         = +gamma * max_a' Q(s',a')    if the opponent had to pass (same player again)
  Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))

Usage:
  python othello_q.py train --agent tabular --size 4 --episodes 50000
  python othello_q.py train --agent linear  --size 8 --episodes 20000
  python othello_q.py eval  --agent linear  --size 8 --load q_linear_8.json
  python othello_q.py play  --agent linear  --size 8 --load q_linear_8.json --human x
"""
import argparse
import json
import pickle
import random
import time

DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
X, O = 1, -1                       # x = black, moves first
SYM = {1: 'x', -1: 'o', 0: '.'}


# ---------------------------------------------------------------- game engine
class Othello:
    def __init__(self, n=8):
        assert n % 2 == 0 and n >= 4
        self.n, self.nn = n, n * n
        self.rays = []
        self.nbrs = []
        for i in range(self.nn):
            r, c = divmod(i, n)
            rays, nb = [], []
            for dr, dc in DIRS:
                ray, rr, cc = [], r + dr, c + dc
                while 0 <= rr < n and 0 <= cc < n:
                    ray.append(rr * n + cc)
                    rr += dr
                    cc += dc
                if ray:
                    nb.append(ray[0])
                if len(ray) >= 2:
                    rays.append(ray)
            self.rays.append(rays)
            self.nbrs.append(nb)
        self.syms = self._symmetries()                       # new[j] = old[perm[j]]
        self.sym_inv = [self._inverse(p) for p in self.syms]  # old index -> new index
        # squares grouped by D4 orbit (corner, X-square, C-square, edge, ...)
        seen, self.classes = set(), []
        for i in range(self.nn):
            if i not in seen:
                orbit = sorted({p[i] for p in self.syms})
                seen.update(orbit)
                self.classes.append(orbit)

    def _symmetries(self):
        n = self.n
        def idx(r, c): return r * n + c
        maps = [lambda r, c: (r, c), lambda r, c: (c, n - 1 - r),
                lambda r, c: (n - 1 - r, n - 1 - c), lambda r, c: (n - 1 - c, r),
                lambda r, c: (r, n - 1 - c), lambda r, c: (n - 1 - r, c),
                lambda r, c: (c, r), lambda r, c: (n - 1 - c, n - 1 - r)]
        return [[idx(*f(*divmod(j, n))) for j in range(self.nn)] for f in maps]

    @staticmethod
    def _inverse(perm):
        inv = [0] * len(perm)
        for j, i in enumerate(perm):
            inv[i] = j
        return inv

    def initial(self):
        n, h = self.n, self.n // 2
        b = [0] * self.nn
        b[(h - 1) * n + h - 1] = b[h * n + h] = O
        b[(h - 1) * n + h] = b[h * n + h - 1] = X
        return tuple(b)

    def _flips(self, b, p, i):
        out = []
        for ray in self.rays[i]:
            run = []
            for j in ray:
                v = b[j]
                if v == -p:
                    run.append(j)
                elif v == p:
                    if run:
                        out.extend(run)
                    break
                else:
                    break
        return out

    def _is_legal(self, b, p, i):
        for ray in self.rays[i]:
            if b[ray[0]] != -p:
                continue
            for j in ray[1:]:
                v = b[j]
                if v == p:
                    return True
                if v == 0:
                    break
        return False

    def legal(self, b, p):
        return [i for i in range(self.nn) if b[i] == 0 and self._is_legal(b, p, i)]

    def play(self, b, p, i):
        nb = list(b)
        nb[i] = p
        for j in self._flips(b, p, i):
            nb[j] = p
        return tuple(nb)

    def step(self, b, p, a):
        """Returns (board', player_to_move or None if game over, legal moves)."""
        b2 = self.play(b, p, a)
        m = self.legal(b2, -p)
        if m:
            return b2, -p, m
        m = self.legal(b2, p)
        if m:
            return b2, p, m        # opponent passes
        return b2, None, []

    def show(self, b, moves=()):
        n = self.n
        cols = 'abcdefgh'[:n]
        rows = ['  ' + ' '.join(cols)]
        for r in range(n):
            cells = []
            for c in range(n):
                i = r * n + c
                cells.append('*' if i in moves else SYM[b[i]])
            rows.append(f'{r + 1} ' + ' '.join(cells))
        rows.append(f'x={b.count(X)} o={b.count(O)}')
        return '\n'.join(rows)


def sign(v):
    return (v > 0) - (v < 0)


# ---------------------------------------------------------------- agents
class TabularQ:
    """Q[canonical_state][canonical_action]. Canonical = mover-relative board,
    lexicographically smallest over the 8 board symmetries."""

    def __init__(self, game, alpha=0.2):
        self.g, self.alpha, self.Q = game, alpha, {}

    def _canon(self, b, p):
        """Returns (canonical key, list of symmetries that map b to it).
        More than one symmetry means the position is self-symmetric, and moves that
        are images of each other must share one Q entry."""
        v = tuple(x * p for x in b)
        best, ks = None, []
        for k, perm in enumerate(self.g.syms):
            t = tuple(v[j] for j in perm)
            if best is None or t < best:
                best, ks = t, [k]
            elif t == best:
                ks.append(k)
        return best, ks

    def evaluate(self, b, p, moves):
        """Returns list of (Q value, handle used by update)."""
        key, ks = self._canon(b, p)
        row = self.Q.get(key, {})
        invs = [self.g.sym_inv[k] for k in ks]
        out = []
        for a in moves:
            ca = min(inv[a] for inv in invs)
            out.append((row.get(ca, 0.0), (key, ca)))
        return out

    def update(self, handle, q, target):
        """tabular Q-learning update.
        handle = (key, ca): canonical state key and canonical action index.
        q      = current Q(s,a) (what evaluate returned for this action).
        target = TD target from q_target.
        Store the new value in self.Q[key][ca]. self.Q[key] may not exist yet.
        Learning rate is self.alpha."""
        # Unpack which state and action this update belongs to.
        state_key, action = handle

        # First visit to this state: create its table of action values.
        if state_key not in self.Q:
            self.Q[state_key] = {}

        # TD error: how far the current estimate is from the target.
        td_error = target - q

        # Move the estimate a fraction (alpha) of the way toward the target.
        self.Q[state_key][action] = q + self.alpha * td_error

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'n': self.g.n, 'Q': self.Q}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.Q = pickle.load(f)['Q']

    def stats(self):
        return f'states={len(self.Q)}'


class LinearQ:
    """Q(s,a) = w_phase . phi(after-board from mover's view).
    Features: mean disc ownership per symmetry class (corners, X, C, edges...),
    disc diff, frontier diff, opponent mobility, bias. Separate weights per game phase."""

    PHASES = 3

    def __init__(self, game, alpha=0.01):
        self.g, self.alpha = game, alpha
        self.nf = len(game.classes) + 3
        self.w = [[0.0] * self.nf for _ in range(self.PHASES)]

    def features(self, after, p):
        g = self.g
        v = [x * p for x in after]
        f = [sum(v[i] for i in cls) / len(cls) for cls in g.classes]
        occupied = g.nn - v.count(0)
        f.append(sum(v) / occupied)
        fr = 0
        for i in range(g.nn):
            if v[i] and any(v[j] == 0 for j in g.nbrs[i]):
                fr += v[i]
        f.append(-fr / 10.0)
        f.append(1.0)
        empties = g.nn - occupied
        phase = min(self.PHASES - 1, (g.nn - empties) * self.PHASES // g.nn)
        return phase, f

    def evaluate(self, b, p, moves):
        out = []
        for a in moves:
            phase, f = self.features(self.g.play(b, p, a), p)
            out.append((self.q_value(phase, f), (phase, f)))
        return out

    def q_value(self, phase, f):
        """linear Q value.
        phase: int in [0, PHASES). f: list of floats (feature vector phi(s,a)).
        Return Q(s,a) = w[phase] . f   (self.w is a list of weight lists)."""
        # Compute the dot product of the weights for the current phase and the feature vector.
        return sum(w_i * f_i for w_i, f_i in zip(self.w[phase], f))

    def update(self, handle, q, target):
        """semi-gradient Q-learning update for a linear Q.
        handle = (phase, f). Only the weights for this phase change.
        Hint: what is dQ/dw[phase][i]?"""
        phase, f = handle
        td_error = target - q

        # For a linear Q, the gradient dQ/dw[i] is simply f[i].
        for i in range(self.nf):
            self.w[phase][i] += self.alpha * td_error * f[i]

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'n': self.g.n, 'w': self.w}, f, indent=1)

    def load(self, path):
        with open(path) as f:
            self.w = json.load(f)['w']

    def stats(self):
        return 'w_mid=' + ' '.join(f'{x:+.2f}' for x in self.w[1])


def terminal_reward(board, p):
    """final result from player p's point of view: +1.0 win, 0.0 draw, -1.0 loss.
    board is a tuple of +1 (x) / -1 (o) / 0 (empty). p is +1 or -1. Return a float."""
    # Count each player's discs; empty squares match neither side.
    my_discs = sum(cell == p for cell in board)
    opponent_discs = sum(cell == -p for cell in board)

    if my_discs > opponent_discs:
        return 1.0
    elif my_discs < opponent_discs:
        return -1.0
    else:
        return 0.0


def q_target(reward, next_qs, same_player, gamma):
    """TD target for Q(s,a), from the point of view of the player who chose a.
    reward:      immediate reward to that player (the final result if s' is terminal, else 0).
    next_qs:     list of Q(s',a') values, from the point of view of whoever moves in s',
                 or None if s' is terminal.
    same_player: True if the player to move in s' is the one who chose a (opponent passed).
    gamma:       discount factor.
    Remember: Q values are always from the mover's perspective, and the game is zero-sum."""
    # Terminal state: no future moves, so the target is just the final result.
    if next_qs is None:
        return reward
    # Opponent passed: we move again, so next_qs is already from our perspective.
    elif same_player:
        return reward + gamma * max(next_qs)
    # Opponent moves next: their best value is our loss in a zero-sum game.
    else:
        return reward - gamma * max(next_qs)




def epsilon_greedy(evals, eps):
    """epsilon-greedy action selection.
    evals: list of (q_value, handle) pairs, one per legal move. Return an INDEX into evals.
    With probability eps pick uniformly at random; otherwise pick a highest-Q index,
    breaking ties uniformly at random (ties are common early on when everything is 0)."""
    if random.random() < eps:
        return random.randrange(len(evals))

    # Exploit: find the highest Q value.
    q_values = [q_value for q_value, _handle in evals]
    best_q = max(q_values)

    # Collect every move tied for the best value.
    best_indices = []
    for index, q_value in enumerate(q_values):
        if q_value == best_q:
            best_indices.append(index)

    # Break ties at random.
    return random.choice(best_indices)
    


# ---------------------------------------------------------------- training
def train(game, agent, episodes, eps_start, eps_end, gamma=1.0, eval_every=0, log=True):
    t0 = time.time()
    for ep in range(1, episodes + 1):
        eps = eps_start + (eps_end - eps_start) * ep / episodes
        b, p = game.initial(), X
        moves = game.legal(b, p)
        evals = agent.evaluate(b, p, moves)
        while True:
            k = epsilon_greedy(evals, eps)
            q, handle = evals[k]
            b2, p2, moves2 = game.step(b, p, moves[k])
            if p2 is None:
                evals2 = None
                target = q_target(terminal_reward(b2, p), None, False, gamma)
            else:
                evals2 = agent.evaluate(b2, p2, moves2)
                target = q_target(0.0, [v for v, _ in evals2], p2 == p, gamma)
            agent.update(handle, q, target)
            if p2 is None:
                break
            # Q values for s' were computed before the update; for tabular
            # s' != s so they are still exact, for linear they are slightly stale.
            b, p, moves, evals = b2, p2, moves2, evals2
        if log and eval_every and ep % eval_every == 0:
            r = evaluate(game, agent, random_player, 200)
            gr = evaluate(game, agent, greedy_player, 200)
            print(f'ep {ep:>7}  eps {eps:.3f}  vs random W/D/L {r}  vs greedy {gr}  '
                  f'{agent.stats()}  {time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- opponents and evaluation
def random_player(game, b, p, moves):
    return random.choice(moves)


def greedy_player(game, b, p, moves):
    """Maximize discs flipped this move."""
    return max(moves, key=lambda a: (len(game._flips(b, p, a)), random.random()))


def agent_player(agent):
    def f(game, b, p, moves):
        evals = agent.evaluate(b, p, moves)
        return moves[epsilon_greedy(evals, 0.0)]
    return f


def play_game(game, px, po):
    b, p = game.initial(), X
    moves = game.legal(b, p)
    while p is not None:
        a = (px if p == X else po)(game, b, p, moves)
        b, p, moves = game.step(b, p, a)
    return sign(sum(b))


def evaluate(game, agent, opponent, games):
    me = agent_player(agent)
    w = d = l = 0
    for g in range(games):
        if g % 2 == 0:
            r = play_game(game, me, opponent)
        else:
            r = -play_game(game, opponent, me)
        w += r > 0
        d += r == 0
        l += r < 0
    return f'{w}/{d}/{l}'


def human_player(game, b, p, moves):
    print(game.show(b, moves))
    while True:
        s = input(f'{SYM[p]} to move (e.g. c4): ').strip().lower()
        try:
            i = (int(s[1:]) - 1) * game.n + 'abcdefgh'.index(s[0])
            if i in moves:
                return i
        except (ValueError, IndexError):
            pass
        print('illegal')


# ---------------------------------------------------------------- string-board interface
def best_move_from_string(agent, board_str, token):
    """board_str: n*n chars of '.xo' (row-major, like the lab format). token: 'x' or 'o'.
    Returns the index of the agent's greedy move, or -1 if none."""
    b = tuple({'x': X, 'o': O}.get(ch, 0) for ch in board_str.lower())
    p = X if token.lower() == 'x' else O
    moves = agent.g.legal(b, p)
    return agent_player(agent)(agent.g, b, p, moves) if moves else -1


class Strategy:
    """Drop-in for the lab server: board is a 64-char string, player is 'x'/'o'."""
    logging = True
    WEIGHTS = 'q_linear_8.json'

    def __init__(self):
        self.agent = LinearQ(Othello(8))
        self.agent.load(self.WEIGHTS)

    def best_strategy(self, board, player, best_move, running):
        if running.value:
            best_move.value = best_move_from_string(self.agent, board, player)


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'eval', 'play'])
    ap.add_argument('--agent', choices=['tabular', 'linear'], default='linear')
    ap.add_argument('--size', type=int, default=8)
    ap.add_argument('--episodes', type=int, default=20000)
    ap.add_argument('--alpha', type=float)
    ap.add_argument('--eps-start', type=float, default=0.3)
    ap.add_argument('--eps-end', type=float, default=0.02)
    ap.add_argument('--eval-every', type=int, default=0)
    ap.add_argument('--games', type=int, default=500)
    ap.add_argument('--load')
    ap.add_argument('--save')
    ap.add_argument('--human', choices=['x', 'o'], default='x')
    ap.add_argument('--seed', type=int)
    a = ap.parse_args()
    if a.seed is not None:
        random.seed(a.seed)

    game = Othello(a.size)
    if a.agent == 'tabular':
        agent = TabularQ(game, a.alpha or 0.2)
        ext = 'pkl'
    else:
        agent = LinearQ(game, a.alpha or 0.01)
        ext = 'json'
    if a.load:
        agent.load(a.load)

    if a.cmd == 'train':
        ee = a.eval_every or max(1, a.episodes // 10)
        train(game, agent, a.episodes, a.eps_start, a.eps_end, eval_every=ee)
        path = a.save or f'q_{a.agent}_{a.size}.{ext}'
        agent.save(path)
        print('saved', path)
    elif a.cmd == 'eval':
        print('vs random W/D/L', evaluate(game, agent, random_player, a.games))
        print('vs greedy W/D/L', evaluate(game, agent, greedy_player, a.games))
    else:
        me = agent_player(agent)
        hp = human_player
        r = play_game(game, hp, me) if a.human == 'x' else play_game(game, me, hp)
        print({1: 'x wins', -1: 'o wins', 0: 'draw'}[r])


if __name__ == '__main__':
    main()
