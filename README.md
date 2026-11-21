# Q-Learning Othello

Angelina Huang's F26 ML Team Lab 7: an Othello agent trained from self-play with Q-learning, using a lookup table on small boards and a linear model on the full 8×8 board. Written in pure Python.

## Code

The game engine, board symmetries, features, training loop and evaluation were provided. I implemented the six learning functions in [othello_q.py](othello_q.py):

- `terminal_reward`: the final result from one player's perspective (+1 win, 0 draw, −1 loss)
- `q_target`: the TD target for a two-player game, including opponent passes
- `epsilon_greedy`: ε-greedy action selection with random tie-breaking
- `TabularQ.update`: the tabular Q-learning update
- `LinearQ.q_value`: the linear Q value, `Q(s, a) = w · φ(s, a)`, with separate weights per game phase
- `LinearQ.update`: the semi-gradient update, `w ← w + α (target − Q) φ`

One agent plays both sides, so Q is always stored from the perspective of the player to move. Because the game is zero-sum, the target is `r − γ · max Q(s', a')` when the opponent moves next and `r + γ · max Q(s', a')` when the opponent has to pass.

## Results

The saved 8×8 model, [q_linear_8.json](q_linear_8.json), was trained for 6,000 games. Over 500 evaluation games, it won **87.6%** against a random player and **91.4%** against a greedy player.

### Tabular Q on 4×4 and 6×6

Both boards were trained for 30,000 games and evaluated over 500 games against random.

| Board | States stored | Evaluation positions seen | Win rate |
|---|---|---|---|
| 4×4 | 4,408 | 99.9% | 88.4% |
| 6×6 | 712,265 | 27.6% | 46.4% |

A table cannot generalize to positions it has not visited, so the 6×6 agent plays no better than random despite storing over 700,000 states.

### Exploration

These 4×4 agents were trained for 30,000 games and evaluated over 200 games per opponent.

| Training policy | States stored | Win rate vs random | Win rate vs greedy |
|---|---|---|---|
| No exploration (ε = 0) | 2,395 | 78.5% | 81.5% |
| Random moves (ε = 1) | 8,644 | 97.5% | 100% |

Q-learning is off-policy: its target uses the best next action rather than the action actually played, so it can learn the greedy policy's values from random play.

### Learned weights

Mid-game weights from the 8×8 model:

| Square type | Example | Weight |
|---|---|---|
| Corner | a1 | +1.71 |
| Edge | c1 | +0.42 |
| Edge | d1 | +0.11 |
| C-square | b1 | −0.32 |
| X-square | b2 | −0.32 |
| Inner squares | c2, d2, c3, d3, d4 | −0.08 to +0.07 |

The agent learned standard Othello strategy from self-play: corners are the most valuable because they can never be flipped, while X-squares and C-squares are risky because they can give the opponent a corner.

### Learning rate experiment

Win rate against greedy, evaluated over 200 games every 600 training games:

| Learning rate | After 600 games | After 6,000 games | Range during training |
|---|---|---|---|
| 0.1 | 92.5% | 93.5% | 80.0–95.5% |
| **0.01** | **61.5%** | **97.0%** | **61.5–98.0%** |
| 0.001 | 59.0% | 64.0% | 54.5–64.5% |

- **Too large (0.1):** learns quickly, but the win rate is unstable because each update changes the weights too much.
- **Best (0.01):** improves steadily and ends with the highest win rate.
- **Too small (0.001):** the weights change so slowly that the agent is still weak after 6,000 games.

### Mobility feature

I removed the opponent-mobility feature and retrained for 6,000 games. Win rates over 500 games:

| Model | vs random | vs greedy |
|---|---|---|
| With mobility | 95.0% | 95.2% |
| Without mobility | 95.0% | 92.4% |

The 2.8-point drop against greedy is within the variation between training runs, so this single experiment does not show that the feature is important.

Raw results are in [logs/](logs), and the written analysis is in [experiments_and_questions.ipynb](experiments_and_questions.ipynb). [othello_q_no_mobility.py](othello_q_no_mobility.py) is the version used for the mobility experiment. The original lab instructions are in [Lab Q-Learning Othello.pdf](<Lab Q-Learning Othello.pdf>).

## Running

Requires Python 3.8+ with no additional packages.

```bash
python3 test_othello_q.py                                                  # run the tests
python3 othello_q.py train --agent linear --size 8 --episodes 6000         # train the 8x8 agent
python3 othello_q.py eval --agent linear --size 8 --load q_linear_8.json   # evaluate the saved model
python3 othello_q.py play --agent linear --size 8 --load q_linear_8.json --human x   # play against it
```
