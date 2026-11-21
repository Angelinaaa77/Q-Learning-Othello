# Q-Learning Othello

Angelina Huang's F26 ML Team Lab 7: an Othello agent trained from self-play with Q-learning, using a lookup table on small boards and a linear function approximator on the full 8×8 board.

## What I implemented

The game engine, board symmetries, features, training loop and evaluation were provided. I wrote the six functions that make up the learning algorithm in [othello_q.py](othello_q.py):

- `terminal_reward`: the final result from one player's point of view (+1 win, 0 draw, −1 loss)
- `q_target`: the two-player TD target, including the case where the opponent has to pass
- `epsilon_greedy`: action selection with random tie-breaking
- `TabularQ.update`: the tabular Q-learning update
- `LinearQ.q_value`: `Q(s, a) = w · φ(s, a)`, with separate weights for each game phase
- `LinearQ.update`: the semi-gradient update, `w ← w + α (target − Q) φ`

## The two-player target

One agent plays both sides, so Q is always stored from the point of view of the player about to move. The game is zero-sum, which means the opponent's best outcome is our loss:

| After our move | Target |
|---|---|
| Game over | `r` |
| Opponent moves | `r − γ · max_a' Q(s', a')` |
| Opponent must pass | `r + γ · max_a' Q(s', a')` |

## Findings

- **Why a table stops working:** on 4×4 the table stores 4,408 states, has seen 99.9% of evaluation positions and wins 88.4% against random. On 6×6 it stores 712,265 states but has seen only 27.6% of them and wins 46.4%, no better than random play.
- **Off-policy learning:** on 4×4, training with no exploration reaches 78.5% against random, while training on purely random moves reaches 97.5%. Q-learning can behave randomly and still learn the value of the best move.
- **Learned strategy:** without being told any Othello strategy, the 8×8 agent gives corners the highest mid-game weight (+1.71) and gives X-squares and C-squares, which can hand the opponent a corner, negative weights (−0.32).
- **Learning rate:** against a greedy opponent, α = 0.1 learns fastest but swings between 80% and 95.5%, α = 0.001 is still at 64% after 6,000 games, and α = 0.01 climbs from 61.5% to 97%.
- **Mobility feature:** removing opponent mobility lowered the win rate against greedy from 95.2% to 92.4% over 500 games. This gap is within the variation between training runs, so one run does not prove the feature matters.

The saved model, [q_linear_8.json](q_linear_8.json), won 87.6% against random and 91.4% against greedy over 500 games. Raw results are in [logs/](logs), and [experiments_and_questions.ipynb](experiments_and_questions.ipynb) contains the written analysis. [othello_q_no_mobility.py](othello_q_no_mobility.py) is the copy used for the mobility experiment.

## Running

Requires Python 3.8+ and no other packages.

```bash
# Run the tests
python3 test_othello_q.py

# Train
python3 othello_q.py train --agent tabular --size 4 --episodes 30000
python3 othello_q.py train --agent linear --size 8 --episodes 6000

# Evaluate the saved 8x8 agent
python3 othello_q.py eval --agent linear --size 8 --load q_linear_8.json

# Play against it as x
python3 othello_q.py play --agent linear --size 8 --load q_linear_8.json --human x
```
