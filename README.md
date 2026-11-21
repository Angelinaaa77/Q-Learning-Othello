# Q-Learning Othello

I taught a program to play Othello by having it play against itself. Nobody told it any strategy. It learned from wins and losses alone.

The final agent beats a random player **87.6%** of the time and a simple greedy player **91.4%** of the time.

This was Lab 7 for the F26 ML Team. The game engine and training loop were provided, and I wrote the learning algorithm.

## What I built

- **Scoring a game:** decides whether a finished game was a win, draw, or loss.
- **Learning from each move:** updates how good a move looks based on what happened next.
- **Thinking about the opponent:** assumes the opponent will make their best move, because their gain is our loss.
- **Choosing moves:** usually plays the best known move, but sometimes tries a random one to keep exploring.
- **Two kinds of memory:** a lookup table for small boards, and a simple formula that works on the full 8x8 board.

## What I found

**A lookup table only works on small boards.** On 4x4, the agent had seen almost every position and won 88.4% of games. On 6x6, it stored over 700,000 positions but had seen only 27.6% of the ones it faced, so it played no better than random.

**Exploring matters.** On 4x4, an agent that never explored won 78.5% of games. An agent that trained on completely random moves won 97.5%, because it saw far more of the game.

**It discovered real Othello strategy.** The agent learned that corners are the most valuable squares, and that squares next to corners are risky because they can give the opponent a corner.

**Learning speed is a balance.** Learning too fast made results unstable, and learning too slowly barely improved. The middle setting worked best and reached 97% against the greedy player.

**Limiting the opponent's moves may help.** Removing this signal dropped the win rate against greedy from 95.2% to 92.4%. The gap is small, so it could also be normal variation between runs.

## Try it

You only need Python 3.8 or newer.

Play against the trained agent:

```bash
python3 othello_q.py play --agent linear --size 8 --load q_linear_8.json --human x
```

See how well it plays:

```bash
python3 othello_q.py eval --agent linear --size 8 --load q_linear_8.json
```

Train a new agent:

```bash
python3 othello_q.py train --agent linear --size 8 --episodes 6000
```

Run the tests:

```bash
python3 test_othello_q.py
```

## Files

- [othello_q.py](othello_q.py): the game and the agents
- [test_othello_q.py](test_othello_q.py): the tests
- [q_linear_8.json](q_linear_8.json): the trained 8x8 agent
- [experiments_and_questions.ipynb](experiments_and_questions.ipynb): my experiments and written answers
- [logs/](logs): raw results from each experiment
- [othello_q_no_mobility.py](othello_q_no_mobility.py): the version used to test removing the opponent-move signal
- [Lab Q-Learning Othello.pdf](<Lab Q-Learning Othello.pdf>): the original lab instructions
