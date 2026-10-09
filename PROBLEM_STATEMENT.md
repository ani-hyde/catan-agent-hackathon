# Build an agent that plays Catan

**Duration: 8 hours. Team size: up to 3. Compute: CPU only.**

You will build a program that plays 2-player Catan. It plays against our bots and against the
programs the other teams build.

Use any method: search, reinforcement learning, a hand-written evaluation function, a small neural
network, a lookup table, or a combination.

**You may not use a language model as your player.** Your agent runs on one CPU core inside the
limit in section 5. You may use a language model to write code and to analyze results.

---

## 1. The game

Standard Catan with these settings.

| Setting | Value |
|---|---|
| Players | 2 |
| Victory points to win | 10 |
| Engine | catanatron, the exact build we give you |
| Board | official spiral layout, randomized board seeds |
| Trade between players | off |
| Trade with the bank and ports | on |

**Use the engine we give you. Do not install catanatron from PyPI.** The action space changed
between catanatron releases. In 3.2.1 the discard action is `DISCARD` and carries no value. In
3.3.0 it is `DISCARD_RESOURCE` and carries a resource. The robber action changed from a 3-tuple to
a 2-tuple. An agent written against another release produces actions our engine rejects. Our build
also carries a bug fix of our own, described in section 10.

---

## 2. What your agent sees

Catan is an imperfect information game and we keep it that way.

Your agent **sees**:

1. The board: tiles, numbers, ports, the robber, every settlement, city and road.
2. Your own hand: exact resource cards and exact development cards.
3. The bank: the exact count of each resource left.
4. The opponent's **public** victory points.
5. The opponent's exact **resource** hand.
6. The **number** of unplayed development cards the opponent holds.
7. Every development card either player has played.
8. The dice history and the turn number.

Your agent **does not see** which development cards the opponent holds unplayed. It therefore does
not see hidden victory point cards. **The opponent can win on a turn that looks safe to you.**

**Why the opponent's resources are not hidden.** There are 19 cards of each resource. You know the
bank and your own hand, so the opponent's hand is exactly `19 - bank - mine` for every resource.
Hiding it would hide nothing. We compute it for you instead of making every team write the same
subtraction. The development deck is the only real hidden information in 2-player Catan.

### 2.1 Searching under imperfect information

Call `obs.sample_world(rng)`. It returns a complete catanatron `Game` that agrees with everything
you can see. The opponent's unplayed development cards are drawn at random from the cards nobody
has shown. Search that copy with the normal catanatron API.

Sample several worlds and combine the results if you want to act well under uncertainty. How many
worlds you sample, and how you combine them, is your decision.

---

## 3. The engine interface

### 3.1 Actions

The engine gives you a list of legal `Action` objects. Each carries a colour, a type and a value.

| Action type | What `value` holds |
|---|---|
| `ROLL`, `END_TURN`, `BUY_DEVELOPMENT_CARD` | `None` |
| `PLAY_KNIGHT_CARD`, `PLAY_ROAD_BUILDING` | `None` |
| `BUILD_SETTLEMENT`, `BUILD_CITY` | a node id, `int` |
| `BUILD_ROAD` | a node pair, `(int, int)` |
| `MOVE_ROBBER` | `(coordinate, victim_colour_or_None)` |
| `DISCARD_RESOURCE`, `PLAY_MONOPOLY` | a resource name, for example `"WHEAT"` |
| `PLAY_YEAR_OF_PLENTY` | two resource names |
| `MARITIME_TRADE` | the cards you give, padded with `None`, then the card you get |

Five trade action types exist in the engine's enum and never appear. The engine does not generate
trades between players.

**You return an index into the list.** You never build an `Action` yourself.

### 3.2 Forced moves

When the engine offers exactly one legal action, the harness plays it. It does not call your agent.
In F-against-F play, 47% of all decisions have exactly one legal action. You do not need to handle
them and they do not consume your time budget.

### 3.3 State

`game.state` holds the board, a flat `player_state` dictionary, the resource bank and the
development deck. Your agent reads the observation object instead, which hides what section 2 says
is hidden.

The engine also offers `create_sample_vector`, which returns 614 numbers describing the position.
It is there if you want it. Whether it contains what your method needs is worth ten minutes of
thought before you train anything on it.

---

## 4. What you submit

One directory containing `agent.py` and any parameter file it loads. Total size limit 100 MB.
Your agent must import nothing from `examples/` and must not read any file outside its own
directory.

```python
class Agent:
    def __init__(self):
        """Load weights or tables here. Called once per match."""

    def decide(self, obs, playable_actions) -> int:
        """Return the INDEX of your chosen action in playable_actions."""
```

### 4.1 Two complete agents ship with the kit

**`examples/random_agent.py`** is legal, runs, and scores 0.000 against F.

```python
import random


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        return self.rng.randrange(len(playable_actions))
```

**`examples/f_agent.py` is F itself**, written against this interface. F is the bot you have to
beat, and we give you its code. It scores each position with a weighted sum of 13 features and
plays the move that leads to the best-scoring position. It looks one move ahead and no further.

```python
import random

from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

VALUE_FN = get_value_fn("base_fn", DEFAULT_WEIGHTS)


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        world = obs.sample_world(self.rng)
        best_i, best_v = 0, None
        for i, action in enumerate(playable_actions):
            successor = world.copy()
            try:
                successor.execute(action)
            except Exception:
                continue
            v = VALUE_FN(successor, obs.me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i
```

Two things in this file deserve your attention. It samples **one** world, so it is guessing the
opponent's development cards. Sampling more and averaging is strictly better. And `DEFAULT_WEIGHTS`
names 13 features: ask which of them the 614-number vector actually contains.

**Submitting F unchanged scores 0.50 against F.** Section 7 explains why that is not enough to
qualify.

### 4.2 Rules

1. Return an integer index into `playable_actions`.
2. No network access. No reading or writing outside your own directory.
3. If your agent raises an exception or returns an illegal index, we play a random legal action and
   record it. Above 2% of your moves, we disqualify the submission.
4. The practice harness is unlimited. Scored runs on our machine are rate-limited: one at a time
   per team, with a short cooldown.

---

## 5. The machine, and the time limit

### 5.1 Our machine

We publish the exact evaluation environment before the event. Every team gets the same one:

* the instance type, and the vCPU model and clock,
* **1 pinned vCPU per game, 2 GB of memory**,
* the Python version and the pinned version of every package,
* the patched catanatron build.

Timings on your own laptop are for your use only. **Our machine is the reference**, and the
leaderboard reports the numbers measured on it.

### 5.2 What we cap

We cap your agent's **own mean CPU time per game**. That number excludes the engine and excludes
your opponent, and the evaluation script prints it.

**The limit is 1,000 ms per game**, averaged over the run.

For scale, measured on our machine, mean CPU per game:

| | ms |
|---|---|
| `random_agent.py` | under 1 |
| `f_agent.py`, which is F | 81 |
| the limit | **1,000** |

So you get about twelve times F's compute. That is enough for a learned position score with a
search on top. It is not enough for brute force.

**One core means one core.** Threads and processes share it. If you exceed the budget, we play
random legal actions for the rest of that game.

### 5.3 The live leaderboard

Every scored submission shows you, measured on our machine:

| Column | Meaning |
|---|---|
| Win rate against F | over 1,000 games, with its 95% interval |
| Win rate against our bot | over 1,000 games |
| Mean agent CPU per game | against the 400 ms limit |
| Worst agent CPU per game | diagnostic |
| Agent errors | illegal index or exception |
| Submission time | |

---

## 6. The evaluation script

`catan_kit/evaluate.py` is **the script we score with**. The only things that change on the day are
the seed set and the opponents.

```
python -m catan_kit.evaluate --a examples/f_agent.py --b examples/random_agent.py --games 1000
```

Before anything else, check your install:

```
python -m catan_kit.selfcheck
```

It plays the games with seats alternated on every board seed, then prints your win rate with its
interval, your mean and worst agent CPU per game, your agent error count, and the mean game time.

Run it before every submission. A team that runs this script against the anchors can predict its
own leaderboard row.

---

## 7. Your opponents

**F** is catanatron's `ValueFunctionPlayer`. Here is the rest of the library measured against it, at
10 victory points, with seats alternated.

| Bot | Win rate against F | Games |
|---|---|---|
| `RandomPlayer` | 0.000 | 2,000 |
| `VictoryPointPlayer` | 0.001 | 2,000 |
| `WeightedRandomPlayer` | 0.003 | 2,000 |
| `MCTSPlayer`, 25 simulations | 0.000 | 400 |
| `AlphaBetaPlayer`, depth 3 | 0.399 | 1,000 |
| `SameTurnAlphaBetaPlayer`, depth 2 | 0.400 | 2,000 |

Three facts follow.

1. **Four of the six bots win almost nothing.** Your first working agent will probably win no games.
   That is normal.
2. **No bot in the library beats F.** The best reaches 0.40.
3. **The naive search bot is both the weakest and the slowest.** An obvious method is not always a
   good one.

You also play **our own bot**, which beats F by a clear margin. We do not tell you by how much.

Both bots are anchors. You may submit a copy of one. It is legal, it will place where it places, and
you will have nothing to say in the questions.

---

## 8. The funnel

### Qualification, closing at 5:00

**Score at least 0.55 against F over 1,000 games.**

The threshold sits above 0.50 on purpose. F against F reads exactly 0.50, so submitting F unchanged
scores 0.50, and any threshold at or below that admits a copy.

**If more than 30 teams qualify**, we rank by the mean of two win rates, against F and against our
bot, and take the top 30. Both numbers come from your leaderboard row, so nothing extra is required
of you.

### Group stage, 6:30

The 30 qualifiers split into 3 groups of 10, balanced by qualification score. Inside a group, every
team plays every other team 1,000 games, with seats alternated. **The top 2 from each group
present.**

Your group result decides your robustness score. Section 9 explains how.

---

## 9. How we score you

| Part | Points |
|---|---|
| Robustness | 30 |
| Strength | 20 |
| Report | 25 |
| Presentation and questions | 25 |

### 9.1 Robustness, 30 points

Your robustness score is your **mean win rate against the worst half of your group**, not your
average win rate.

Here is why, with three agents:

| | against A | against B | against C | average | worst half |
|---|---|---|---|---|---|
| A | — | 0.90 | 0.35 | 0.625 | 0.35 |
| C | 0.65 | 0.60 | — | 0.625 | 0.60 |

A and C have the same average. A beats B heavily and loses to C. C beats both by a smaller margin.
We rank C above A, and an average cannot separate them.

**An agent that beats one opponent and folds against another scores less than an agent that beats
both by a little.** Do not tune against a single opponent.

### 9.2 Strength, 20 points

Your win rate against the two anchor bots, measured during the day:

```
strength = 20 * (0.5 * win_rate_vs_F + 0.5 * win_rate_vs_our_bot)
```

### 9.3 Report, 25 points

**Up to 5 pages**, not counting figures, tables and references. There is no minimum. A clear 3-page
report beats a padded 5-page one. Submit a PDF at 6:30. Every team writes one.

Write these five sections.

1. **What you built.** The method, the inputs it reads, the final numbers. Give the win rate against
   F with the number of games.
2. **Where your data came from, and what you removed.** How many positions, from where, and what you
   discarded. Give the measurement behind each decision. "We removed opening positions" is not an
   answer. "We removed opening positions, which cost 0.02 win rate over 400 games, because it halved
   training time" is.
3. **What you decided not to do.** You cannot solve Catan in 8 hours. Name the part you ignored and
   what you think it cost.
4. **Your experiments.** Each comparison with its sample size. **Say clearly when a difference is
   inside the noise.** A win rate from 100 games carries about 10 points of uncertainty, so a
   5-point improvement measured over 100 games is not an improvement.
5. **What you would do next**, and why that and not something else.

| Marked on | Points |
|---|---|
| Method clear enough for us to re-implement | 5 |
| Data and filtering decisions, each with its number | 7 |
| Scope: what you did not do, and its cost | 5 |
| Experimental quality: controls, sample sizes, honest noise | 8 |

**A well-designed experiment that failed, reported honestly, scores near full marks.** A strong win
rate with no account of why it works does not.

### 9.4 Presentation and questions, 25 points

Six minutes, then three minutes of questions. We stop the talk at six minutes.

| Marked on | Points | Full marks require |
|---|---|---|
| **You defend a decision we pick from your report** | 12 | The number behind it, its sample size, and what you would have concluded if that number had come out the other way |
| **You diagnose your own agent** | 8 | The positions where it loses, and the evidence that found them. Not a guess |
| **You separate measurement from belief** | 5 | You say "we did not measure that" where it is true, and you name which of your results sit inside the noise |

The questions carry the most weight on purpose. A language model writes a competent search agent in
twenty minutes, and we cannot tell that from your code. You cannot outsource knowing why your
evaluation function has the terms it has.

---

## 10. The day

| Time | Event |
|---|---|
| 0:00 | We explain Catan, this problem statement, and the directions you could take |
| 0:30 | Kit released. Leaderboard opens |
| 5:00 | **Qualification closes.** 0.55 against F over 1,000 games |
| 6:30 | Final submission closes. Report due |
| 6:30 | Group stage runs |
| 7:10 | Result matrix published. Top 6 announced |
| 7:10 | Presentations |
| 8:00 | End |

There is no earlier mandatory submission. You have the evaluation script, so you can check your own
agent whenever you want.

### About reproducibility

Two separate things had to be fixed before the same seeds gave the same games twice. Both are
handled for you by the kit, and both are worth knowing about.

**1. A bug in the engine.** The function listing bank and port trades returned them in a Python
`set`, and those entries contain `None`. On Python 3.11 and older, `hash(None)` comes from the
memory address of `None`, which changes every time a process starts, so the order of the list
changed between runs. When two moves scored equally, a bot picked a different one and the game
diverged. `catan_kit` sorts that list at import. We reported the bug upstream.

**2. Python's hash randomization.** The engine also iterates sets of strings, and Python randomizes
string hashes per process. Measured: the same 60-game run gave 3953, 3785 and 3822 agent decisions
on three tries. With `PYTHONHASHSEED=0` it gives 3901 every time. That variable only takes effect
at interpreter start, so `catan_kit.evaluate` and `catan_kit.selfcheck` restart themselves with it
set. **If you write your own driver script, call `catan_kit._patch.ensure_hash_seed()` at the top
of it**, or run it with `PYTHONHASHSEED=0`.

With both in place, a run replays exactly, so you can replay a game your agent lost.

**One thing we cannot fix for you: the Python version.** The same seeds give slightly different
games on Python 3.9 and on 3.11, with every dependency pinned. **We score on Python 3.11.** Use it
if you want your local numbers to match the leaderboard exactly. Your rank does not depend on this,
because we score every submission on our own machine.

---

## 11. Reading

None of these is about Catan. All are about teaching a machine to play a game well, and each
contains an idea you could use today.

| Paper | Idea worth stealing |
|---|---|
| Tesauro, *Temporal Difference Learning and TD-Gammon*, CACM 1995 | Backgammon, a dice game, learned by self-play. The network scores the position **after** a move rather than scoring moves, which handles a changing action list. The closest of these to your problem |
| Campbell, Hoane, Hsu, *Deep Blue*, Artificial Intelligence 2002 | A hand-tuned evaluation function plus deep search beat a world champion. No learning at all |
| Silver et al., *Mastering the game of Go with deep neural networks and tree search*, Nature 2016 | A learned value function guides the search, and the search improves the training data |
| Anthony, Tian, Barber, *Thinking Fast and Slow with Deep Learning and Tree Search*, 2017, [arXiv:1705.08439](https://arxiv.org/abs/1705.08439) | The same loop stated plainly: search, train on what the search found, search again. The most copyable recipe here |
| Silver et al., *A general reinforcement learning algorithm that masters chess, shogi and Go*, [Science 2018](https://www.science.org/doi/10.1126/science.aar6404) | One algorithm, three games, no hand-written features |
| Schrittwieser et al., *MuZero*, Nature 2020, [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) | Planning with a **learned** model of the rules |
| Moravčík et al., *DeepStack*, Science 2017, [arXiv:1701.01724](https://arxiv.org/abs/1701.01724) | Poker. Acting well when you cannot see the opponent's cards, which is your situation |
| Brown and Sandholm, *Libratus*, Science 2018 | Also poker, and **not** reinforcement learning. Regret minimization plus search |
| Schmid et al., *Student of Games*, 2021, [arXiv:2112.03178](https://arxiv.org/abs/2112.03178) | One algorithm for both perfect and imperfect information games |

Catan-specific work exists. We have not listed it. You are free to find it.

---

## 12. Advice

1. **Spend the first hour on the game, not on a training loop.** The kit already plays games. Watch
   F beat your agent and work out how.
2. **Look at the state your agent receives before you train on it.** Ask whether it contains what
   your method needs.
3. **Measure against more than one opponent from the start.** Section 9.1 is built so a narrow agent
   loses to a well-rounded one.
