# Catan agent hackathon — starter kit

Build a program that plays 2-player Catan. It will play against our bots and against the other
teams' programs.

**Read [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) first.** It is the rules, the scoring and the
schedule. This file only gets you running.

---

## 1. Install

Python 3.9 or newer.

```bash
git clone https://github.com/REPO_SLUG.git
cd catan-hackathon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The engine version is pinned. **Do not upgrade it.** The action space changed between catanatron
releases, so an agent written against another version produces actions our scorer rejects.

## 2. Check the install

```bash
python -m catan_kit.selfcheck
```

Every check in that script has failed for somebody at least once. Fix any failure before you write
agent code. A broken environment here produces plausible numbers, not error messages.

## 3. Play a game

```bash
python -m catan_kit.evaluate --a examples/f_agent.py --b examples/random_agent.py --games 200
```

```
A win rate         1.0000 +-0.0000   (200/200)
A mean CPU/game        81.1 ms   limit 1000 ms   ok
A worst CPU/game      152.5 ms
A agent errors            0      of 5063 decisions (0.00%; above 2% is disqualified)
mean game time         86.3 ms   both agents plus engine
```

This is the script we score with. On the day we change only the seed set and the opponents.

## 4. Write your agent

Copy `examples/random_agent.py` and edit it.

```python
class Agent:
    def __init__(self):
        """Load weights or tables here. Called once per match."""

    def decide(self, obs, playable_actions) -> int:
        """Return the INDEX of your chosen action in playable_actions."""
```

Then measure it against F:

```bash
python -m catan_kit.evaluate --a my_agent.py --b examples/f_agent.py --games 1000
```

## 5. What is in here

| Path | What it is |
|---|---|
| `PROBLEM_STATEMENT.md` | the rules, scoring and schedule |
| `catan_kit/obs.py` | what your agent sees, and `sample_world` for search |
| `catan_kit/evaluate.py` | the scoring script |
| `catan_kit/selfcheck.py` | install check |
| `catan_kit/_patch.py` | a determinism fix for the engine, applied at import |
| `examples/random_agent.py` | 6 lines, legal, scores 0.000 against F |
| `examples/f_agent.py` | **F itself**, the bot you have to beat |

## 6. Three things that will save you an hour

1. **Spend the first hour on the game, not on a training loop.** The kit already plays games.
   Watch F beat your agent and work out how.
2. **`examples/f_agent.py` is the bot you have to beat, and you have its source.** Read it before
   you design anything.
3. **Sample size is part of every number.** A win rate from 100 games carries about 10 points of
   uncertainty. Judge a change on 1,000 games, never on 100.

## Licence

GPL-3.0-or-later, because this kit builds on catanatron. See [LICENSE](LICENSE) and
[NOTICE.md](NOTICE.md).
