"""Play two agents against each other and report the result.

This is the script we score with. On the day we change only the seed set and the opponents.

    python -m catan_kit.evaluate --a examples/f_agent.py --b examples/random_agent.py --games 1000

Seats alternate: every board seed is played twice, once from each seat, so neither agent gets
the easier side of a lucky board.

Forced decisions are not shown to your agent. When the engine offers exactly one legal action,
the harness plays it. In F-against-F play about 47% of decisions are forced.

The number to watch is "mean CPU/game". That is your agent's own processor time. It excludes
the engine and it excludes your opponent, and it is the quantity we cap.
"""
import argparse
import importlib.util
import random
import sys
import time
from pathlib import Path

import catan_kit  # noqa: F401  (import applies the determinism patch)
from catan_kit.obs import Obs

from catanatron.game import Game
from catanatron.models.player import Color, Player

COLORS = [Color.RED, Color.BLUE]
#: Mean agent CPU seconds per game. See PROBLEM_STATEMENT.md section 5.
CPU_LIMIT_S = 1.0


def load_agent(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no such agent file: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Agent"):
        raise SystemExit(f"{path} defines no class named Agent")
    return module.Agent


class _Seat(Player):
    """Adapts a submitted Agent to the engine's player interface."""

    def __init__(self, color, agent):
        super().__init__(color)
        self.agent = agent
        self.cpu = 0.0
        self.errors = 0
        self.calls = 0

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:            # forced: do not call the agent
            return playable_actions[0]
        actions = list(playable_actions)
        self.calls += 1
        start = time.process_time()
        try:
            index = self.agent.decide(Obs(game, self.color), actions)
            action = actions[int(index)]
        except Exception:
            self.errors += 1
            action = random.choice(actions)
        self.cpu += time.process_time() - start
        return action


def play_one(AgentA, AgentB, seed, swap, vps=10):
    # The engine also draws from the bare `random` module, so seed it too.
    random.seed(seed)
    a = _Seat(COLORS[1] if swap else COLORS[0], AgentA())
    b = _Seat(COLORS[0] if swap else COLORS[1], AgentB())
    game = Game([b, a] if swap else [a, b], seed=seed, vps_to_win=vps)
    winner = game.play()
    return winner, a, b


def run(path_a, path_b, games, seed0=990000, vps=10, quiet=False):
    AgentA, AgentB = load_agent(path_a), load_agent(path_b)
    wins = errors = calls = 0
    cpu_total = cpu_worst = 0.0
    start = time.time()
    for i in range(games):
        winner, a, _ = play_one(AgentA, AgentB, seed0 + i // 2, i % 2 == 1, vps)
        wins += winner == a.color
        errors += a.errors
        calls += a.calls
        cpu_total += a.cpu
        cpu_worst = max(cpu_worst, a.cpu)
    rate = wins / games
    result = {
        "win_rate": rate,
        "ci95": 1.96 * (rate * (1 - rate) / games) ** 0.5,
        "games": games,
        "mean_cpu_s": cpu_total / games,
        "worst_cpu_s": cpu_worst,
        "errors": errors,
        "decisions": calls,
        "wall_s": time.time() - start,
    }
    if not quiet:
        over = result["mean_cpu_s"] > CPU_LIMIT_S
        err_rate = errors / calls if calls else 0.0
        print(f"A = {path_a}")
        print(f"B = {path_b}")
        print(f"A win rate         {rate:.4f} +-{result['ci95']:.4f}   ({wins}/{games})")
        print(f"A mean CPU/game    {result['mean_cpu_s']*1000:8.1f} ms   limit "
              f"{CPU_LIMIT_S*1000:.0f} ms   {'OVER LIMIT' if over else 'ok'}")
        print(f"A worst CPU/game   {result['worst_cpu_s']*1000:8.1f} ms")
        print(f"A agent errors     {errors:8d}      of {calls} decisions "
              f"({err_rate:.2%}; above 2% is disqualified)")
        print(f"mean game time     {result['wall_s']/games*1000:8.1f} ms   both agents plus engine")
    return result


def main(argv=None):
    from catan_kit._patch import ensure_hash_seed
    ensure_hash_seed()          # may restart this process; see the docstring

    p = argparse.ArgumentParser(description="Play two Catan agents and report the result.")
    p.add_argument("--a", required=True, help="path to your agent file")
    p.add_argument("--b", required=True, help="path to the opponent agent file")
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed0", type=int, default=990000, help="first board seed")
    p.add_argument("--vps", type=int, default=10)
    args = p.parse_args(argv)
    run(args.a, args.b, args.games, args.seed0, args.vps)


if __name__ == "__main__":
    sys.exit(main())
