"""
Fireball Q-Learning — Deep Model Trainer v2
=============================================
10M episodes with aggressive state exploration.

Key improvements over v1:
  - 10x more training episodes (10M total)
  - Randomized starting positions to learn mid/late-game states
  - Higher epsilon maintained longer for deeper exploration
  - More opponent variety including counter-heuristic opponents
  - Q(λ) eligibility traces for faster value propagation

Usage:
    python3 train_dmodel.py
"""

import pickle
import random
import copy
import time
import os
from collections import defaultdict

# ─── Game Logic ──────────────────────────────────────────────────────────

class FireballGame:
    def __init__(self):
        self.player_charges = 0
        self.comp_charges = 0
        self.game_over = False
        self.winner = None
        self.turns = 0

    def get_move_cost(self, move):
        return {"charge": -1, "fireball": 1, "iceball": 2, "megaball": 5}.get(move, 0)

    def execute_turn(self, p1_move, p2_move):
        self.turns += 1
        self.player_charges = max(0, self.player_charges - self.get_move_cost(p1_move))
        self.comp_charges = max(0, self.comp_charges - self.get_move_cost(p2_move))
        result = self.determine_winner(p1_move, p2_move)
        if result != "continue":
            self.game_over, self.winner = True, result
        return result

    @staticmethod
    def determine_winner(p1, p2):
        if p1 == p2 and p1 != "megaball": return "continue"
        if p1 == "megaball": return "player1" if p2 != "megaball" else "continue"
        if p2 == "megaball": return "player2"
        if "shield" in [p1, p2]: return "continue"
        win_map = {"fireball": ["charge"], "iceball": ["charge", "fireball"]}
        if p2 in win_map.get(p1, []): return "player1"
        if p1 in win_map.get(p2, []): return "player2"
        return "continue"


# ─── Q-Learning Agent ────────────────────────────────────────────────────

LEGAL_CACHE = {}

def get_legal_moves(charges):
    c = min(charges, 5)
    if c not in LEGAL_CACHE:
        moves = ["charge", "shield"]
        if c >= 1: moves.append("fireball")
        if c >= 2: moves.append("iceball")
        if c >= 5: moves.append("megaball")
        LEGAL_CACHE[c] = moves
    return LEGAL_CACHE[c]


class FireballQLearning:
    def __init__(self, lr=0.1, discount=0.9, epsilon=0.3, lam=0.9):
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.lam = lam
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.eligibility = defaultdict(lambda: defaultdict(float))
        self.move_history = []
        self.opp_move_history = []

    def reset_episode(self):
        self.eligibility.clear()
        self.move_history = []
        self.opp_move_history = []

    def seed_history(self, my_hist, opp_hist):
        """Inject fake history for mid-game starts."""
        self.move_history = list(my_hist)
        self.opp_move_history = list(opp_hist)

    def update_histories(self, my_action, opp_action):
        for h, a in [(self.move_history, my_action), (self.opp_move_history, opp_action)]:
            h.append(a)
            if len(h) > 4: h.pop(0)

    def get_state(self, my_charges, opp_charges):
        """Same state format as deployed code — ensures compatibility."""
        my_patt = "_".join(self.move_history[-3:]) or "start"
        opp_patt = "_".join(self.opp_move_history[-3:]) or "start"
        return f"mc_{min(my_charges, 10)}_oc_{min(opp_charges, 10)}_mypatt_{my_patt}_opppatt_{opp_patt}"

    def choose_action(self, state, legal_moves, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(legal_moves)
        if state in self.q_table:
            q_vals = {m: self.q_table[state][m] + random.uniform(-0.005, 0.005) for m in legal_moves}
            return max(q_vals, key=q_vals.get)
        return random.choice(legal_moves)

    def update(self, state, action, reward, next_state, next_legal, done):
        max_next_q = 0.0
        if not done and next_legal:
            max_next_q = max(self.q_table[next_state][m] for m in next_legal)
        delta = reward + self.discount * max_next_q - self.q_table[state][action]
        self.eligibility[state][action] += 1

        for s in list(self.eligibility.keys()):
            a_dict = self.eligibility[s]
            for a in list(a_dict.keys()):
                e_val = a_dict[a]
                self.q_table[s][a] += self.lr * delta * e_val
                new_e = e_val * self.discount * self.lam
                if new_e < 0.001:
                    del a_dict[a]
                else:
                    a_dict[a] = new_e
            if not a_dict:
                del self.eligibility[s]


# ─── Reward Shaping ─────────────────────────────────────────────────────

def compute_reward(result, action, opp_action,
                   ai_charges, opp_charges,
                   ai_charges_before, opp_charges_before,
                   ai_history):
    if result == "player2":
        return 25.0
    if result == "player1":
        return -25.0

    reward = 0.0

    # Charge advantage
    reward += (ai_charges - opp_charges) * 0.2

    # Punish 0-charge vulnerability
    if ai_charges == 0 and opp_charges > 0:
        reward -= 1.5
        if opp_charges >= 3: reward -= 1.0
        if opp_charges >= 5: reward -= 3.0

    # Reward charge dominance
    if ai_charges > 0 and opp_charges == 0:
        reward += 1.0

    # Megaball progress
    if ai_charges >= 3: reward += 0.2
    if ai_charges >= 4: reward += 0.4
    if ai_charges >= 5: reward += 0.5

    # Shield penalties
    if action == "shield":
        reward -= 0.2
        if opp_charges_before == 0:
            reward -= 1.5  # Useless shield
        streak = sum(1 for m in reversed(ai_history[-3:]) if m == "shield")
        if streak >= 2:
            reward -= 1.2  # Loop penalty

    # Hitting a charging opponent
    if action == "fireball" and opp_action == "charge":
        reward += 2.0
    if action == "iceball" and opp_action in ["charge", "fireball"]:
        reward += 2.5

    # Wasted attack into shield
    if action in ["fireball", "iceball"] and opp_action == "shield":
        reward -= 1.0

    # Free charge while opponent shields
    if action == "charge" and opp_action == "shield":
        reward += 0.5

    # Wasteful iceball
    if action == "iceball" and ai_charges_before == 2 and opp_charges_before < 2:
        reward -= 0.3

    # Time pressure
    reward -= 0.03

    return reward


# ─── Opponent Pool ───────────────────────────────────────────────────────

ALL_MOVES = ["charge", "shield", "fireball", "iceball", "megaball"]

def _rand_history(n=3):
    return [random.choice(ALL_MOVES) for _ in range(random.randint(0, n))]

def opp_random(c, oc, h, oh):
    return random.choice(get_legal_moves(c))

def opp_charge_spam(c, oc, h, oh):
    return "megaball" if c >= 5 else "charge"

def opp_charge_spam_smart(c, oc, h, oh):
    if c >= 5: return "megaball"
    if oc >= 1 and random.random() < 0.3: return "shield"
    return "charge"

def opp_shield_loop(c, oc, h, oh):
    if c >= 5: return "megaball"
    return "shield" if oc >= 1 else "charge"

def opp_shield_camp(c, oc, h, oh):
    if c >= 5: return "megaball"
    if oc == 0: return "charge"
    return "shield"

def opp_aggressive(c, oc, h, oh):
    if c >= 5: return "megaball"
    if c >= 2 and random.random() < 0.4: return "iceball"
    if c >= 1 and random.random() < 0.6: return "fireball"
    return "charge"

def opp_adaptive(c, oc, h, oh):
    if c >= 5: return "megaball"
    if oh:
        last = oh[-1]
        if last == "charge" and c >= 1: return "fireball"
        if last in ["fireball", "iceball"]: return "shield"
        if last == "shield": return "charge"
    return random.choice(get_legal_moves(c))

def opp_mixed_optimal(c, oc, h, oh):
    if c >= 5: return "megaball"
    if oc == 0:
        return "fireball" if c >= 1 and random.random() < 0.3 else "charge"
    if oc >= 5:
        return "fireball" if c >= 1 else "charge"
    if c == 0: return "charge"
    r = random.random()
    if r < 0.30 and c >= 1: return "fireball"
    if r < 0.45 and c >= 2: return "iceball"
    if r < 0.60: return "shield"
    return "charge"

def opp_pattern_punisher(c, oc, h, oh):
    """Watches AI history and tries to exploit patterns."""
    if c >= 5: return "megaball"
    if oh and len(oh) >= 2:
        # If AI charged twice, punish
        if oh[-1] == "charge" and oh[-2] == "charge" and c >= 1:
            return "fireball"
        # If AI shielded twice, charge freely
        if oh[-1] == "shield" and oh[-2] == "shield":
            return "charge"
        # If AI attacked, shield
        if oh[-1] in ["fireball", "iceball"]:
            return "shield"
    return random.choice(get_legal_moves(c))

def opp_bait_and_switch(c, oc, h, oh):
    """Charges a few times then switches to shield to bait AI attacks."""
    if c >= 5: return "megaball"
    if len(h) < 3:
        return "charge"
    # After baiting with charges, switch to shields
    if c >= 2 and random.random() < 0.5:
        return "shield"
    if c >= 1 and oc == 0 and random.random() < 0.6:
        return "fireball"
    return "charge"


OPPONENT_POOL = [
    (opp_random,           0.10),
    (opp_charge_spam,      0.10),
    (opp_charge_spam_smart,0.10),
    (opp_shield_loop,      0.10),
    (opp_shield_camp,      0.10),
    (opp_aggressive,       0.10),
    (opp_adaptive,         0.10),
    (opp_mixed_optimal,    0.10),
    (opp_pattern_punisher, 0.10),
    (opp_bait_and_switch,  0.10),
]

def pick_opponent():
    r = random.random()
    cum = 0
    for fn, w in OPPONENT_POOL:
        cum += w
        if r <= cum:
            return fn
    return opp_random


# ─── Training Engine ─────────────────────────────────────────────────────

def random_start():
    """Generate randomized starting charges + fake move histories.
    This forces the AI to explore mid-game and late-game states
    that it would rarely reach from a fresh start."""
    ai_c = random.choice([0, 0, 0, 1, 1, 2, 2, 3, 3, 4])
    opp_c = random.choice([0, 0, 0, 1, 1, 2, 2, 3, 3, 4])
    ai_h = _rand_history(3)
    opp_h = _rand_history(3)
    return ai_c, opp_c, ai_h, opp_h


def train_episode(ai, opponent_fn, frozen=None, use_random_start=False):
    game = FireballGame()
    ai.reset_episode()
    opp_hist, ai_hist_for_opp = [], []

    # Optionally start from a mid-game position
    if use_random_start and random.random() < 0.4:
        ai_c, opp_c, ai_h, opp_h = random_start()
        game.comp_charges = ai_c
        game.player_charges = opp_c
        ai.seed_history(ai_h, opp_h)
        opp_hist = list(opp_h)  # opponent knows its own history
        ai_hist_for_opp = list(ai_h)

    state = ai.get_state(game.comp_charges, game.player_charges)
    legal = get_legal_moves(game.comp_charges)
    action = ai.choose_action(state, legal, training=True)

    max_turns = 60

    while not game.game_over and game.turns < max_turns:
        ai_chg_before = game.comp_charges
        opp_chg_before = game.player_charges

        # Opponent move
        if frozen and random.random() < 0.5:
            opp_state = frozen.get_state(game.player_charges, game.comp_charges)
            opp_legal = get_legal_moves(game.player_charges)
            opp_move = frozen.choose_action(opp_state, opp_legal, training=False)
        else:
            opp_move = opponent_fn(game.player_charges, game.comp_charges, opp_hist, ai_hist_for_opp)

        result = game.execute_turn(opp_move, action)
        ai.update_histories(action, opp_move)
        opp_hist.append(opp_move)
        ai_hist_for_opp.append(action)

        reward = compute_reward(
            result, action, opp_move,
            game.comp_charges, game.player_charges,
            ai_chg_before, opp_chg_before,
            ai.move_history
        )

        if game.turns >= max_turns and not game.game_over:
            reward -= 5.0
            game.game_over = True

        next_state = ai.get_state(game.comp_charges, game.player_charges)
        next_legal = get_legal_moves(game.comp_charges)
        ai.update(state, action, reward, next_state, next_legal, game.game_over)

        if not game.game_over:
            state = next_state
            legal = next_legal
            action = ai.choose_action(state, legal, training=True)

    return game.winner, game.turns


def evaluate(ai, opponent_fn, name, n=1000):
    old_eps = ai.epsilon
    ai.epsilon = 0
    wins, losses, stale, tot = 0, 0, 0, 0

    for _ in range(n):
        game = FireballGame()
        ai.reset_episode()
        oh, aoh = [], []

        while not game.game_over and game.turns < 50:
            om = opponent_fn(game.player_charges, game.comp_charges, oh, aoh)
            st = ai.get_state(game.comp_charges, game.player_charges)
            ll = get_legal_moves(game.comp_charges)
            am = ai.choose_action(st, ll, training=False)
            game.execute_turn(om, am)
            ai.update_histories(am, om)
            oh.append(om)
            aoh.append(am)

        tot += game.turns
        if game.winner == "player2": wins += 1
        elif game.winner == "player1": losses += 1
        else: stale += 1

    ai.epsilon = old_eps
    wr = wins / n * 100
    print(f"    {name:30s}  W:{wins:>4} L:{losses:>4} S:{stale:>4}  WR:{wr:5.1f}%  Avg:{tot/n:.1f}t")
    return wr


def full_eval(ai, label=""):
    print(f"\n  ── Eval {label} (Q-states: {len(ai.q_table):,}) ──")
    scores = {}
    scores['random']       = evaluate(ai, opp_random,            "Random")
    scores['charge']       = evaluate(ai, opp_charge_spam,       "Charge Spam")
    scores['charge_smart'] = evaluate(ai, opp_charge_spam_smart, "Charge Spam (Smart)")
    scores['shield_loop']  = evaluate(ai, opp_shield_loop,       "Shield Loop")
    scores['shield_camp']  = evaluate(ai, opp_shield_camp,       "Shield Camp")
    scores['aggressive']   = evaluate(ai, opp_aggressive,        "Aggressive")
    scores['adaptive']     = evaluate(ai, opp_adaptive,          "Adaptive")
    scores['mixed']        = evaluate(ai, opp_mixed_optimal,     "Mixed Optimal")
    scores['punisher']     = evaluate(ai, opp_pattern_punisher,  "Pattern Punisher")
    scores['bait']         = evaluate(ai, opp_bait_and_switch,   "Bait & Switch")
    avg = sum(scores.values()) / len(scores)
    print(f"    {'AVERAGE':30s}                             {avg:5.1f}%")
    return scores


# ─── Main Pipeline ───────────────────────────────────────────────────────

def main():
    total_episodes = 10_000_000
    print("=" * 65)
    print("  FIREBALL DEEP MODEL TRAINER v2")
    print(f"  Target: dmodel.pkl | {total_episodes/1e6:.0f}M episodes")
    print("=" * 65)

    ai = FireballQLearning(lr=0.1, discount=0.9, epsilon=0.6, lam=0.9)
    frozen = None
    t0 = time.time()

    # ── PHASE 1: Foundation — Random + Diverse Starts (2M) ───────────
    phase_eps = 2_000_000
    print(f"\n▸ Phase 1/5 — Foundation (2M episodes, random starts)")
    for ep in range(phase_eps):
        train_episode(ai, opp_random, use_random_start=True)

        if (ep + 1) % 3000 == 0:
            ai.epsilon = max(0.15, ai.epsilon * 0.975)
        if (ep + 1) % 500_000 == 0:
            elapsed = time.time() - t0
            print(f"    {ep+1:>10,} | ε={ai.epsilon:.4f} | Q={len(ai.q_table):>6,} | {elapsed:.0f}s")

    full_eval(ai, "Phase 1")

    # ── PHASE 2: Self-Play (2.5M) ───────────────────────────────────
    phase_eps = 2_500_000
    print(f"\n▸ Phase 2/5 — Self-Play (2.5M episodes)")
    ai.epsilon = 0.30
    ai.lr = 0.08

    for ep in range(phase_eps):
        if ep % 25_000 == 0:
            frozen = FireballQLearning(epsilon=0.03)
            frozen.q_table = copy.deepcopy(ai.q_table)

        train_episode(ai, opp_random, frozen=frozen, use_random_start=True)

        if (ep + 1) % 3000 == 0:
            ai.epsilon = max(0.08, ai.epsilon * 0.98)
        if (ep + 1) % 500_000 == 0:
            elapsed = time.time() - t0
            print(f"    {ep+1:>10,} | ε={ai.epsilon:.4f} | Q={len(ai.q_table):>6,} | {elapsed:.0f}s")

    full_eval(ai, "Phase 2")

    # ── PHASE 3: Anti-Exploit Hardening (2.5M) ──────────────────────
    phase_eps = 2_500_000
    print(f"\n▸ Phase 3/5 — Anti-Exploit Hardening (2.5M episodes)")
    ai.epsilon = 0.20
    ai.lr = 0.06

    for ep in range(phase_eps):
        fn = pick_opponent()
        train_episode(ai, fn, use_random_start=True)

        if (ep + 1) % 3000 == 0:
            ai.epsilon = max(0.05, ai.epsilon * 0.98)
        if (ep + 1) % 500_000 == 0:
            elapsed = time.time() - t0
            print(f"    {ep+1:>10,} | ε={ai.epsilon:.4f} | Q={len(ai.q_table):>6,} | {elapsed:.0f}s")

    full_eval(ai, "Phase 3")

    # ── PHASE 4: Advanced Self-Play + Exploit (2M) ──────────────────
    phase_eps = 2_000_000
    print(f"\n▸ Phase 4/5 — Advanced Mixed Training (2M episodes)")
    ai.epsilon = 0.10
    ai.lr = 0.04

    for ep in range(phase_eps):
        if ep % 20_000 == 0:
            frozen = FireballQLearning(epsilon=0.02)
            frozen.q_table = copy.deepcopy(ai.q_table)

        if random.random() < 0.5:
            train_episode(ai, opp_random, frozen=frozen, use_random_start=True)
        else:
            fn = pick_opponent()
            train_episode(ai, fn, use_random_start=True)

        if (ep + 1) % 3000 == 0:
            ai.epsilon = max(0.02, ai.epsilon * 0.985)
        if (ep + 1) % 500_000 == 0:
            elapsed = time.time() - t0
            print(f"    {ep+1:>10,} | ε={ai.epsilon:.4f} | Q={len(ai.q_table):>6,} | {elapsed:.0f}s")

    full_eval(ai, "Phase 4")

    # ── PHASE 5: Final Polish — Low Epsilon (1M) ────────────────────
    phase_eps = 1_000_000
    print(f"\n▸ Phase 5/5 — Final Polish (1M episodes)")
    ai.epsilon = 0.04
    ai.lr = 0.02

    for ep in range(phase_eps):
        if ep % 15_000 == 0:
            frozen = FireballQLearning(epsilon=0.01)
            frozen.q_table = copy.deepcopy(ai.q_table)

        if random.random() < 0.4:
            train_episode(ai, opp_random, frozen=frozen, use_random_start=True)
        else:
            fn = pick_opponent()
            train_episode(ai, fn, use_random_start=True)

        if (ep + 1) % 3000 == 0:
            ai.epsilon = max(0.005, ai.epsilon * 0.99)
        if (ep + 1) % 250_000 == 0:
            elapsed = time.time() - t0
            print(f"    {ep+1:>10,} | ε={ai.epsilon:.4f} | Q={len(ai.q_table):>6,} | {elapsed:.0f}s")

    # ── FINAL EVALUATION ─────────────────────────────────────────────
    scores = full_eval(ai, "FINAL")

    # ── SAVE ─────────────────────────────────────────────────────────
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dmodel.pkl")
    plain_q = {k: dict(v) for k, v in ai.q_table.items()}
    with open(output_path, "wb") as f:
        pickle.dump(plain_q, f)

    file_size = os.path.getsize(output_path)
    total_time = time.time() - t0

    print(f"\n{'=' * 65}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Total episodes:    {total_episodes:>12,}")
    print(f"  Training time:     {total_time:>10.0f}s  ({total_time/60:.1f} min)")
    print(f"  Q-table states:    {len(plain_q):>12,}")
    print(f"  File size:         {file_size:>12,} bytes  ({file_size/1024:.0f} KB)")
    print(f"  Saved to:          {output_path}")
    print(f"{'=' * 65}")

    # Deployment guidance
    fb_limit = 750_000  # ~750 KB raw → 1 MB base64 → Firestore limit
    if file_size < fb_limit:
        print(f"\n  ✅ Under 750 KB — fits in Firebase Firestore document.")
        print(f"     Upload via admin panel as usual.")
    elif file_size < 5 * 1024 * 1024:
        print(f"\n  ⚠️  {file_size/1024:.0f} KB exceeds Firestore 1 MB doc limit (base64 overhead).")
        print(f"     Options:")
        print(f"       1. Include dmodel.pkl in the Git repo (Vercel bundles it)")
        print(f"       2. Compress with gzip before base64 encoding")
        print(f"       3. Use Firebase Storage instead of Firestore")
    else:
        print(f"\n  ❌ {file_size/1024/1024:.1f} MB is too large for simple deployment.")
        print(f"     Use Firebase Storage or an external CDN.")


if __name__ == "__main__":
    main()
