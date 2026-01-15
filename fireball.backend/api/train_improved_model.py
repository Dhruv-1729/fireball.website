#!/usr/bin/env python3
"""
Fireball Ultimate Training Script
==================================
Creates ONE highly-trained model through intensive multi-phase training.
Takes ~8-12 minutes with optimizations to produce the best possible model.

Training phases:
1. Exploration phase: Learn all game mechanics (random opponents)
2. Competitive phase: Train against strong self-play opponents
3. Anti-exploit phase: Learn to counter charge-only & passive tactics
4. Refinement phase: Final polish against mixture of strategies
"""

import random
import copy
import pickle
import os
from collections import defaultdict
from datetime import datetime

class FireballQLearning:
    """Q-Learning agent for Fireball game."""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, lambda_val=0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.lambda_val = lambda_val
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.eligibility_traces = defaultdict(lambda: defaultdict(float))
        self.move_history = []
        self.opp_move_history = []

    def update_histories(self, my_action, opp_action):
        self.move_history.append(my_action)
        self.opp_move_history.append(opp_action)
        if len(self.move_history) > 4:
            self.move_history.pop(0)
        if len(self.opp_move_history) > 4:
            self.opp_move_history.pop(0)

    def get_state(self, my_charges, opp_charges):
        my_patt = "_".join(self.move_history[-3:]) or "start"
        opp_patt = "_".join(self.opp_move_history[-3:]) or "start"
        return f"mc_{min(my_charges, 10)}_oc_{min(opp_charges, 10)}_mypatt_{my_patt}_opppatt_{opp_patt}"

    @staticmethod
    def get_legal_moves(charges):
        moves = ["charge", "shield"]
        if charges >= 1:
            moves.append("fireball")
        if charges >= 2:
            moves.append("iceball")
        if charges >= 5:
            moves.append("megaball")
        return moves

    def choose_action(self, state, legal_moves, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(legal_moves)
        if state not in self.q_table:
            return random.choice(legal_moves)
        q_vals = {m: self.q_table[state][m] + random.uniform(-0.01, 0.01) for m in legal_moves}
        return max(q_vals, key=q_vals.get)

    def reset_for_game(self):
        self.move_history = []
        self.opp_move_history = []
    
    def clone(self):
        new_agent = FireballQLearning(
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            epsilon=self.epsilon,
            lambda_val=self.lambda_val
        )
        new_agent.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in self.q_table.items():
            for action, value in actions.items():
                new_agent.q_table[state][action] = value
        return new_agent


class FireballGame:
    def __init__(self):
        self.reset_game()

    def reset_game(self):
        self.ai1_charges = 0
        self.ai2_charges = 0
        self.game_over = False
        self.winner = None

    def get_move_cost(self, move):
        return {"charge": -1, "fireball": 1, "iceball": 2, "megaball": 5}.get(move, 0)

    def execute_turn(self, ai1_move, ai2_move):
        self.ai1_charges = max(0, self.ai1_charges - self.get_move_cost(ai1_move))
        self.ai2_charges = max(0, self.ai2_charges - self.get_move_cost(ai2_move))
        result = self.determine_winner(ai1_move, ai2_move)
        if result != "continue":
            self.game_over = True
            self.winner = result
        return result

    def determine_winner(self, move1, move2):
        if move1 == move2 and move1 != "megaball":
            return "continue"
        if move1 == "megaball":
            return "ai1" if move2 != "megaball" else "continue"
        if move2 == "megaball":
            return "ai2"
        if "shield" in [move1, move2]:
            return "continue"
        win_map = {"fireball": ["charge"], "iceball": ["charge", "fireball"]}
        if move2 in win_map.get(move1, []):
            return "ai1"
        if move1 in win_map.get(move2, []):
            return "ai2"
        return "continue"


def get_aggressive_reward(game, action, result, prev_my_charges, prev_opp_charges):
    """
    Ultra-aggressive reward structure.
    HEAVILY penalizes passive play and rewards finishing opponents.
    """
    reward = 0
    
    if result == "ai1":  # Win
        reward = 30
        # HUGE bonus for dominant wins
        if game.ai1_charges >= 3:
            reward += 8
        if prev_opp_charges <= 2:  # Caught them with low charges
            reward += 5
        return reward
    elif result == "ai2":  # Loss
        reward = -30
        # Extra penalty for losing with many charges (wasted opportunity)
        if game.ai1_charges >= 3:
            reward -= 10
        return reward
    
    # Continuing game - here's where we shape behavior
    
    # === CRITICAL: Being at 0 charges is EXTREMELY BAD ===
    if game.ai1_charges == 0:
        if game.ai2_charges > 0:
            reward -= 5.0  # MASSIVE penalty
            if game.ai2_charges >= 2:
                reward -= 3.0
            if game.ai2_charges >= 5:
                reward -= 5.0
    
    # === Reward charge advantage HEAVILY ===
    charge_diff = game.ai1_charges - game.ai2_charges
    reward += charge_diff * 0.5  # Strong weight
    
    # === Encourage reaching attack thresholds ===
    if game.ai1_charges >= 5:
        reward += 1.5  # Megaball ready
    elif game.ai1_charges >= 3:
        reward += 0.8
    elif game.ai1_charges >= 2:
        reward += 0.4
    
    # === HEAVILY penalize shield spam ===
    if action == "shield":
        reward -= 1.0  # Big penalty for shielding
        # Extra penalties for useless shields
        if game.ai2_charges == 0:
            reward -= 2.0  # They can't hurt you!
        if game.ai1_charges >= 5:
            reward -= 1.5  # You should be attacking with megaball!
    
    # === Reward aggressive attacks when you have advantage ===
    if action == "megaball":
        reward += 2.0  # Always good
    elif action == "iceball":
        if game.ai1_charges > game.ai2_charges:
            reward += 1.0  # Good when ahead
        if prev_my_charges == 2:  # Using iceball immediately at 2 charges
            if prev_opp_charges < 2:
                reward -= 0.8  # Wasteful
    elif action == "fireball":
        if game.ai2_charges == 0 and game.ai1_charges > 2:
            reward += 0.8  # Perfect time to attack
    
    # === Reward charging ONLY when safe ===
    if action == "charge":
        if game.ai2_charges == 0:
            reward += 0.5  # Safe to charge
        elif game.ai2_charges >= 5:
            reward -= 1.5  # VERY dangerous to charge
        elif game.ai2_charges >= 2:
            reward -= 0.5  # Risky
    
    # === Penalize long games (encourage finishing) ===
    reward -= 0.1
    
    return reward


def create_exploit_opponent():
    """
    Returns a function that implements a smart exploit strategy:
    - Takes advantage of opponent having 0 charges
    - Uses shields defensively when opponent has high charges
    - Prioritizes winning attacks over unnecessary charging
    """
    def choose_move(ai_charges, opp_charges):
        # We are the opponent, so:
        # ai_charges = the model's charges (the AI we're testing)
        # opp_charges = OUR charges (the exploit bot)
        my_charges = opp_charges  # Rename for clarity
        their_charges = ai_charges
        
        # If we have megaball, use it (instant win unless they megaball too)
        if my_charges >= 5:
            return "megaball"
        
        # If we can attack and they have 0 charges (safe attack)
        if their_charges == 0:
            if my_charges >= 2:
                return "iceball"  # Best attack when they can't counter
            elif my_charges >= 1:
                return "fireball"
            else:
                return "charge"  # Build up for attack
        
        # If they have megaball ready, we must shield or counter
        if their_charges >= 5:
            if my_charges >= 5:
                return "megaball"  # Counter megaball
            return "shield"  # Must shield
        
        # If they could iceball us and we can't counter, shield
        if their_charges >= 2 and my_charges < 2:
            # 60% chance to shield, 40% to charge (some risk-taking)
            return "shield" if random.random() < 0.6 else "charge"
        
        # If we have iceball and they have 1 charge, use it
        if my_charges >= 2 and their_charges <= 1:
            return "iceball"
        
        # If we have fireball and they're likely to charge
        if my_charges >= 1 and their_charges == 0:
            return "fireball"
        
        # Default: build charges or attack opportunistically
        if my_charges >= 2:
            return "iceball" if random.random() < 0.5 else "charge"
        elif my_charges >= 1:
            return "fireball" if random.random() < 0.3 else "charge"
        else:
            return "charge"
    
    return choose_move


def train_phase(model, phase_name, episodes, opponent_strategy, verbose=True):
    """
    Train for a specific phase with given opponent strategy.
    """
    if verbose:
        print(f"\n{phase_name}:")
        print(f"  Training for {episodes:,} episodes...")
    
    frozen_self = None
    wins = 0
    
    for episode in range(episodes):
        # Update frozen self periodically
        if episode > 0 and episode % 10000 == 0:
            frozen_self = model.clone()
            frozen_self.epsilon = 0.05
        
        game = FireballGame()
        model.eligibility_traces.clear()
        model.move_history, model.opp_move_history = [], []
        
        state = model.get_state(game.ai1_charges, game.ai2_charges)
        action = model.choose_action(state, FireballQLearning.get_legal_moves(game.ai1_charges), True)
        
        # Choose opponent based on strategy
        if opponent_strategy == 'self_play' and frozen_self:
            opponent = frozen_self
        elif opponent_strategy == 'exploit':
            opponent = create_exploit_opponent()
        else:  # random
            opponent = None
        
        if opponent and hasattr(opponent, 'reset_for_game'):
            opponent.reset_for_game()
        
        rounds = 0
        max_rounds = 50  # Prevent infinite stalemates
        
        while not game.game_over and rounds < max_rounds:
            rounds += 1
            prev_my_charges = game.ai1_charges
            prev_opp_charges = game.ai2_charges
            
            # Get opponent move
            if opponent:
                if callable(opponent):  # Exploit function
                    opp_move = opponent(game.ai1_charges, game.ai2_charges)
                else:  # AI opponent
                    opp_state = opponent.get_state(game.ai2_charges, game.ai1_charges)
                    opp_legal = FireballQLearning.get_legal_moves(game.ai2_charges)
                    opp_move = opponent.choose_action(opp_state, opp_legal, training=False)
            else:
                opp_legal = FireballQLearning.get_legal_moves(game.ai2_charges)
                opp_move = random.choice(opp_legal)
            
            result = game.execute_turn(action, opp_move)
            model.update_histories(action, opp_move)
            if opponent and hasattr(opponent, 'update_histories'):
                opponent.update_histories(opp_move, action)
            
            # Get reward
            reward = get_aggressive_reward(game, action, result, prev_my_charges, prev_opp_charges)
            
            # Q-learning update
            next_state = model.get_state(game.ai1_charges, game.ai2_charges)
            next_legal = FireballQLearning.get_legal_moves(game.ai1_charges)
            next_action = model.choose_action(next_state, next_legal, True)
            max_next_q = 0.0 if game.game_over else max([model.q_table[next_state][m] for m in next_legal] or [0.0])
            
            delta = reward + (model.discount_factor * max_next_q) - model.q_table[state][action]
            model.eligibility_traces[state][action] += 1
            
            for s_trace, a_dict in list(model.eligibility_traces.items()):
                for a_trace, e_val in list(a_dict.items()):
                    model.q_table[s_trace][a_trace] += model.learning_rate * delta * e_val
                    model.eligibility_traces[s_trace][a_trace] *= model.discount_factor * model.lambda_val
            
            state, action = next_state, next_action
            
            if result == "ai1":
                wins += 1
        
        # Epsilon decay
        if (episode + 1) % 5000 == 0:
            model.epsilon = max(0.01, model.epsilon * 0.94)
            if verbose and (episode + 1) % 10000 == 0:
                win_rate = wins / (episode + 1) * 100
                print(f"    Episode {episode + 1:,}/{episodes:,} | Epsilon: {model.epsilon:.3f} | Win rate: {win_rate:.1f}%")
    
    final_win_rate = wins / episodes * 100
    if verbose:
        print(f"  Phase complete! Win rate: {final_win_rate:.1f}%")
    
    return model


def test_against_exploit(model, num_games=500, show_progress=False):
    """Test model against the exploit strategy."""
    exploit_opponent = create_exploit_opponent()
    wins = 0
    ties = 0
    max_rounds = 50  # Prevent infinite stalemates
    
    for game_num in range(num_games):
        if show_progress and game_num > 0 and game_num % 100 == 0:
            print(f".", end="", flush=True)
        
        game = FireballGame()
        model.reset_for_game()
        rounds = 0
        
        while not game.game_over and rounds < max_rounds:
            state = model.get_state(game.ai1_charges, game.ai2_charges)
            legal = FireballQLearning.get_legal_moves(game.ai1_charges)
            model_move = model.choose_action(state, legal, training=False)
            
            exploit_move = exploit_opponent(game.ai1_charges, game.ai2_charges)
            
            result = game.execute_turn(model_move, exploit_move)
            model.update_histories(model_move, exploit_move)
            rounds += 1
            
            if result == "ai1":
                wins += 1
                break
            elif result == "ai2":
                break
        else:
            # Game reached max rounds without winner - tiebreaker by charges
            if game.ai1_charges > game.ai2_charges:
                wins += 1
            elif game.ai1_charges == game.ai2_charges:
                ties += 1
                # Split ties 50/50 for fairness
                if random.random() < 0.5:
                    wins += 1
    
    if show_progress:
        print()  # newline after dots
    
    return wins / num_games


def main():
    print("=" * 70)
    print("Fireball ULTIMATE Training")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will take ~8-12 minutes to produce the best possible model.")
    print("=" * 70)
    
    # Initialize model
    model = FireballQLearning(
        learning_rate=0.12,
        discount_factor=0.92,
        epsilon=0.5,  # Start with high exploration
        lambda_val=0.9
    )
    
    # Phase 1: Exploration (learn the game)
    model = train_phase(model, "Phase 1: Exploration & Fundamentals", 40000, 'random')
    
    # Phase 2: Self-play (learn strategy)
    model = train_phase(model, "Phase 2: Competitive Self-Play", 50000, 'self_play')
    
    # Phase 3: Anti-exploit training
    print("\nPhase 3: Anti-Exploit Training")
    print("  Testing current exploit resistance", end="", flush=True)
    exploit_resist_before = test_against_exploit(model, 200, show_progress=True)
    print(f"  Before: {exploit_resist_before * 100:.1f}%")
    
    print("  Training against exploit strategy...")
    model = train_phase(model, "  Anti-exploit training", 30000, 'exploit', verbose=True)
    
    print("  Testing improved exploit resistance", end="", flush=True)
    exploit_resist_after = test_against_exploit(model, 500, show_progress=True)
    print(f"  After: {exploit_resist_after * 100:.1f}%")
    print(f"  Improvement: +{(exploit_resist_after - exploit_resist_before) * 100:.1f}%")
    
    # Phase 4: Refinement (mix of everything) - EFFICIENT batch training
    print("\nPhase 4: Final Refinement")
    print("  Polishing with mixed opponents...")
    
    # Do mixed training efficiently (not 30k individual function calls!)
    frozen_self = model.clone()
    frozen_self.epsilon = 0.05
    exploit_fn = create_exploit_opponent()
    total_episodes = 30000
    wins = 0
    
    for episode in range(total_episodes):
        # Progress output
        if episode > 0 and episode % 10000 == 0:
            win_rate = wins / episode * 100
            print(f"    Episode {episode:,}/{total_episodes:,} | Win rate: {win_rate:.1f}%")
        
        # Rotate opponents for variety
        strategy_idx = episode % 4
        
        game = FireballGame()
        model.eligibility_traces.clear()
        model.reset_for_game()
        
        # Choose opponent based on strategy rotation
        if strategy_idx == 0:  # self-play
            opponent = frozen_self
            frozen_self.reset_for_game()
        elif strategy_idx == 1:  # exploit
            opponent = exploit_fn
        else:  # random (2 and 3)
            opponent = None
        
        state = model.get_state(game.ai1_charges, game.ai2_charges)
        action = model.choose_action(state, FireballQLearning.get_legal_moves(game.ai1_charges), True)
        
        rounds = 0
        max_rounds = 50
        
        while not game.game_over and rounds < max_rounds:
            rounds += 1
            prev_my_charges = game.ai1_charges
            prev_opp_charges = game.ai2_charges
            
            # Get opponent move
            if opponent:
                if callable(opponent):
                    opp_move = opponent(game.ai1_charges, game.ai2_charges)
                else:
                    opp_state = opponent.get_state(game.ai2_charges, game.ai1_charges)
                    opp_legal = FireballQLearning.get_legal_moves(game.ai2_charges)
                    opp_move = opponent.choose_action(opp_state, opp_legal, training=False)
            else:
                opp_legal = FireballQLearning.get_legal_moves(game.ai2_charges)
                opp_move = random.choice(opp_legal)
            
            result = game.execute_turn(action, opp_move)
            model.update_histories(action, opp_move)
            if opponent and hasattr(opponent, 'update_histories'):
                opponent.update_histories(opp_move, action)
            
            reward = get_aggressive_reward(game, action, result, prev_my_charges, prev_opp_charges)
            
            next_state = model.get_state(game.ai1_charges, game.ai2_charges)
            next_legal = FireballQLearning.get_legal_moves(game.ai1_charges)
            next_action = model.choose_action(next_state, next_legal, True)
            max_next_q = 0.0 if game.game_over else max([model.q_table[next_state][m] for m in next_legal] or [0.0])
            
            delta = reward + (model.discount_factor * max_next_q) - model.q_table[state][action]
            model.eligibility_traces[state][action] += 1
            
            for s_trace, a_dict in list(model.eligibility_traces.items()):
                for a_trace, e_val in list(a_dict.items()):
                    model.q_table[s_trace][a_trace] += model.learning_rate * delta * e_val
                    model.eligibility_traces[s_trace][a_trace] *= model.discount_factor * model.lambda_val
            
            state, action = next_state, next_action
            
            if result == "ai1":
                wins += 1
        
        # Slow epsilon decay during refinement
        if (episode + 1) % 5000 == 0:
            model.epsilon = max(0.01, model.epsilon * 0.96)
            # Update frozen self periodically
            frozen_self = model.clone()
            frozen_self.epsilon = 0.05
    
    final_refine_winrate = wins / total_episodes * 100
    print(f"    Episode {total_episodes:,}/{total_episodes:,} | Win rate: {final_refine_winrate:.1f}%")
    print("  Refinement complete!")
    
    # Set epsilon to 0 for evaluation
    model.epsilon = 0
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    print("\nExploit resistance test (1000 games)", end="", flush=True)
    final_exploit = test_against_exploit(model, 1000, show_progress=True)
    print(f"Exploit win rate: {final_exploit * 100:.1f}%")
    
    # Save model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "improved_model.pkl")
    model_data = dict(model.q_table)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model saved to: {output_path}")
    print(f"✓ Q-table size: {len(model.q_table):,} states")
    print(f"✓ Final exploit resistance: {final_exploit * 100:.1f}%")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nUpload improved_model.pkl via admin panel to deploy!")
    print("=" * 70)


if __name__ == "__main__":
    main()
