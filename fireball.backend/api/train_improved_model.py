#!/usr/bin/env python3
"""
Fireball Ultimate Training Script
==================================
Creates ONE highly-trained model through intensive multi-phase training.
Takes 15+ minutes but produces the best possible model.

Training phases:
1. Exploration phase: Learn all game mechanics (random opponents)
2. Competitive phase: Train against strong self-play opponents
3. Anti-exploit phase: Learn to counter charge-only tactics
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
    Returns a function that implements the exploit strategy:
    - Charge when AI has 0 charges
    - Attack aggressively otherwise
    """
    def choose_move(ai_charges, opp_charges):
        if ai_charges == 0:
            return "charge"  # Safe to charge
        # Be aggressive
        if opp_charges >= 5:
            return "megaball"
        elif opp_charges >= 2:
            return "iceball"
        elif opp_charges >= 1:
            return "fireball"
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
        
        while not game.game_over:
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


def test_against_exploit(model, num_games=500):
    """Test model against the exploit strategy."""
    exploit_opponent = create_exploit_opponent()
    wins = 0
    
    for _ in range(num_games):
        game = FireballGame()
        model.reset_for_game()
        
        while not game.game_over and len(model.move_history) < 100:
            state = model.get_state(game.ai1_charges, game.ai2_charges)
            legal = FireballQLearning.get_legal_moves(game.ai1_charges)
            model_move = model.choose_action(state, legal, training=False)
            
            exploit_move = exploit_opponent(game.ai1_charges, game.ai2_charges)
            
            result = game.execute_turn(model_move, exploit_move)
            model.update_histories(model_move, exploit_move)
            
            if result == "ai1":
                wins += 1
                break
            elif result == "ai2":
                break
    
    return wins / num_games


def main():
    print("=" * 70)
    print("Fireball ULTIMATE Training")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will take 15-20 minutes to produce the best possible model.")
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
    print("  Testing current exploit resistance...")
    exploit_resist_before = test_against_exploit(model, 200)
    print(f"  Before: {exploit_resist_before * 100:.1f}%")
    
    model = train_phase(model, "  Training against exploit", 30000, 'exploit', verbose=False)
    
    exploit_resist_after = test_against_exploit(model, 500)
    print(f"  After: {exploit_resist_after * 100:.1f}%")
    print(f"  Improvement: +{(exploit_resist_after - exploit_resist_before) * 100:.1f}%")
    
    # Phase 4: Refinement (mix of everything)
    print("\nPhase 4: Final Refinement")
    print("  Polishing with mixed opponents...")
    
    # Do mixed training
    for i in range(30000):
        if i % 10000 == 0 and i > 0:
            print(f"    Episode {i:,}/30,000")
        
        strategy = random.choice(['self_play', 'exploit', 'random', 'random'])
        train_phase(model, "", 1, strategy, verbose=False)
    
    print("  Refinement complete!")
    
    # Set epsilon to 0 for evaluation
    model.epsilon = 0
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    print("\nExploit resistance test (1000 games)...")
    final_exploit = test_against_exploit(model, 1000)
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
