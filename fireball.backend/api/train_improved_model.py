#!/usr/bin/env python3
"""
Fireball Evolutionary Model Training
=====================================
This script uses an evolutionary approach:
1. Load the original model.pkl as the starting champion
2. Train a new challenger model
3. Play 1000 games between challenger and champion
4. If challenger wins >50%, it becomes the new champion
5. Repeat 100 rounds
"""

import random
import copy
import pickle
import os
from collections import defaultdict
from datetime import datetime

# ============================================================
# Q-Learning Implementation
# ============================================================

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
        """Track move history for pattern recognition."""
        self.move_history.append(my_action)
        self.opp_move_history.append(opp_action)
        if len(self.move_history) > 4:
            self.move_history.pop(0)
        if len(self.opp_move_history) > 4:
            self.opp_move_history.pop(0)

    def get_state(self, my_charges, opp_charges):
        """Generate state representation including move patterns."""
        my_patt = "_".join(self.move_history[-3:]) or "start"
        opp_patt = "_".join(self.opp_move_history[-3:]) or "start"
        return f"mc_{min(my_charges, 10)}_oc_{min(opp_charges, 10)}_mypatt_{my_patt}_opppatt_{opp_patt}"

    @staticmethod
    def get_legal_moves(charges):
        """Return list of legal moves based on current charges."""
        moves = ["charge", "shield"]
        if charges >= 1:
            moves.append("fireball")
        if charges >= 2:
            moves.append("iceball")
        if charges >= 5:
            moves.append("megaball")
        return moves

    def choose_action(self, state, legal_moves, training=True):
        """Choose action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.choice(legal_moves)
        if state not in self.q_table:
            return random.choice(legal_moves)
        q_vals = {m: self.q_table[state][m] + random.uniform(-0.01, 0.01) for m in legal_moves}
        return max(q_vals, key=q_vals.get)

    def reset_for_game(self):
        """Reset histories for a new game."""
        self.move_history = []
        self.opp_move_history = []
    
    def clone(self):
        """Create a deep copy of this agent."""
        new_agent = FireballQLearning(
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            epsilon=self.epsilon,
            lambda_val=self.lambda_val
        )
        # Properly copy the q_table
        new_agent.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in self.q_table.items():
            for action, value in actions.items():
                new_agent.q_table[state][action] = value
        return new_agent


class FireballGame:
    """Game simulation for training and evaluation."""
    
    def __init__(self):
        self.reset_game()

    def reset_game(self):
        # Using generic names: ai1 and ai2 charges
        self.ai1_charges = 0
        self.ai2_charges = 0
        self.game_over = False
        self.winner = None

    def get_move_cost(self, move):
        return {"charge": -1, "fireball": 1, "iceball": 2, "megaball": 5}.get(move, 0)

    def execute_turn(self, ai1_move, ai2_move):
        """Execute a turn. Returns 'ai1', 'ai2', or 'continue'."""
        self.ai1_charges = max(0, self.ai1_charges - self.get_move_cost(ai1_move))
        self.ai2_charges = max(0, self.ai2_charges - self.get_move_cost(ai2_move))
        result = self.determine_winner(ai1_move, ai2_move)
        if result != "continue":
            self.game_over = True
            self.winner = result
        return result

    def determine_winner(self, move1, move2):
        """Determine round winner based on moves."""
        # Same moves = continue (except megaball vs megaball)
        if move1 == move2 and move1 != "megaball":
            return "continue"
        
        # Megaball beats everything except another megaball
        if move1 == "megaball":
            return "ai1" if move2 != "megaball" else "continue"
        if move2 == "megaball":
            return "ai2"
        
        # Shield blocks attacks
        if "shield" in [move1, move2]:
            return "continue"
        
        # Attack hierarchy
        win_map = {"fireball": ["charge"], "iceball": ["charge", "fireball"]}
        if move2 in win_map.get(move1, []):
            return "ai1"
        if move1 in win_map.get(move2, []):
            return "ai2"
        
        return "continue"


def load_original_model(model_path):
    """Load the original model.pkl file."""
    try:
        with open(model_path, 'rb') as f:
            q_table_data = pickle.load(f)
        
        model = FireballQLearning(epsilon=0)  # No exploration for evaluation
        
        # Handle different pickle formats
        if isinstance(q_table_data, dict):
            model.q_table = defaultdict(lambda: defaultdict(float))
            for state, actions in q_table_data.items():
                if isinstance(actions, dict):
                    for action, value in actions.items():
                        model.q_table[state][action] = value
        
        print(f"Loaded model with {len(model.q_table)} states")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def train_challenger(champion, episodes=30000, self_play_with_champion_ratio=0.5):
    """
    Train a new challenger model FROM SCRATCH.
    Uses a mix of:
    - Play against the current champion
    - Self-play against frozen versions of itself
    - Random opponent play
    
    IMPORTANT: Does NOT copy champion's Q-table - must discover strategies independently.
    """
    challenger = FireballQLearning(
        learning_rate=0.1, 
        discount_factor=0.9, 
        epsilon=0.4,
        lambda_val=0.9
    )
    
    # NO copying from champion - challenger must earn its wins
    
    frozen_self = None
    
    for episode in range(episodes):
        # Update frozen self periodically
        if episode > 0 and episode % 5000 == 0:
            frozen_self = challenger.clone()
            frozen_self.epsilon = 0.05
        
        game = FireballGame()
        challenger.eligibility_traces.clear()
        challenger.move_history, challenger.opp_move_history = [], []
        
        # Challenger is always ai1 during training
        state = challenger.get_state(game.ai1_charges, game.ai2_charges)
        action = challenger.choose_action(state, FireballQLearning.get_legal_moves(game.ai1_charges), True)
        
        # Decide opponent type
        opponent_type = random.random()
        if champion and opponent_type < self_play_with_champion_ratio:
            opponent = champion.clone()
            opponent.epsilon = 0
        elif frozen_self and opponent_type < self_play_with_champion_ratio + 0.3:
            opponent = frozen_self
        else:
            opponent = None
        
        if opponent:
            opponent.reset_for_game()
        
        while not game.game_over:
            prev_my_charges = game.ai1_charges
            prev_opp_charges = game.ai2_charges
            
            # Get opponent move
            if opponent:
                opp_state = opponent.get_state(game.ai2_charges, game.ai1_charges)
                opp_legal = FireballQLearning.get_legal_moves(game.ai2_charges)
                opp_move = opponent.choose_action(opp_state, opp_legal, training=False)
            else:
                opp_legal = FireballQLearning.get_legal_moves(game.ai2_charges)
                opp_move = random.choice(opp_legal)
            
            result = game.execute_turn(action, opp_move)
            challenger.update_histories(action, opp_move)
            if opponent:
                opponent.update_histories(opp_move, action)
            
            # === IMPROVED REWARD STRUCTURE ===
            reward = 0
            if result == "ai1":  # Challenger wins
                reward = 25
                if game.ai1_charges >= 3:
                    reward += 5
            elif result == "ai2":  # Challenger loses
                reward = -25
            else:
                # Penalize being at 0 charges while opponent has charges
                if game.ai1_charges == 0 and game.ai2_charges > 0:
                    reward -= 3.0
                    if game.ai2_charges >= 2:
                        reward -= 2.0
                    if game.ai2_charges >= 5:
                        reward -= 4.0
                
                # Reward having charges when opponent has 0
                if game.ai1_charges > 0 and game.ai2_charges == 0:
                    reward += 2.0
                
                # Charge advantage
                charge_diff = game.ai1_charges - game.ai2_charges
                reward += charge_diff * 0.3
                
                # Encourage megaball readiness
                if game.ai1_charges >= 3:
                    reward += 0.4
                if game.ai1_charges >= 4:
                    reward += 0.5
                if game.ai1_charges >= 5:
                    reward += 1.0
                
                # Penalize premature iceball
                if action == "iceball" and prev_my_charges == 2 and prev_opp_charges < 2:
                    reward -= 0.5
                
                # Penalize useless shields
                if action == "shield":
                    reward -= 0.2
                    if game.ai2_charges == 0:
                        reward -= 0.6
                
                # Reward safe charging
                if action == "charge" and game.ai2_charges <= 1:
                    reward += 0.3
                
                reward -= 0.05
            
            next_state = challenger.get_state(game.ai1_charges, game.ai2_charges)
            next_legal = FireballQLearning.get_legal_moves(game.ai1_charges)
            next_action = challenger.choose_action(next_state, next_legal, True)
            max_next_q = 0.0 if game.game_over else max([challenger.q_table[next_state][m] for m in next_legal] or [0.0])
            
            delta = reward + (challenger.discount_factor * max_next_q) - challenger.q_table[state][action]
            challenger.eligibility_traces[state][action] += 1
            
            for s_trace, a_dict in list(challenger.eligibility_traces.items()):
                for a_trace, e_val in list(a_dict.items()):
                    challenger.q_table[s_trace][a_trace] += challenger.learning_rate * delta * e_val
                    challenger.eligibility_traces[s_trace][a_trace] *= challenger.discount_factor * challenger.lambda_val
            
            state, action = next_state, next_action
        
        # Decay epsilon
        if (episode + 1) % 1000 == 0:
            challenger.epsilon = max(0.02, challenger.epsilon * 0.95)
    
    challenger.epsilon = 0
    return challenger


def play_match(ai1, ai2, num_games=1000):
    """
    Play num_games between two AIs.
    Each AI always uses its own perspective (my_charges, opp_charges).
    Returns (ai1_wins, ai2_wins, draws).
    """
    ai1_wins = 0
    ai2_wins = 0
    draws = 0
    
    for _ in range(num_games):
        game = FireballGame()
        ai1.reset_for_game()
        ai2.reset_for_game()
        
        max_turns = 150
        turn = 0
        
        while not game.game_over and turn < max_turns:
            turn += 1
            
            # Each AI uses its own perspective
            # AI1 sees: my charges = ai1_charges, opponent charges = ai2_charges
            state1 = ai1.get_state(game.ai1_charges, game.ai2_charges)
            legal1 = FireballQLearning.get_legal_moves(game.ai1_charges)
            move1 = ai1.choose_action(state1, legal1, training=False)
            
            # AI2 sees: my charges = ai2_charges, opponent charges = ai1_charges
            state2 = ai2.get_state(game.ai2_charges, game.ai1_charges)
            legal2 = FireballQLearning.get_legal_moves(game.ai2_charges)
            move2 = ai2.choose_action(state2, legal2, training=False)
            
            result = game.execute_turn(move1, move2)
            
            ai1.update_histories(move1, move2)
            ai2.update_histories(move2, move1)
            
            if result == "ai1":
                ai1_wins += 1
                break
            elif result == "ai2":
                ai2_wins += 1
                break
        
        if not game.game_over:
            draws += 1
    
    return ai1_wins, ai2_wins, draws


def test_against_exploit_strategy(model, num_games=200):
    """
    Test against the known exploit:
    - Always charge when model has 0 charges
    - Use strongest available attack otherwise
    """
    wins = 0
    
    for _ in range(num_games):
        game = FireballGame()
        model.reset_for_game()
        
        max_turns = 100
        turn = 0
        
        # Model is ai1, exploiter is ai2
        while not game.game_over and turn < max_turns:
            turn += 1
            
            # Exploit strategy
            if game.ai1_charges == 0:  # Model has 0 charges
                exploiter_move = "charge"  # Safe to charge
            else:
                # Use best available attack
                if game.ai2_charges >= 5:
                    exploiter_move = "megaball"
                elif game.ai2_charges >= 2:
                    exploiter_move = "iceball"
                elif game.ai2_charges >= 1:
                    exploiter_move = "fireball"
                else:
                    exploiter_move = "charge"
            
            # Model plays
            state = model.get_state(game.ai1_charges, game.ai2_charges)
            legal = FireballQLearning.get_legal_moves(game.ai1_charges)
            model_move = model.choose_action(state, legal, training=False)
            
            result = game.execute_turn(model_move, exploiter_move)
            model.update_histories(model_move, exploiter_move)
            
            if result == "ai1":  # Model wins
                wins += 1
                break
            elif result == "ai2":  # Exploiter wins
                break
    
    return wins / num_games


def main():
    print("=" * 70)
    print("Fireball Evolutionary Model Training")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    NUM_ROUNDS = 3
    GAMES_PER_MATCH = 5000
    WIN_THRESHOLD = 0.60
    EPISODES_PER_TRAINING = 30000
    
    # Path to original model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    original_model_path = os.path.join(script_dir, '..', 'model.pkl')
    
    print(f"Loading original model from: {original_model_path}")
    champion = load_original_model(original_model_path)
    
    if not champion:
        print("ERROR: Could not load original model.pkl!")
        print("Please ensure model.pkl exists in fireball.backend/")
        return
    
    # Test original model's exploit resistance
    print("\nTesting original model against exploit strategy...")
    orig_exploit_resistance = test_against_exploit_strategy(champion, 200)
    print(f"Original model exploit resistance: {orig_exploit_resistance * 100:.1f}%")
    
    # Verify the champion can play - test match against itself
    print("\nVerifying champion plays correctly (self-test)...")
    test_wins1, test_wins2, test_draws = play_match(champion, champion.clone(), 100)
    print(f"Self-test: {test_wins1} - {test_wins2} (draws: {test_draws})")
    if abs(test_wins1 - test_wins2) > 30:
        print("WARNING: Self-play is asymmetric! This suggests a bug.")
    else:
        print("Self-play looks balanced - good!")
    
    print()
    print(f"Starting evolutionary training: {NUM_ROUNDS} rounds")
    print(f"Challenger must win >{WIN_THRESHOLD*100:.0f}% of {GAMES_PER_MATCH} games")
    print("-" * 70)
    
    # Track stats
    champion_changes = 0
    best_winrate_seen = 0.0
    round_history = []
    
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\nRound {round_num}/{NUM_ROUNDS}")
        print(f"  Training challenger ({EPISODES_PER_TRAINING} episodes)...", end=" ", flush=True)
        
        # Train new challenger
        challenger = train_challenger(champion, episodes=EPISODES_PER_TRAINING)
        print("Done")
        
        # Play match against champion
        print(f"  Playing {GAMES_PER_MATCH} games vs champion...", end=" ", flush=True)
        challenger_wins, champion_wins, draws = play_match(challenger, champion, GAMES_PER_MATCH)
        
        total_decisive = challenger_wins + champion_wins
        if total_decisive > 0:
            challenger_winrate = challenger_wins / total_decisive
        else:
            challenger_winrate = 0.5
        
        print("Done")
        print(f"  Results: Challenger {challenger_wins} - {champion_wins} Champion (Draws: {draws})")
        print(f"  Challenger win rate: {challenger_winrate * 100:.1f}%")
        
        # Test exploit resistance
        exploit_resistance = test_against_exploit_strategy(challenger, 100)
        print(f"  Exploit resistance: {exploit_resistance * 100:.1f}%")
        
        round_history.append({
            'round': round_num,
            'challenger_wins': challenger_wins,
            'champion_wins': champion_wins,
            'draws': draws,
            'winrate': challenger_winrate,
            'exploit_resistance': exploit_resistance,
            'promoted': False
        })
        
        # Check promotion
        if challenger_winrate > WIN_THRESHOLD:
            print(f"  >>> CHALLENGER WINS! Becoming new champion.")
            champion = challenger
            champion_changes += 1
            round_history[-1]['promoted'] = True
            
            if challenger_winrate > best_winrate_seen:
                best_winrate_seen = challenger_winrate
        else:
            print(f"  Champion defends ({challenger_winrate*100:.1f}% < {WIN_THRESHOLD*100:.0f}%).")
    
    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    print(f"\nFinal Statistics:")
    print(f"  Total rounds: {NUM_ROUNDS}")
    print(f"  Champion changes: {champion_changes}")
    print(f"  Best challenger win rate: {best_winrate_seen * 100:.1f}%")
    
    # Final exploit test
    print("\nFinal champion exploit resistance (500 games)...")
    final_exploit_resistance = test_against_exploit_strategy(champion, 500)
    print(f"  Final: {final_exploit_resistance * 100:.1f}%")
    print(f"  Original: {orig_exploit_resistance * 100:.1f}%")
    
    # Save final champion
    output_path = os.path.join(script_dir, "improved_model.pkl")
    model_data = dict(champion.q_table)
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nSaved to: {output_path}")
    print(f"Q-table size: {len(champion.q_table)} states")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
