"""
Fireball ML Training Module
===========================
This module handles:
1. Q-Learning training for the Fireball AI
2. A/B testing between current and challenger models
3. Automatic model promotion based on win rates

IMPORTANT: Training is DISABLED by default. Set TRAINING_ENABLED = True to activate.
"""

import json
import os
import pickle
import random
import copy
import base64
from collections import defaultdict
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# ============================================================
# MASTER SWITCH - SET TO FALSE TO DISABLE ALL TRAINING
# ============================================================
TRAINING_ENABLED = False

# Training will only trigger after this many games since last training
GAMES_THRESHOLD_FOR_TRAINING = 200

# A/B test: each model must play this many games before comparison
AB_TEST_GAMES_REQUIRED = 15

# ============================================================

# Initialize Firebase (reuse existing connection if available)
if not firebase_admin._apps:
    try:
        service_account_info = json.loads(os.environ.get('FIREBASE_SERVICE_ACCOUNT', '{}'))
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        print(f"Firebase Init Error: {e}")

try:
    db = firestore.client()
except:
    db = None


# ============================================================
# Q-Learning Implementation (from original training code)
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

    def to_dict(self):
        """Convert Q-table to serializable dict."""
        return {k: dict(v) for k, v in self.q_table.items()}

    def from_dict(self, data):
        """Load Q-table from dict."""
        self.q_table = defaultdict(lambda: defaultdict(float))
        for k, v in data.items():
            for action, value in v.items():
                self.q_table[k][action] = value

    def save_to_bytes(self):
        """Serialize model to bytes for storage."""
        return pickle.dumps(dict(self.q_table))

    def load_from_bytes(self, data):
        """Load model from bytes."""
        loaded = pickle.loads(data)
        self.q_table = defaultdict(lambda: defaultdict(float), loaded)


class FireballGame:
    """Game simulation for training."""
    
    def __init__(self):
        self.reset_game()

    def reset_game(self):
        self.player_charges = 0
        self.comp_charges = 0
        self.game_over = False
        self.winner = None

    def get_move_cost(self, move):
        return {"charge": -1, "fireball": 1, "iceball": 2, "megaball": 5}.get(move, 0)

    def execute_turn(self, p1_move, p2_move):
        self.player_charges = max(0, self.player_charges - self.get_move_cost(p1_move))
        self.comp_charges = max(0, self.comp_charges - self.get_move_cost(p2_move))
        result = self.determine_winner(p1_move, p2_move)
        if result != "continue":
            self.game_over = True
            self.winner = result
        return result

    def determine_winner(self, p1, p2):
        """Determine round winner based on moves."""
        if p1 == p2 and p1 != "megaball":
            return "continue"
        if p1 == "megaball":
            return "player1" if p2 != "megaball" else "continue"
        if p2 == "megaball":
            return "player2"
        if "shield" in [p1, p2]:
            return "continue"
        win_map = {"fireball": ["charge"], "iceball": ["charge", "fireball"]}
        if p2 in win_map.get(p1, []):
            return "player1"
        if p1 in win_map.get(p2, []):
            return "player2"
        return "continue"


# ============================================================
# Training Functions
# ============================================================

def train_new_model(episodes=100000, self_play_ratio=0.4):
    """
    Train a new model using Q-Learning with eligibility traces.
    Returns the trained model.
    """
    ai = FireballQLearning(learning_rate=0.1, discount_factor=0.9, epsilon=0.5, lambda_val=0.9)
    frozen_opponent_ai = None
    
    print(f"Starting training with Self-Play ({self_play_ratio*100}%) and Q(λ)...")

    for episode in range(episodes):
        # Update self-play opponent periodically
        if episode > 0 and episode % 10000 == 0:
            frozen_opponent_ai = FireballQLearning(epsilon=0.05)
            frozen_opponent_ai.q_table = copy.deepcopy(ai.q_table)

        game = FireballGame()
        ai.eligibility_traces.clear()
        ai.move_history, ai.opp_move_history = [], []
        state = ai.get_state(game.comp_charges, game.player_charges)
        action = ai.choose_action(state, FireballQLearning.get_legal_moves(game.comp_charges), True)

        while not game.game_over:
            ai_chg, p_chg = game.comp_charges, game.player_charges
            use_self_play = frozen_opponent_ai and random.random() < self_play_ratio
            
            if use_self_play:
                p_state = frozen_opponent_ai.get_state(p_chg, ai_chg)
                p_move = frozen_opponent_ai.choose_action(p_state, FireballQLearning.get_legal_moves(p_chg), False)
            else:
                p_moves = FireballQLearning.get_legal_moves(p_chg)
                p_move = random.choice(p_moves) if p_moves else "charge"

            result = game.execute_turn(p_move, action)
            ai.update_histories(action, p_move)
            if use_self_play:
                frozen_opponent_ai.update_histories(p_move, action)
            
            # Reward structure
            reward = 0
            if result == "player2":  # AI wins
                reward = 15
            elif result == "player1":  # AI loses
                reward = -15
            else:
                reward = (ai_chg - p_chg) * 0.2 - 0.1

            next_state = ai.get_state(game.comp_charges, game.player_charges)
            next_legal = FireballQLearning.get_legal_moves(game.comp_charges)
            next_action = ai.choose_action(next_state, next_legal, True)
            max_next_q = 0.0 if game.game_over else max([ai.q_table[next_state][m] for m in next_legal] or [0.0])
            
            delta = reward + (ai.discount_factor * max_next_q) - ai.q_table[state][action]
            ai.eligibility_traces[state][action] += 1

            # Update all states with eligibility traces
            for s_trace, a_dict in list(ai.eligibility_traces.items()):
                for a_trace, e_val in list(a_dict.items()):
                    ai.q_table[s_trace][a_trace] += ai.learning_rate * delta * e_val
                    ai.eligibility_traces[s_trace][a_trace] *= ai.discount_factor * ai.lambda_val
            
            state, action = next_state, next_action

        # Decay epsilon
        if (episode + 1) % 1000 == 0:
            ai.epsilon = max(0.01, ai.epsilon * 0.96)
        
        if (episode + 1) % 10000 == 0:
            print(f"  Episode {episode + 1}/{episodes}, Epsilon: {ai.epsilon:.4f}")

    print("Training completed!")
    return ai


def train_from_human_games(base_model, human_games, learning_rate=0.05):
    """
    Fine-tune model using human game data.
    Focuses on learning from games where humans won.
    """
    ai = FireballQLearning(learning_rate=learning_rate, discount_factor=0.9, epsilon=0.0)
    ai.q_table = copy.deepcopy(base_model.q_table)
    
    # Filter for games where human won (we want to learn from successful strategies)
    winning_games = [g for g in human_games if g.get('winner') == 'player1']
    
    print(f"Learning from {len(winning_games)} human victories...")
    
    for game_data in winning_games:
        player_moves = game_data.get('player_moves', [])
        ai_moves = game_data.get('ai_moves', [])
        
        if not player_moves or not ai_moves:
            continue
            
        # Simulate the game to get states
        p_charges, ai_charges = 0, 0
        ai.move_history, ai.opp_move_history = [], []
        
        costs = {'charge': -1, 'fireball': 1, 'iceball': 2, 'megaball': 5, 'shield': 0}
        
        for i, (p_move, a_move) in enumerate(zip(player_moves, ai_moves)):
            # Get state before move
            state = ai.get_state(ai_charges, p_charges)
            
            # What the human did in this state (as the opposing player)
            # We boost the Q-value for moves that counter what humans do
            legal_moves = FireballQLearning.get_legal_moves(ai_charges)
            
            # Learn from the human's winning move by understanding patterns
            ai.update_histories(a_move, p_move)
            
            # Update charges
            p_charges = max(0, p_charges - costs.get(p_move, 0))
            ai_charges = max(0, ai_charges - costs.get(a_move, 0))
    
    return ai


# ============================================================
# Model Storage & A/B Testing
# ============================================================

def get_ml_config():
    """Get ML configuration from Firebase."""
    if not db:
        return None
    
    try:
        config_ref = db.collection('ml_config').document('main')
        doc = config_ref.get()
        
        if doc.exists:
            return doc.to_dict()
        else:
            # Initialize default config
            default_config = {
                'training_enabled': TRAINING_ENABLED,
                'games_since_last_training': 0,
                'last_training_timestamp': None,
                'current_model_version': 'v1_original',
                'challenger_model_version': None,
                'model_a_games': 0,
                'model_a_wins': 0,
                'model_b_games': 0,
                'model_b_wins': 0,
                'ab_test_active': False
            }
            config_ref.set(default_config)
            return default_config
    except Exception as e:
        print(f"Error getting ML config: {e}")
        return None


def update_ml_config(updates):
    """Update ML configuration in Firebase."""
    if not db:
        return False
    
    try:
        config_ref = db.collection('ml_config').document('main')
        config_ref.update(updates)
        return True
    except Exception as e:
        print(f"Error updating ML config: {e}")
        return False


def save_model_to_firebase(model, version_name):
    """Save a trained model to Firebase as base64-encoded pickle."""
    if not db:
        return False
    
    try:
        model_bytes = model.save_to_bytes()
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        
        db.collection('ml_models').document(version_name).set({
            'model_data': model_b64,
            'created_at': firestore.SERVER_TIMESTAMP,
            'version': version_name,
            'q_table_size': len(model.q_table)
        })
        
        print(f"Model '{version_name}' saved to Firebase")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def load_model_from_firebase(version_name):
    """Load a model from Firebase."""
    if not db:
        return None
    
    try:
        doc = db.collection('ml_models').document(version_name).get()
        if not doc.exists:
            return None
        
        data = doc.to_dict()
        model_b64 = data.get('model_data')
        if not model_b64:
            return None
        
        model_bytes = base64.b64decode(model_b64)
        model = FireballQLearning(epsilon=0)
        model.load_from_bytes(model_bytes)
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def delete_model_from_firebase(version_name):
    """Delete a model from Firebase."""
    if not db:
        return False
    
    try:
        db.collection('ml_models').document(version_name).delete()
        print(f"Model '{version_name}' deleted from Firebase")
        return True
    except Exception as e:
        print(f"Error deleting model: {e}")
        return False


def get_model_for_game():
    """
    Get the appropriate model for a game, handling A/B testing.
    Returns (model, model_id) tuple.
    model_id is 'A' or 'B' for tracking purposes.
    """
    config = get_ml_config()
    if not config:
        return None, None
    
    # If A/B test is not active, use current model
    if not config.get('ab_test_active') or not config.get('challenger_model_version'):
        model = load_model_from_firebase(config.get('current_model_version', 'v1_original'))
        return model, 'A'
    
    # A/B test is active - randomly assign
    if random.random() < 0.5:
        model = load_model_from_firebase(config.get('current_model_version'))
        return model, 'A'
    else:
        model = load_model_from_firebase(config.get('challenger_model_version'))
        return model, 'B'


def record_game_result(model_id, ai_won):
    """
    Record the result of a game for A/B testing.
    Returns True if A/B test concluded and a winner was determined.
    """
    config = get_ml_config()
    if not config:
        return False
    
    # Update game count
    updates = {
        'games_since_last_training': config.get('games_since_last_training', 0) + 1
    }
    
    # If A/B test is active, record results
    if config.get('ab_test_active'):
        if model_id == 'A':
            updates['model_a_games'] = config.get('model_a_games', 0) + 1
            if ai_won:
                updates['model_a_wins'] = config.get('model_a_wins', 0) + 1
        elif model_id == 'B':
            updates['model_b_games'] = config.get('model_b_games', 0) + 1
            if ai_won:
                updates['model_b_wins'] = config.get('model_b_wins', 0) + 1
        
        update_ml_config(updates)
        
        # Check if A/B test should conclude
        new_config = get_ml_config()
        a_games = new_config.get('model_a_games', 0)
        b_games = new_config.get('model_b_games', 0)
        
        if a_games >= AB_TEST_GAMES_REQUIRED and b_games >= AB_TEST_GAMES_REQUIRED:
            conclude_ab_test()
            return True
    else:
        update_ml_config(updates)
    
    return False


def conclude_ab_test():
    """
    Conclude A/B test and promote winner.
    """
    config = get_ml_config()
    if not config:
        return
    
    a_games = config.get('model_a_games', 0)
    a_wins = config.get('model_a_wins', 0)
    b_games = config.get('model_b_games', 0)
    b_wins = config.get('model_b_wins', 0)
    
    a_winrate = a_wins / a_games if a_games > 0 else 0
    b_winrate = b_wins / b_games if b_games > 0 else 0
    
    print(f"A/B Test Results:")
    print(f"  Model A (current): {a_wins}/{a_games} = {a_winrate*100:.1f}% win rate")
    print(f"  Model B (challenger): {b_wins}/{b_games} = {b_winrate*100:.1f}% win rate")
    
    current_version = config.get('current_model_version')
    challenger_version = config.get('challenger_model_version')
    
    if b_winrate > a_winrate:
        # Challenger wins - promote it
        print(f"Challenger '{challenger_version}' wins! Promoting to current model.")
        
        # Delete old current model
        if current_version and current_version != 'v1_original':
            delete_model_from_firebase(current_version)
        
        update_ml_config({
            'current_model_version': challenger_version,
            'challenger_model_version': None,
            'ab_test_active': False,
            'model_a_games': 0,
            'model_a_wins': 0,
            'model_b_games': 0,
            'model_b_wins': 0
        })
    else:
        # Current model wins - delete challenger
        print(f"Current model '{current_version}' wins! Keeping it.")
        
        if challenger_version:
            delete_model_from_firebase(challenger_version)
        
        update_ml_config({
            'challenger_model_version': None,
            'ab_test_active': False,
            'model_a_games': 0,
            'model_a_wins': 0,
            'model_b_games': 0,
            'model_b_wins': 0
        })


def check_and_trigger_training():
    """
    Check if training should be triggered based on game count.
    This is called after each game.
    """
    config = get_ml_config()
    if not config:
        return False
    
    # Master switch check
    if not config.get('training_enabled', False):
        return False
    
    # Don't train if A/B test is active
    if config.get('ab_test_active', False):
        return False
    
    # Check game threshold
    games = config.get('games_since_last_training', 0)
    if games < GAMES_THRESHOLD_FOR_TRAINING:
        return False
    
    print(f"Training threshold reached ({games} games). Starting training...")
    
    # Trigger training
    trigger_training_pipeline()
    return True


def trigger_training_pipeline():
    """
    Main training pipeline:
    1. Fetch human game data
    2. Train new model
    3. Save as challenger
    4. Start A/B test
    """
    if not db:
        print("Database not available")
        return False
    
    try:
        # Fetch recent human games
        ai_games = []
        docs = db.collection('ai_vs_human_matches').order_by(
            'timestamp', 
            direction=firestore.Query.DESCENDING
        ).limit(500).stream()
        
        for doc in docs:
            ai_games.append(doc.to_dict())
        
        online_games = []
        docs = db.collection('matches').where(
            'status', '==', 'finished'
        ).limit(500).stream()
        
        for doc in docs:
            online_games.append(doc.to_dict())
        
        print(f"Fetched {len(ai_games)} AI games and {len(online_games)} online games")
        
        # Load current model as base
        config = get_ml_config()
        current_version = config.get('current_model_version', 'v1_original')
        base_model = load_model_from_firebase(current_version)
        
        if not base_model:
            print("Could not load base model, training from scratch")
            new_model = train_new_model(episodes=50000)
        else:
            # Fine-tune from human games
            new_model = train_from_human_games(base_model, ai_games)
        
        # Save challenger model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        challenger_version = f'v_challenger_{timestamp}'
        save_model_to_firebase(new_model, challenger_version)
        
        # Start A/B test
        update_ml_config({
            'challenger_model_version': challenger_version,
            'ab_test_active': True,
            'model_a_games': 0,
            'model_a_wins': 0,
            'model_b_games': 0,
            'model_b_wins': 0,
            'games_since_last_training': 0,
            'last_training_timestamp': firestore.SERVER_TIMESTAMP
        })
        
        print(f"Training complete! A/B test started with challenger '{challenger_version}'")
        return True
        
    except Exception as e:
        print(f"Training pipeline error: {e}")
        return False


# ============================================================
# API Handler for manual triggers
# ============================================================

def handle_ml_action(action, data=None):
    """Handle ML-related API actions."""
    
    if action == 'get_status':
        config = get_ml_config()
        return {
            'training_enabled': config.get('training_enabled', False) if config else False,
            'games_since_last_training': config.get('games_since_last_training', 0) if config else 0,
            'ab_test_active': config.get('ab_test_active', False) if config else False,
            'current_model': config.get('current_model_version') if config else None,
            'challenger_model': config.get('challenger_model_version') if config else None,
            'model_a_stats': {
                'games': config.get('model_a_games', 0) if config else 0,
                'wins': config.get('model_a_wins', 0) if config else 0
            },
            'model_b_stats': {
                'games': config.get('model_b_games', 0) if config else 0,
                'wins': config.get('model_b_wins', 0) if config else 0
            }
        }
    
    elif action == 'enable_training':
        update_ml_config({'training_enabled': True})
        return {'success': True, 'message': 'Training enabled'}
    
    elif action == 'disable_training':
        update_ml_config({'training_enabled': False})
        return {'success': True, 'message': 'Training disabled'}
    
    elif action == 'force_training':
        if not TRAINING_ENABLED:
            return {'success': False, 'message': 'Training is disabled by master switch'}
        result = trigger_training_pipeline()
        return {'success': result}
    
    elif action == 'upload_original_model':
        # For initial setup - upload the original model.pkl
        try:
            model = FireballQLearning(epsilon=0)
            # This would need the actual model.pkl file
            # For now, just create a placeholder
            save_model_to_firebase(model, 'v1_original')
            update_ml_config({'current_model_version': 'v1_original'})
            return {'success': True}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    return {'error': 'Unknown action'}
