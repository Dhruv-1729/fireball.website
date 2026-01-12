import json
import os
import random
import pickle
import base64
from collections import defaultdict
import firebase_admin
from firebase_admin import credentials, firestore
from http.server import BaseHTTPRequestHandler

# ============ GAME LOGIC (inlined to avoid import issues) ============

class FireballQLearning:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.move_history = []
        self.opp_move_history = []

    def update_histories(self, my_action, opp_action):
        for history, action in [(self.move_history, my_action), (self.opp_move_history, opp_action)]:
            history.append(action)
            if len(history) > 4: history.pop(0)

    def get_state(self, my_charges, opp_charges):
        my_patt = "_".join(self.move_history[-3:]) or "start"
        opp_patt = "_".join(self.opp_move_history[-3:]) or "start"
        return f"mc_{min(my_charges, 10)}_oc_{min(opp_charges, 10)}_mypatt_{my_patt}_opppatt_{opp_patt}"

    @staticmethod
    def get_legal_moves(charges):
        moves = ["charge", "shield"]
        if charges >= 1: moves.append("fireball")
        if charges >= 2: moves.append("iceball")
        if charges >= 5: moves.append("megaball")
        return moves

    def choose_action(self, state, legal_moves, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # If state is in Q-table, use learned values
        if state in self.q_table:
            q_vals = {m: self.q_table[state][m] + random.uniform(-0.01, 0.01) for m in legal_moves}
            return max(q_vals, key=q_vals.get)
        
        # Fallback to strategic heuristics when state not found
        return self._heuristic_action(state, legal_moves)
    
    def _heuristic_action(self, state, legal_moves):
        """Strategic heuristics when Q-table doesn't have the state."""
        # Parse state to extract charges
        try:
            parts = state.split('_')
            my_charges = int(parts[1]) if len(parts) > 1 else 0
            opp_charges = int(parts[3]) if len(parts) > 3 else 0
        except:
            my_charges, opp_charges = 0, 0
        
        # If we have 5 charges, use megaball for instant win
        if 'megaball' in legal_moves:
            return 'megaball'
        
        # If opponent has 5+ charges, they might megaball - shield is useless, so attack or charge
        if opp_charges >= 5:
            # Try to attack before they megaball
            if 'iceball' in legal_moves:
                return 'iceball'
            elif 'fireball' in legal_moves:
                return 'fireball'
            return 'charge'
        
        # If opponent has no charges, they will likely charge - attack them!
        if opp_charges == 0:
            if 'fireball' in legal_moves:
                return 'fireball' if random.random() < 0.7 else 'charge'
            return 'charge'
        
        # If we have low charges and opponent has charges, be strategic
        if my_charges <= 1 and opp_charges >= 2:
            # Opponent likely to attack, consider shield but not too often
            if random.random() < 0.4:
                return 'shield'
            return 'charge'
        
        # If we have more charges than opponent, be aggressive
        if my_charges > opp_charges and my_charges >= 2:
            if 'iceball' in legal_moves and random.random() < 0.5:
                return 'iceball'
            elif 'fireball' in legal_moves:
                return 'fireball'
        
        # Default: Mix of charge and attacks, avoid shield overuse
        weights = {}
        for move in legal_moves:
            if move == 'charge':
                weights[move] = 0.4
            elif move == 'shield':
                weights[move] = 0.15  # Low weight to avoid overuse
            elif move == 'fireball':
                weights[move] = 0.25
            elif move == 'iceball':
                weights[move] = 0.15
            elif move == 'megaball':
                weights[move] = 0.05
        
        total = sum(weights.get(m, 0.1) for m in legal_moves)
        r = random.random() * total
        cumulative = 0
        for move in legal_moves:
            cumulative += weights.get(move, 0.1)
            if r <= cumulative:
                return move
        return legal_moves[0]

    def load_model(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(f))
            return True
        except Exception:
            return False

    def load_from_bytes(self, data):
        """Load model from bytes (for Firebase models)."""
        loaded = pickle.loads(data)
        self.q_table = defaultdict(lambda: defaultdict(float), loaded)


class FireballGame:
    def __init__(self):
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
            self.game_over, self.winner = True, result
        return result

    def determine_winner(self, p1, p2):
        if p1 == p2 and p1 != "megaball": return "continue"
        if p1 == "megaball": return "player1" if p2 != "megaball" else "continue"
        if p2 == "megaball": return "player2"
        if "shield" in [p1, p2]: return "continue"
        win_map = {"fireball": ["charge"], "iceball": ["charge", "fireball"]}
        if p2 in win_map.get(p1, []): return "player1"
        if p1 in win_map.get(p2, []): return "player2"
        return "continue"


# ============ FIREBASE INIT ============

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

# ============ MODEL LOADING ============

# Default: Load from local file
ai_player = FireballQLearning(epsilon=0)
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
model_loaded = ai_player.load_model(model_path)

# A/B Testing thresholds
AB_TEST_GAMES_REQUIRED = 15


def get_ml_config():
    """Get ML configuration from Firebase."""
    if not db:
        return None
    try:
        config_ref = db.collection('ml_config').document('main')
        doc = config_ref.get()
        return doc.to_dict() if doc.exists else None
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


def load_model_from_firebase(version_name):
    """Load a model from Firebase."""
    if not db or not version_name:
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
        
        # Validate that model has sufficient Q-table entries
        # An empty or near-empty Q-table indicates a broken model
        if len(model.q_table) < 10:
            print(f"Warning: Model {version_name} has only {len(model.q_table)} states, possibly corrupted")
            return None
        
        return model
    except Exception as e:
        print(f"Error loading model from Firebase: {e}")
        return None


def get_model_for_game():
    """
    Get the appropriate model for a game, handling A/B testing.
    Returns (model, model_id, version_name) tuple.
    model_id is 'A' (current) or 'B' (challenger) for tracking.
    Falls back to local model.pkl if Firebase models not available.
    """
    config = get_ml_config()
    
    # If no config, use local model
    if not config:
        return ai_player if model_loaded else None, 'A', 'local'
    
    # If A/B test is not active, use current model
    if not config.get('ab_test_active') or not config.get('challenger_model_version'):
        current_version = config.get('current_model_version')
        if current_version:
            model = load_model_from_firebase(current_version)
            if model:
                return model, 'A', current_version
        # Fallback to local
        return ai_player if model_loaded else None, 'A', 'local'
    
    # A/B test is active - randomly assign
    if random.random() < 0.5:
        model = load_model_from_firebase(config.get('current_model_version'))
        if model:
            return model, 'A', config.get('current_model_version')
        return ai_player if model_loaded else None, 'A', 'local'
    else:
        model = load_model_from_firebase(config.get('challenger_model_version'))
        if model:
            return model, 'B', config.get('challenger_model_version')
        # Fallback to current if challenger fails
        model = load_model_from_firebase(config.get('current_model_version'))
        if model:
            return model, 'A', config.get('current_model_version')
        return ai_player if model_loaded else None, 'A', 'local'


def delete_model_from_firebase(version_name):
    """Delete a model from Firebase."""
    if not db or not version_name:
        return False
    try:
        db.collection('ml_models').document(version_name).delete()
        return True
    except:
        return False


def record_game_result_and_check_ab(model_id, ai_won, config):
    """
    Record the result of a game for A/B testing and check if test should conclude.
    Returns updated config status.
    """
    if not config or not config.get('ab_test_active'):
        # Just increment game count
        update_ml_config({
            'games_since_last_training': (config.get('games_since_last_training', 0) if config else 0) + 1
        })
        return
    
    updates = {
        'games_since_last_training': config.get('games_since_last_training', 0) + 1
    }
    
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
    a_games = (config.get('model_a_games', 0) + (1 if model_id == 'A' else 0))
    b_games = (config.get('model_b_games', 0) + (1 if model_id == 'B' else 0))
    
    if a_games >= AB_TEST_GAMES_REQUIRED and b_games >= AB_TEST_GAMES_REQUIRED:
        conclude_ab_test(config, model_id, ai_won)


def conclude_ab_test(config, last_model_id, last_ai_won):
    """Conclude A/B test and promote winner."""
    # Re-fetch config to get latest values
    config = get_ml_config()
    if not config:
        return
    
    a_games = config.get('model_a_games', 0)
    a_wins = config.get('model_a_wins', 0)
    b_games = config.get('model_b_games', 0)
    b_wins = config.get('model_b_wins', 0)
    
    a_winrate = a_wins / a_games if a_games > 0 else 0
    b_winrate = b_wins / b_games if b_games > 0 else 0
    
    print(f"A/B Test Complete - A: {a_wins}/{a_games} ({a_winrate*100:.1f}%), B: {b_wins}/{b_games} ({b_winrate*100:.1f}%)")
    
    current_version = config.get('current_model_version')
    challenger_version = config.get('challenger_model_version')
    
    if b_winrate > a_winrate:
        # Challenger wins - promote it
        print(f"Challenger wins! Promoting {challenger_version}")
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
        # Current model wins
        print(f"Current model wins! Keeping {current_version}")
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


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            content_len = int(self.headers.get('Content-Length'))
            post_body = self.rfile.read(content_len)
            data = json.loads(post_body)

            player_move = data.get('playerMove')
            user_id = data.get('userId', 'unknown')

            # Get model for this game (handles A/B testing)
            model, model_id, model_version = get_model_for_game()
            
            if not model:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "AI model not found."}).encode())
                return

            game = FireballGame()
            game.player_charges = int(data.get('playerCharges', 0))
            game.comp_charges = int(data.get('aiCharges', 0))

            # Restore move history
            model.move_history = data.get('oppMoveHistory', [])
            model.opp_move_history = data.get('moveHistory', [])

            ai_legal_moves = model.get_legal_moves(game.comp_charges)
            ai_state = model.get_state(game.comp_charges, game.player_charges)
            ai_move = model.choose_action(ai_state, ai_legal_moves, training=False)

            result = game.execute_turn(player_move, ai_move)

            # Log game data to Firebase
            if db:
                try:
                    move_history = data.get('moveHistory', []) + [player_move]
                    ai_history = data.get('oppMoveHistory', []) + [ai_move]

                    db.collection('ai_game_turns').add({
                        'player_charges_before': data.get('playerCharges', 0),
                        'ai_charges_before': data.get('aiCharges', 0),
                        'player_move': player_move,
                        'ai_move': ai_move,
                        'result': result,
                        'turn_number': len(move_history),
                        'model_id': model_id,
                        'model_version': model_version,
                        'user_id': user_id,
                        'timestamp': firestore.SERVER_TIMESTAMP
                    })

                    if game.game_over:
                        ai_won = game.winner == 'player2'
                        
                        db.collection('ai_vs_human_matches').add({
                            'winner': 'ai' if ai_won else 'human',
                            'turns': len(move_history),
                            'player_moves': move_history,
                            'ai_moves': ai_history,
                            'model_id': model_id,
                            'model_version': model_version,
                            'user_id': user_id,
                            'timestamp': firestore.SERVER_TIMESTAMP
                        })
                        
                        # Record for A/B testing
                        config = get_ml_config()
                        record_game_result_and_check_ab(model_id, ai_won, config)
                        
                except Exception as log_err:
                    print(f"Logging error: {log_err}")

            response = {
                "playerCharges": game.player_charges,
                "aiCharges": game.comp_charges,
                "aiMove": ai_move,
                "result": result,
                "gameOver": game.game_over,
                "winner": game.winner,
                "modelId": model_id  # Optional: for debugging which model was used
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())