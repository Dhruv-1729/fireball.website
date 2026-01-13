"""
Game Session Management API
Handles server-authoritative game state for anti-cheat protection.
- POST /api/game_session?action=start - Start a new game session
- POST /api/game_session?action=move - Make a move with session validation
- Automatic cleanup of sessions older than 24 hours
"""

import json
import os
import random
import pickle
import base64
import uuid
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import firebase_admin
from firebase_admin import credentials, firestore
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


# ============ GAME LOGIC ============

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
        
        if state in self.q_table:
            q_vals = {m: self.q_table[state][m] + random.uniform(-0.01, 0.01) for m in legal_moves}
            return max(q_vals, key=q_vals.get)
        
        return self._heuristic_action(state, legal_moves)
    
    def _heuristic_action(self, state, legal_moves):
        try:
            parts = state.split('_')
            my_charges = int(parts[1]) if len(parts) > 1 else 0
            opp_charges = int(parts[3]) if len(parts) > 3 else 0
        except:
            my_charges, opp_charges = 0, 0
        
        if 'megaball' in legal_moves:
            return 'megaball'
        
        if opp_charges >= 5:
            if 'iceball' in legal_moves:
                return 'iceball'
            elif 'fireball' in legal_moves:
                return 'fireball'
            return 'charge'
        
        if opp_charges == 0:
            if 'fireball' in legal_moves:
                return 'fireball' if random.random() < 0.7 else 'charge'
            return 'charge'
        
        if my_charges <= 1 and opp_charges >= 2:
            if random.random() < 0.4:
                return 'shield'
            return 'charge'
        
        if my_charges > opp_charges and my_charges >= 2:
            if 'iceball' in legal_moves and random.random() < 0.5:
                return 'iceball'
            elif 'fireball' in legal_moves:
                return 'fireball'
        
        weights = {}
        for move in legal_moves:
            if move == 'charge':
                weights[move] = 0.4
            elif move == 'shield':
                weights[move] = 0.15
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

ai_player = FireballQLearning(epsilon=0)
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
model_loaded = ai_player.load_model(model_path)

AB_TEST_GAMES_REQUIRED = 15


def get_ml_config():
    if not db:
        return None
    try:
        config_ref = db.collection('ml_config').document('main')
        doc = config_ref.get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        print(f"Error getting ML config: {e}")
        return None


def load_model_from_firebase(version_name):
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
        
        if len(model.q_table) < 10:
            print(f"Warning: Model {version_name} has only {len(model.q_table)} states")
            return None
        
        return model
    except Exception as e:
        print(f"Error loading model from Firebase: {e}")
        return None


def get_model_for_game():
    config = get_ml_config()
    
    if not config:
        return ai_player if model_loaded else None, 'A', 'local'
    
    if not config.get('ab_test_active') or not config.get('challenger_model_version'):
        current_version = config.get('current_model_version')
        if current_version:
            model = load_model_from_firebase(current_version)
            if model:
                return model, 'A', current_version
        return ai_player if model_loaded else None, 'A', 'local'
    
    if random.random() < 0.5:
        model = load_model_from_firebase(config.get('current_model_version'))
        if model:
            return model, 'A', config.get('current_model_version')
        return ai_player if model_loaded else None, 'A', 'local'
    else:
        model = load_model_from_firebase(config.get('challenger_model_version'))
        if model:
            return model, 'B', config.get('challenger_model_version')
        model = load_model_from_firebase(config.get('current_model_version'))
        if model:
            return model, 'A', config.get('current_model_version')
        return ai_player if model_loaded else None, 'A', 'local'


def update_ml_config(updates):
    if not db:
        return False
    try:
        config_ref = db.collection('ml_config').document('main')
        config_ref.update(updates)
        return True
    except Exception as e:
        print(f"Error updating ML config: {e}")
        return False


def delete_model_from_firebase(version_name):
    if not db or not version_name:
        return False
    try:
        db.collection('ml_models').document(version_name).delete()
        return True
    except:
        return False


def record_game_result_and_check_ab(model_id, ai_won, config):
    if not config or not config.get('ab_test_active'):
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
    
    a_games = (config.get('model_a_games', 0) + (1 if model_id == 'A' else 0))
    b_games = (config.get('model_b_games', 0) + (1 if model_id == 'B' else 0))
    
    if a_games >= AB_TEST_GAMES_REQUIRED and b_games >= AB_TEST_GAMES_REQUIRED:
        conclude_ab_test(config, model_id, ai_won)


def conclude_ab_test(config, last_model_id, last_ai_won):
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


# ============ SESSION MANAGEMENT ============

SESSION_EXPIRY_HOURS = 24

def cleanup_old_sessions():
    """Delete game sessions older than 24 hours."""
    if not db:
        return 0
    
    try:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=SESSION_EXPIRY_HOURS)
        old_sessions = db.collection('game_sessions').where('created_at', '<', cutoff_time).limit(100).stream()
        
        deleted_count = 0
        batch = db.batch()
        
        for doc in old_sessions:
            batch.delete(doc.reference)
            deleted_count += 1
        
        if deleted_count > 0:
            batch.commit()
            print(f"Cleaned up {deleted_count} old game sessions")
        
        return deleted_count
    except Exception as e:
        print(f"Session cleanup error: {e}")
        return 0


def create_game_session(user_id):
    """Create a new server-authoritative game session."""
    if not db:
        return None
    
    # Probabilistically cleanup old sessions (1 in 10 requests)
    if random.random() < 0.1:
        cleanup_old_sessions()
    
    session_id = str(uuid.uuid4())
    model, model_id, model_version = get_model_for_game()
    
    if not model:
        return None
    
    session_data = {
        'session_id': session_id,
        'user_id': user_id,
        'player_charges': 0,
        'ai_charges': 0,
        'player_moves': [],
        'ai_moves': [],
        'game_over': False,
        'winner': None,
        'model_id': model_id,
        'model_version': model_version,
        'turn': 0,
        'created_at': firestore.SERVER_TIMESTAMP,
        'updated_at': firestore.SERVER_TIMESTAMP
    }
    
    try:
        db.collection('game_sessions').document(session_id).set(session_data)
        return session_id
    except Exception as e:
        print(f"Error creating session: {e}")
        return None


def get_game_session(session_id):
    """Retrieve a game session by ID."""
    if not db or not session_id:
        return None
    
    try:
        doc = db.collection('game_sessions').document(session_id).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        print(f"Error getting session: {e}")
        return None


def update_game_session(session_id, updates):
    """Update a game session."""
    if not db or not session_id:
        return False
    
    try:
        updates['updated_at'] = firestore.SERVER_TIMESTAMP
        db.collection('game_sessions').document(session_id).update(updates)
        return True
    except Exception as e:
        print(f"Error updating session: {e}")
        return False


def process_move(session_id, player_move):
    """
    Process a player move with server-side validation.
    Returns the game state after the move.
    """
    session = get_game_session(session_id)
    
    if not session:
        return {'error': 'Session not found or expired'}
    
    if session.get('game_over'):
        return {'error': 'Game is already over'}
    
    # Validate move is legal
    player_charges = session.get('player_charges', 0)
    legal_moves = FireballQLearning.get_legal_moves(player_charges)
    
    if player_move not in legal_moves:
        return {'error': f'Illegal move: {player_move}. Legal moves: {legal_moves}'}
    
    # Get model for AI move
    model, model_id, model_version = get_model_for_game()
    
    if not model:
        return {'error': 'AI model not available'}
    
    # Restore AI move history from session
    model.move_history = session.get('ai_moves', [])[-4:]
    model.opp_move_history = session.get('player_moves', [])[-4:]
    
    # AI chooses move
    ai_charges = session.get('ai_charges', 0)
    ai_legal_moves = model.get_legal_moves(ai_charges)
    ai_state = model.get_state(ai_charges, player_charges)
    ai_move = model.choose_action(ai_state, ai_legal_moves, training=False)
    
    # Execute the turn
    game = FireballGame()
    game.player_charges = player_charges
    game.comp_charges = ai_charges
    result = game.execute_turn(player_move, ai_move)
    
    # Update session
    new_player_moves = session.get('player_moves', []) + [player_move]
    new_ai_moves = session.get('ai_moves', []) + [ai_move]
    
    updates = {
        'player_charges': game.player_charges,
        'ai_charges': game.comp_charges,
        'player_moves': new_player_moves,
        'ai_moves': new_ai_moves,
        'game_over': game.game_over,
        'winner': game.winner,
        'turn': session.get('turn', 0) + 1
    }
    
    update_game_session(session_id, updates)
    
    # Log game data if game is over
    if game.game_over and db:
        try:
            ai_won = game.winner == 'player2'
            
            db.collection('ai_vs_human_matches').add({
                'winner': 'ai' if ai_won else 'human',
                'turns': len(new_player_moves),
                'player_moves': new_player_moves,
                'ai_moves': new_ai_moves,
                'model_id': session.get('model_id', 'A'),
                'model_version': session.get('model_version', 'local'),
                'user_id': session.get('user_id', 'unknown'),
                'session_id': session_id,
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            
            # Record for A/B testing
            config = get_ml_config()
            record_game_result_and_check_ab(session.get('model_id', 'A'), ai_won, config)
            
        except Exception as log_err:
            print(f"Logging error: {log_err}")
    
    return {
        'playerCharges': game.player_charges,
        'aiCharges': game.comp_charges,
        'aiMove': ai_move,
        'result': result,
        'gameOver': game.game_over,
        'winner': game.winner,
        'turn': updates['turn']
    }


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            # Parse query parameters
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            action = params.get('action', [''])[0]
            
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len) if content_len > 0 else b'{}'
            data = json.loads(post_body) if post_body else {}
            
            response = {}
            
            if action == 'start':
                # Start a new game session
                user_id = data.get('userId', 'anonymous')
                session_id = create_game_session(user_id)
                
                if session_id:
                    response = {
                        'success': True,
                        'sessionId': session_id,
                        'playerCharges': 0,
                        'aiCharges': 0
                    }
                else:
                    response = {'error': 'Failed to create game session'}
                    
            elif action == 'move':
                # Process a move
                session_id = data.get('sessionId')
                player_move = data.get('move')
                
                if not session_id or not player_move:
                    response = {'error': 'Missing sessionId or move'}
                else:
                    response = process_move(session_id, player_move)
                    
            elif action == 'status':
                # Get current game status
                session_id = data.get('sessionId')
                session = get_game_session(session_id)
                
                if session:
                    response = {
                        'playerCharges': session.get('player_charges', 0),
                        'aiCharges': session.get('ai_charges', 0),
                        'gameOver': session.get('game_over', False),
                        'winner': session.get('winner'),
                        'turn': session.get('turn', 0)
                    }
                else:
                    response = {'error': 'Session not found'}
                    
            else:
                response = {'error': f'Unknown action: {action}'}
            
            status_code = 200 if 'error' not in response else 400
            
            self.send_response(status_code)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
