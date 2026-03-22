
import json
import os
import random
import pickle
import base64
import uuid
import time
import threading
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs



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
            import math
            temperature = 3.0
            
            q_vals = {m: self.q_table[state][m] for m in legal_moves}
            max_q = max(q_vals.values())
            
            exp_vals = {m: math.exp((q - max_q) / temperature) for m, q in q_vals.items()}
            total_exp = sum(exp_vals.values())
            
            probs = {m: v / total_exp for m, v in exp_vals.items()}
            
            r = random.random()
            cumulative = 0.0
            for move, prob in probs.items():
                cumulative += prob
                if r <= cumulative:
                    return move
            return legal_moves[-1]
        
        return self._heuristic_action(state, legal_moves)

    @staticmethod
    def _count_streak(history, move):
        streak = 0
        for m in reversed(history):
            if m == move:
                streak += 1
            else:
                break
        return streak

    def _heuristic_action(self, state, legal_moves):
        try:
            parts = state.split('_')
            my_charges = int(parts[1]) if len(parts) > 1 else 0
            opp_charges = int(parts[3]) if len(parts) > 3 else 0
        except:
            my_charges, opp_charges = 0, 0

        opp_recent = self.opp_move_history[-4:] if self.opp_move_history else []
        my_recent = self.move_history[-4:] if self.move_history else []

        opp_charge_streak = self._count_streak(opp_recent, 'charge')
        opp_shield_streak = self._count_streak(opp_recent, 'shield')
        my_shield_streak = self._count_streak(my_recent, 'shield')

        aggressive = random.random() < 0.25

        if 'megaball' in legal_moves:
            return 'megaball'

        if opp_charge_streak >= 2:
            if opp_charges >= 4:
                if 'iceball' in legal_moves:
                    return 'iceball'
                if 'fireball' in legal_moves:
                    return 'fireball'
            if 'fireball' in legal_moves:
                return 'fireball'
            if 'iceball' in legal_moves:
                return 'iceball'

        if opp_shield_streak >= 2:
            if my_charges >= 4:
                return 'charge'
            return 'charge'

        if my_shield_streak >= 2 and opp_shield_streak >= 1:
            return 'charge'

        if opp_charges >= 5:
            if 'iceball' in legal_moves:
                return 'iceball'
            if 'fireball' in legal_moves:
                return 'fireball'
            return 'charge'

        if opp_charges == 0:
            if 'fireball' in legal_moves:
                prob = 0.85 if aggressive else 0.6
                return 'fireball' if random.random() < prob else 'charge'
            return 'charge'

        if len(opp_recent) > 0 and opp_recent[-1] == 'charge':
            if 'fireball' in legal_moves:
                prob = 0.7 if aggressive else 0.45
                if random.random() < prob:
                    return 'fireball'

        if aggressive:
            if my_charges > opp_charges and my_charges >= 2:
                if 'iceball' in legal_moves and random.random() < 0.5:
                    return 'iceball'
                if 'fireball' in legal_moves:
                    return 'fireball'

            if opp_charges >= 3 and my_charges <= 1:
                return 'shield' if random.random() < 0.5 else 'charge'

            if 'fireball' in legal_moves and random.random() < 0.4:
                return 'fireball'

            return 'charge'
        else:
            if my_charges <= 1 and opp_charges >= 3:
                return 'shield' if random.random() < 0.4 else 'charge'

            if my_charges > opp_charges and my_charges >= 2:
                if 'iceball' in legal_moves and random.random() < 0.3:
                    return 'iceball'
                if 'fireball' in legal_moves and random.random() < 0.35:
                    return 'fireball'

            weights = {
                'charge': 0.45,
                'shield': 0.08,
                'fireball': 0.30,
                'iceball': 0.12,
                'megaball': 0.05
            }
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
        import gzip
        try:
            if data.startswith(b'\x1f\x8b'):
                data = gzip.decompress(data)
            loaded = pickle.loads(data)
            self.q_table = defaultdict(lambda: defaultdict(float), loaded)
        except Exception as e:
            print(f"Error loading model from bytes: {e}")
            self.q_table = defaultdict(lambda: defaultdict(float))


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



def _load_local_model():
    model = FireballQLearning(epsilon=0)
    base_dir = os.path.join(os.path.dirname(__file__), '..')

    for filename in ('model.pkl.gz', 'model.pkl'):
        path = os.path.join(base_dir, filename)
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model.load_from_bytes(f.read())
                if len(model.q_table) > 10:
                    print(f"[game_session] Loaded local model from {filename}: {len(model.q_table)} states")
                    return model, True
                else:
                    print(f"[game_session] Model {filename} loaded but only {len(model.q_table)} states — skipping")
        except Exception as e:
            print(f"[game_session] Error loading {filename}: {e}")

    print("[game_session] WARNING: No local model found, will fall back to Firebase")
    return model, False


ai_player, _model_loaded = _load_local_model()
_LOCAL_Q_TABLE = ai_player.q_table if _model_loaded else None

_PREFER_LOCAL_MODEL = os.environ.get('PREFER_LOCAL_MODEL', 'true').lower() not in ('0', 'false', 'no')



_db = None
_db_init_lock = threading.Lock()
_firebase_init_done = False


def _get_db():
    global _db, _firebase_init_done
    if _db is not None:
        return _db

    with _db_init_lock:
        if _db is not None:
            return _db

        if not _firebase_init_done:
            _firebase_init_done = True
            try:
                import firebase_admin
                if not firebase_admin._apps:
                    from firebase_admin import credentials as _creds
                    sa_info = json.loads(os.environ.get('FIREBASE_SERVICE_ACCOUNT', '{}'))
                    if sa_info:
                        cred = _creds.Certificate(sa_info)
                        firebase_admin.initialize_app(cred)
                    else:
                        print("[game_session] FIREBASE_SERVICE_ACCOUNT env var not set")
                        return None
            except Exception as e:
                print(f"[game_session] Firebase Init Error: {e}")
                return None

        try:
            from firebase_admin import firestore as _fs
            _db = _fs.client()
        except Exception as e:
            print(f"[game_session] Firestore client error: {e}")

    return _db



AB_TEST_GAMES_REQUIRED = 15


def _build_model_from_q_table(q_table):
    model = FireballQLearning(epsilon=0)
    model.q_table = q_table
    return model


def get_local_model():
    if _LOCAL_Q_TABLE:
        return _build_model_from_q_table(_LOCAL_Q_TABLE), 'A', 'local'
    return None, 'A', 'local'


def get_ml_config():
    db = _get_db()
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
    db = _get_db()
    if not db or not version_name:
        return None
    try:
        doc = db.collection('ml_models').document(version_name).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        
        if data.get('chunked'):
            total_chunks = data.get('total_chunks', 0)
            chunk_refs = [
                db.collection('ml_models').document(f"{version_name}_chunk_{i}")
                for i in range(total_chunks)
            ]
            chunk_docs = list(db.get_all(chunk_refs))
            chunks = [""] * total_chunks
            for chunk_doc in chunk_docs:
                if not chunk_doc.exists:
                    print(f"Missing chunk doc for {version_name}")
                    return None
                payload = chunk_doc.to_dict() or {}
                idx = payload.get('chunk_index')
                if idx is None:
                    try:
                        idx = int(chunk_doc.id.rsplit('_', 1)[-1])
                    except Exception:
                        idx = None
                if idx is None or idx < 0 or idx >= total_chunks:
                    print(f"Invalid chunk index for {version_name}: {idx}")
                    return None
                chunks[idx] = payload.get('data', '')
            if any(not c for c in chunks):
                print(f"Missing chunk data for {version_name}")
                return None
            model_b64 = "".join(chunks)
        else:
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
    if _PREFER_LOCAL_MODEL and _LOCAL_Q_TABLE:
        return _build_model_from_q_table(_LOCAL_Q_TABLE), 'A', 'local'

    config = get_ml_config()
    
    if not config:
        return (_build_model_from_q_table(_LOCAL_Q_TABLE) if _LOCAL_Q_TABLE else None), 'A', 'local'
    
    if not config.get('ab_test_active') or not config.get('challenger_model_version'):
        current_version = config.get('current_model_version')
        if current_version:
            model = load_model_from_firebase(current_version)
            if model:
                return model, 'A', current_version
        return (_build_model_from_q_table(_LOCAL_Q_TABLE) if _LOCAL_Q_TABLE else None), 'A', 'local'
    
    if random.random() < 0.5:
        model = load_model_from_firebase(config.get('current_model_version'))
        if model:
            return model, 'A', config.get('current_model_version')
        return (_build_model_from_q_table(_LOCAL_Q_TABLE) if _LOCAL_Q_TABLE else None), 'A', 'local'
    else:
        model = load_model_from_firebase(config.get('challenger_model_version'))
        if model:
            return model, 'B', config.get('challenger_model_version')
        model = load_model_from_firebase(config.get('current_model_version'))
        if model:
            return model, 'A', config.get('current_model_version')
        return (_build_model_from_q_table(_LOCAL_Q_TABLE) if _LOCAL_Q_TABLE else None), 'A', 'local'


def update_ml_config(updates):
    db = _get_db()
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
    db = _get_db()
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

    limit = config.get('ab_test_games_required', AB_TEST_GAMES_REQUIRED)
    
    current_a_games = config.get('model_a_games', 0)
    current_b_games = config.get('model_b_games', 0)
    
    if model_id == 'A':
        if current_a_games < limit:
            updates['model_a_games'] = current_a_games + 1
            if ai_won:
                updates['model_a_wins'] = config.get('model_a_wins', 0) + 1
    elif model_id == 'B':
        if current_b_games < limit:
            updates['model_b_games'] = current_b_games + 1
            if ai_won:
                updates['model_b_wins'] = config.get('model_b_wins', 0) + 1
    
    update_ml_config(updates)
    
    final_a_games = updates.get('model_a_games', current_a_games)
    final_b_games = updates.get('model_b_games', current_b_games)
    
    if final_a_games >= limit and final_b_games >= limit:
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


SESSION_EXPIRY_HOURS = 24

def cleanup_old_sessions():
    db = _get_db()
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


def track_unique_visitor(user_id):
    db = _get_db()
    if not db or not user_id or user_id == 'anonymous':
        return
    
    try:
        from firebase_admin import firestore as _fs
        doc_ref = db.collection('unique_visitors').document(user_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            doc_ref.set({
                'user_id': user_id,
                'first_seen': _fs.SERVER_TIMESTAMP,
                'last_seen': _fs.SERVER_TIMESTAMP,
                'games_played': 1
            })
        else:
            doc_ref.update({
                'last_seen': _fs.SERVER_TIMESTAMP,
                'games_played': _fs.Increment(1)
            })
    except Exception as e:
        print(f"Error tracking visitor: {e}")


def create_game_session(user_id):
    db = _get_db()
    if not db:
        return None
    
    if random.random() < 0.1:
        cleanup_old_sessions()
    
    track_unique_visitor(user_id)
    
    session_id = str(uuid.uuid4())

    model, model_id, model_version = get_local_model()

    try:
        from firebase_admin import firestore as _fs
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
            'created_at': _fs.SERVER_TIMESTAMP,
            'updated_at': _fs.SERVER_TIMESTAMP
        }
        db.collection('game_sessions').document(session_id).set(session_data)
        return session_id
    except Exception as e:
        print(f"Error creating session: {e}")
        return None


def get_game_session(session_id):
    db = _get_db()
    if not db or not session_id:
        return None
    
    try:
        doc = db.collection('game_sessions').document(session_id).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        print(f"Error getting session: {e}")
        return None


def update_game_session(session_id, updates):
    db = _get_db()
    if not db or not session_id:
        return False
    
    try:
        from firebase_admin import firestore as _fs
        updates['updated_at'] = _fs.SERVER_TIMESTAMP
        db.collection('game_sessions').document(session_id).update(updates)
        return True
    except Exception as e:
        print(f"Error updating session: {e}")
        return False



def _log_game_over_async(session, session_id, ai_won, new_player_moves, new_ai_moves):
    def _do_log():
        try:
            db = _get_db()
            if not db:
                return
            
            from firebase_admin import firestore as _fs

            db.collection('ai_vs_human_matches').add({
                'winner': 'ai' if ai_won else 'human',
                'turns': len(new_player_moves),
                'player_moves': new_player_moves,
                'ai_moves': new_ai_moves,
                'model_id': session.get('model_id', 'A'),
                'model_version': session.get('model_version', 'local'),
                'user_id': session.get('user_id', 'unknown'),
                'session_id': session_id,
                'timestamp': _fs.SERVER_TIMESTAMP
            })

            config = get_ml_config()
            record_game_result_and_check_ab(session.get('model_id', 'A'), ai_won, config)
        except Exception as e:
            print(f"[game_session] Async log error: {e}")

    threading.Thread(target=_do_log, daemon=True).start()


def process_move(session_id, player_move):
    session = get_game_session(session_id)
    player_charges = session.get('player_charges', 0)

    model, model_id, model_version = get_local_model()
    model.move_history = session.get('ai_moves', [])[-4:]
    model.opp_move_history = session.get('player_moves', [])[-4:]
    
    ai_charges = session.get('ai_charges', 0)
    ai_legal_moves = model.get_legal_moves(ai_charges)
    ai_state = model.get_state(ai_charges, player_charges)
    ai_move = model.choose_action(ai_state, ai_legal_moves, training=False)
    
    game = FireballGame()
    game.player_charges = player_charges
    game.comp_charges = ai_charges
    result = game.execute_turn(player_move, ai_move)
    
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
    
    if game.game_over:
        ai_won = game.winner == 'player2'
        _log_game_over_async(session, session_id, ai_won, new_player_moves, new_ai_moves)
    
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
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            action = params.get('action', [''])[0]
            
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len) if content_len > 0 else b'{}'
            data = json.loads(post_body) if post_body else {}
            
            response = {}
            
            if action == 'start':
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
                response = process_move(data.get('sessionId'), data.get('move'))
                    
            elif action == 'status':
                session = get_game_session(data.get('sessionId'))
                response = {
                    'playerCharges': session.get('player_charges', 0),
                    'aiCharges': session.get('ai_charges', 0),
                    'gameOver': session.get('game_over', False),
                    'winner': session.get('winner'),
                    'turn': session.get('turn', 0)
                }
            
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

