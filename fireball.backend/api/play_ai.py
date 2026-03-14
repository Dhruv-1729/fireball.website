import json
import os
import random
import pickle
import base64
import time
import threading
from collections import defaultdict
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

    @staticmethod
    def _count_streak(history, move):
        """Count consecutive occurrences of `move` from the end of history."""
        streak = 0
        for m in reversed(history):
            if m == move:
                streak += 1
            else:
                break
        return streak

    def _heuristic_action(self, state, legal_moves):
        """Pattern-aware heuristic with anti-exploit logic."""
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

        # 25% aggressive, 75% balanced-but-not-exploitable
        aggressive = random.random() < 0.25

        # === RULE 0: Megaball is an instant win — always use it ===
        if 'megaball' in legal_moves:
            return 'megaball'

        # === RULE 1: Anti charge-spam ===
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

        # === RULE 2: Anti shield-loop ===
        if opp_shield_streak >= 2:
            if my_charges >= 4:
                return 'charge'
            return 'charge'

        # === RULE 3: Break OUR OWN shield loop ===
        if my_shield_streak >= 2 and opp_shield_streak >= 1:
            return 'charge'

        # === RULE 4: Opponent can megaball (5+ charges) ===
        if opp_charges >= 5:
            if 'iceball' in legal_moves:
                return 'iceball'
            if 'fireball' in legal_moves:
                return 'fireball'
            return 'charge'

        # === RULE 5: Opponent has 0 charges — they MUST charge ===
        if opp_charges == 0:
            if 'fireball' in legal_moves:
                prob = 0.85 if aggressive else 0.6
                return 'fireball' if random.random() < prob else 'charge'
            return 'charge'

        # === RULE 6: Opponent just charged last turn (not a streak) ===
        if len(opp_recent) > 0 and opp_recent[-1] == 'charge':
            if 'fireball' in legal_moves:
                prob = 0.7 if aggressive else 0.45
                if random.random() < prob:
                    return 'fireball'

        # === STRATEGIC PLAY (mode-dependent) ===
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
        """Load model from bytes (for Firebase models)."""
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


# ============ LOCAL MODEL LOADING (runs once at module import) ============
# On Vercel, the module is imported once per Lambda container. This code runs
# during cold start and persists across warm invocations (~60s idle timeout).
# Loading the 8.59 MB gzip file from the bundled filesystem is ~2-3s on cold
# start, but subsequent requests reuse the already-loaded _LOCAL_Q_TABLE at
# zero cost.

def _load_local_model():
    """Load the model from the filesystem bundled with the Vercel deployment."""
    model = FireballQLearning(epsilon=0)
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    # Try model.pkl.gz first, then model.pkl
    # Note: model.pkl is itself gzip-compressed (despite the .pkl extension),
    # and load_from_bytes handles the gzip magic byte detection automatically.
    for filename in ('model.pkl.gz', 'model.pkl'):
        path = os.path.join(base_dir, filename)
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model.load_from_bytes(f.read())
                if len(model.q_table) > 10:
                    print(f"[play_ai] Loaded local model from {filename}: {len(model.q_table)} states")
                    return model, True, path
                else:
                    print(f"[play_ai] Model {filename} loaded but only {len(model.q_table)} states — skipping")
        except Exception as e:
            print(f"[play_ai] Error loading {filename}: {e}")
    
    print("[play_ai] WARNING: No local model found, will fall back to Firebase")
    return model, False, None


# Load once at module level — persists across warm invocations
ai_player, _model_loaded, _LOCAL_MODEL_PATH = _load_local_model()
_LOCAL_Q_TABLE = ai_player.q_table if _model_loaded else None

# Default to using the local model. Set PREFER_LOCAL_MODEL=false in Vercel
# env vars ONLY if you want to force Firebase model loading (e.g. A/B testing).
_PREFER_LOCAL_MODEL = os.environ.get('PREFER_LOCAL_MODEL', 'true').lower() not in ('0', 'false', 'no')


# ============ LAZY FIREBASE INIT (only for game logging) ============
# Firebase Admin SDK is heavy (~2-3s to initialize). Since we load the model
# from the local filesystem, we only need Firebase for logging game data.
# Lazy-init means we don't pay for it until the first log write, and even
# then it only happens once per container lifetime.

_db = None
_db_init_lock = threading.Lock()
_firebase_init_done = False


def _get_db():
    """Lazy-initialize Firebase and return a Firestore client."""
    global _db, _firebase_init_done
    if _db is not None:
        return _db
    
    with _db_init_lock:
        # Double-check after acquiring lock
        if _db is not None:
            return _db
        
        if not _firebase_init_done:
            _firebase_init_done = True
            try:
                if not __import__('firebase_admin')._apps:
                    import firebase_admin as _fa
                    from firebase_admin import credentials as _creds
                    sa_info = json.loads(os.environ.get('FIREBASE_SERVICE_ACCOUNT', '{}'))
                    if sa_info:
                        cred = _creds.Certificate(sa_info)
                        _fa.initialize_app(cred)
                    else:
                        print("[play_ai] FIREBASE_SERVICE_ACCOUNT env var not set")
                        return None
            except Exception as e:
                print(f"[play_ai] Firebase Init Error: {e}")
                return None
        
        try:
            from firebase_admin import firestore as _fs
            _db = _fs.client()
        except Exception as e:
            print(f"[play_ai] Firestore client error: {e}")
    
    return _db


# ============ MODEL LOADING FROM FIREBASE (fallback only) ============
# This is only used when PREFER_LOCAL_MODEL is False and we need A/B testing.

_MODEL_CACHE = {}
_MODEL_CACHE_USED_AT = {}
_MODEL_CACHE_MAX = max(1, int(os.environ.get('MODEL_CACHE_MAX', '2')))
_MODEL_CACHE_LOCK = threading.Lock()


def _build_model_from_q_table(q_table):
    model = FireballQLearning(epsilon=0)
    model.q_table = q_table
    return model


def _cache_get(version_name):
    with _MODEL_CACHE_LOCK:
        q_table = _MODEL_CACHE.get(version_name)
        if q_table is not None:
            _MODEL_CACHE_USED_AT[version_name] = time.time()
        return q_table


def _cache_put(version_name, q_table):
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[version_name] = q_table
        _MODEL_CACHE_USED_AT[version_name] = time.time()
        if len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
            oldest = min(_MODEL_CACHE_USED_AT, key=_MODEL_CACHE_USED_AT.get)
            if oldest != version_name:
                _MODEL_CACHE.pop(oldest, None)
                _MODEL_CACHE_USED_AT.pop(oldest, None)


# A/B Testing thresholds
AB_TEST_GAMES_REQUIRED = 15


def get_ml_config():
    """Get ML configuration from Firebase."""
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


def update_ml_config(updates):
    """Update ML configuration in Firebase."""
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


def load_model_from_firebase(version_name):
    """Load a model from Firebase (fallback only)."""
    db = _get_db()
    if not db or not version_name:
        return None
    cached_q_table = _cache_get(version_name)
    if cached_q_table is not None:
        return _build_model_from_q_table(cached_q_table)
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
            print(f"Warning: Model {version_name} has only {len(model.q_table)} states, possibly corrupted")
            return None
        
        _cache_put(version_name, model.q_table)
        return _build_model_from_q_table(model.q_table)
    except Exception as e:
        print(f"Error loading model from Firebase: {e}")
        return None


def get_model_for_game():
    """
    Get the appropriate model for a game, handling A/B testing.
    Returns (model, model_id, version_name) tuple.
    
    FAST PATH: If the local model is loaded (the common case), return it
    immediately with zero network calls. This is the key optimization —
    warm invocations skip Firebase entirely.
    """
    # ── Fast path: local model (no Firebase, no network) ──
    if _PREFER_LOCAL_MODEL and _LOCAL_Q_TABLE:
        return _build_model_from_q_table(_LOCAL_Q_TABLE), 'A', 'local'
    
    # ── Slow path: Firebase models (for A/B testing scenarios) ──
    config = get_ml_config()
    
    if not config:
        return (_build_model_from_q_table(_LOCAL_Q_TABLE) if _LOCAL_Q_TABLE else None), 'A', 'local'
    
    # If A/B test is not active, use current model
    if not config.get('ab_test_active') or not config.get('challenger_model_version'):
        current_version = config.get('current_model_version')
        if current_version:
            model = load_model_from_firebase(current_version)
            if model:
                return model, 'A', current_version
        return (_build_model_from_q_table(_LOCAL_Q_TABLE) if _LOCAL_Q_TABLE else None), 'A', 'local'
    
    # A/B test is active - randomly assign
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


def delete_model_from_firebase(version_name):
    """Delete a model from Firebase."""
    db = _get_db()
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
    """
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
    """Conclude A/B test and promote winner."""
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


# ============ NON-BLOCKING GAME LOGGING ============

def _log_game_data_async(turn_data, match_data=None, model_id=None, config=None, ai_won=None):
    """
    Fire-and-forget game logging in a background thread.
    This prevents Firestore writes from blocking the API response.
    """
    def _do_log():
        try:
            db = _get_db()
            if not db:
                return
            
            from firebase_admin import firestore as _fs
            
            # Log the turn
            if turn_data:
                turn_data['timestamp'] = _fs.SERVER_TIMESTAMP
                db.collection('ai_game_turns').add(turn_data)
            
            # Log the match result
            if match_data:
                match_data['timestamp'] = _fs.SERVER_TIMESTAMP
                db.collection('ai_vs_human_matches').add(match_data)
                
                # Record for A/B testing
                if config is not None and ai_won is not None and model_id:
                    record_game_result_and_check_ab(model_id, ai_won, config)
        except Exception as e:
            print(f"[play_ai] Async log error: {e}")
    
    threading.Thread(target=_do_log, daemon=True).start()


# ============ HTTP HANDLER ============

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

            # Get model for this game (fast path: local model, ~0ms)
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

            # Build log data and fire-and-forget in background thread
            move_history = data.get('moveHistory', []) + [player_move]
            ai_history = data.get('oppMoveHistory', []) + [ai_move]

            turn_data = {
                'player_charges_before': data.get('playerCharges', 0),
                'ai_charges_before': data.get('aiCharges', 0),
                'player_move': player_move,
                'ai_move': ai_move,
                'result': result,
                'turn_number': len(move_history),
                'model_id': model_id,
                'model_version': model_version,
                'user_id': user_id,
            }

            match_data = None
            ab_config = None
            ai_won = None
            if game.game_over:
                ai_won = game.winner == 'player2'
                match_data = {
                    'winner': 'ai' if ai_won else 'human',
                    'turns': len(move_history),
                    'player_moves': move_history,
                    'ai_moves': ai_history,
                    'model_id': model_id,
                    'model_version': model_version,
                    'user_id': user_id,
                }
                # Only fetch config for A/B test tracking — this is deferred
                ab_config = 'deferred'

            # Log asynchronously — don't block the response
            def _deferred_log():
                cfg = None
                if ab_config == 'deferred':
                    cfg = get_ml_config()
                _log_game_data_async(turn_data, match_data, model_id, cfg, ai_won)
            
            threading.Thread(target=_deferred_log, daemon=True).start()

            response = {
                "playerCharges": game.player_charges,
                "aiCharges": game.comp_charges,
                "aiMove": ai_move,
                "result": result,
                "gameOver": game.game_over,
                "winner": game.winner,
                "modelId": model_id
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
