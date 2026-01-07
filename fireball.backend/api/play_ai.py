import json
import os
import random
import pickle
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
        if state not in self.q_table:
            return random.choice(legal_moves)
        q_vals = {m: self.q_table[state][m] + random.uniform(-0.01, 0.01) for m in legal_moves}
        return max(q_vals, key=q_vals.get)

    def load_model(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(f))
            return True
        except Exception:
            return False


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

# Load the AI Model
ai_player = FireballQLearning(epsilon=0)
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
model_loaded = ai_player.load_model(model_path)


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if not model_loaded:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "AI model not found."}).encode())
            return

        try:
            content_len = int(self.headers.get('Content-Length'))
            post_body = self.rfile.read(content_len)
            data = json.loads(post_body)

            player_move = data.get('playerMove')

            game = FireballGame()
            game.player_charges = int(data.get('playerCharges', 0))
            game.comp_charges = int(data.get('aiCharges', 0))

            ai_player.move_history = data.get('oppMoveHistory', [])
            ai_player.opp_move_history = data.get('moveHistory', [])

            ai_legal_moves = ai_player.get_legal_moves(game.comp_charges)
            ai_state = ai_player.get_state(game.comp_charges, game.player_charges)
            ai_move = ai_player.choose_action(ai_state, ai_legal_moves, training=False)

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
                        'timestamp': firestore.SERVER_TIMESTAMP
                    })

                    if game.game_over:
                        db.collection('ai_vs_human_matches').add({
                            'winner': 'ai' if game.winner == 'player2' else 'human',
                            'turns': len(move_history),
                            'player_moves': move_history,
                            'ai_moves': ai_history,
                            'timestamp': firestore.SERVER_TIMESTAMP
                        })
                except Exception as log_err:
                    print(f"Logging error: {log_err}")

            response = {
                "playerCharges": game.player_charges,
                "aiCharges": game.comp_charges,
                "aiMove": ai_move,
                "result": result,
                "gameOver": game.game_over,
                "winner": game.winner
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