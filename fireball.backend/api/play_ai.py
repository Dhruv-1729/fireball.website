import json
import os
import firebase_admin
from firebase_admin import credentials, firestore
from http.server import BaseHTTPRequestHandler
from shared_logic.game import FireballQLearning, FireballGame

# Initialize Firebase for game logging
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

# Load the AI Model ONCE when the serverless function starts
ai_player = FireballQLearning(epsilon=0)
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
model_loaded = ai_player.load_model(model_path)


class handler(BaseHTTPRequestHandler):
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
            
            # Re-create game state from frontend
            game = FireballGame()
            game.player_charges = int(data.get('playerCharges', 0))
            game.comp_charges = int(data.get('aiCharges', 0))
            
            # Re-create AI history state
            ai_player.move_history = data.get('oppMoveHistory', [])
            ai_player.opp_move_history = data.get('moveHistory', [])

            # AI chooses its move
            ai_legal_moves = ai_player.get_legal_moves(game.comp_charges)
            ai_state = ai_player.get_state(game.comp_charges, game.player_charges)
            ai_move = ai_player.choose_action(ai_state, ai_legal_moves, training=False)
            
            # Execute the turn
            result = game.execute_turn(player_move, ai_move)
            
            # Log game data to Firebase for continued learning
            if db:
                try:
                    move_history = data.get('moveHistory', []) + [player_move]
                    ai_history = data.get('oppMoveHistory', []) + [ai_move]
                    
                    # Log each turn for training data
                    db.collection('ai_game_turns').add({
                        'player_charges_before': data.get('playerCharges', 0),
                        'ai_charges_before': data.get('aiCharges', 0),
                        'player_move': player_move,
                        'ai_move': ai_move,
                        'result': result,
                        'turn_number': len(move_history),
                        'timestamp': firestore.SERVER_TIMESTAMP
                    })
                    
                    # Log completed games for win/loss stats
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

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()