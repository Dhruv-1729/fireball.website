import json
import os
from http.server import BaseHTTPRequestHandler
from shared_logic.game import FireballQLearning, FireballGame

# --- Load the AI Model ONCE when the serverless function starts ---
# This is a critical optimization.
ai_player = FireballQLearning(epsilon=0) # Epsilon=0 for pure exploitation
model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
model_loaded = ai_player.load_model(model_path)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if not model_loaded:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "AI model 'fireball_ai_model.pkl' not found or failed to load."}).encode())
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
            ai_player.move_history = data.get('oppMoveHistory', []) # AI's history is opp's history
            ai_player.opp_move_history = data.get('moveHistory', []) # AI's opp_history is player's history

            # 1. AI chooses its move
            ai_legal_moves = ai_player.get_legal_moves(game.comp_charges)
            ai_state = ai_player.get_state(game.comp_charges, game.player_charges)
            ai_move = ai_player.choose_action(ai_state, ai_legal_moves, training=False)
            
            # 2. Execute the turn
            result = game.execute_turn(player_move, ai_move)
            
            # 3. Send back the new state
            response = {
                "playerCharges": game.player_charges,
                "aiCharges": game.comp_charges,
                "aiMove": ai_move,
                "result": result,
                "gameOver": game.game_over,
                "winner": game.winner # 'player1', 'player2', or None
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())