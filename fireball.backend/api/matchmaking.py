import json
import os
import time
import firebase_admin
from firebase_admin import credentials, firestore
from http.server import BaseHTTPRequestHandler

# Initialize Firebase
if not firebase_admin._apps:
    try:
        service_account_info = json.loads(os.environ.get('FIREBASE_SERVICE_ACCOUNT', '{}'))
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        print(f"Firebase Init Error: {e}")

db = firestore.client()

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len)
            data = json.loads(post_body)
            action = data.get('action')

            if action == 'find_match':
                result = self.find_match(data)
            elif action == 'check_match':
                result = self.check_match(data)
            elif action == 'submit_move':
                result = self.submit_move(data)
            elif action == 'get_game_state':
                result = self.get_game_state(data)
            elif action == 'leave_queue':
                result = self.leave_queue(data)
            else:
                result = {'error': 'Unknown action'}

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def find_match(self, data):
        username = data.get('username', 'Anonymous')
        player_id = data.get('playerId')
        
        queue_ref = db.collection('matchmaking_queue')
        
        # Check if already in queue
        existing = queue_ref.where('playerId', '==', player_id).limit(1).stream()
        for doc in existing:
            return {'status': 'waiting', 'queueId': doc.id}
        
        # Look for an opponent
        waiting_players = queue_ref.where('status', '==', 'waiting').order_by('timestamp').limit(1).stream()
        
        for opponent_doc in waiting_players:
            opponent = opponent_doc.to_dict()
            if opponent['playerId'] != player_id:
                # Found opponent - create match
                match_id = f"match_{int(time.time() * 1000)}"
                
                match_data = {
                    'player1': opponent['playerId'],
                    'player2': player_id,
                    'player1_username': opponent['username'],
                    'player2_username': username,
                    'player1_charges': 0,
                    'player2_charges': 0,
                    'player1_move': None,
                    'player2_move': None,
                    'turn': 1,
                    'status': 'active',
                    'winner': None,
                    'created_at': firestore.SERVER_TIMESTAMP
                }
                
                db.collection('matches').document(match_id).set(match_data)
                
                # Update opponent's queue entry
                opponent_doc.reference.update({
                    'status': 'matched',
                    'matchId': match_id
                })
                
                return {'status': 'matched', 'matchId': match_id, 'opponent': opponent['username']}
        
        # No opponent found - add to queue
        queue_entry = {
            'playerId': player_id,
            'username': username,
            'status': 'waiting',
            'timestamp': firestore.SERVER_TIMESTAMP
        }
        
        doc_ref = queue_ref.add(queue_entry)
        return {'status': 'waiting', 'queueId': doc_ref[1].id}

    def check_match(self, data):
        player_id = data.get('playerId')
        
        # Check queue for match
        queue_ref = db.collection('matchmaking_queue')
        queue_entries = queue_ref.where('playerId', '==', player_id).limit(1).stream()
        
        for doc in queue_entries:
            entry = doc.to_dict()
            if entry.get('status') == 'matched':
                match_id = entry.get('matchId')
                doc.reference.delete()  # Clean up queue
                
                match_doc = db.collection('matches').document(match_id).get()
                if match_doc.exists:
                    match = match_doc.to_dict()
                    opponent = match['player1_username'] if match['player2'] == player_id else match['player2_username']
                    return {'status': 'matched', 'matchId': match_id, 'opponent': opponent}
            
            return {'status': 'waiting'}
        
        return {'status': 'not_in_queue'}

    def submit_move(self, data):
        match_id = data.get('matchId')
        player_id = data.get('playerId')
        move = data.get('move')
        
        match_ref = db.collection('matches').document(match_id)
        match_doc = match_ref.get()
        
        if not match_doc.exists:
            return {'error': 'Match not found'}
        
        match = match_doc.to_dict()
        
        if match['player1'] == player_id:
            match_ref.update({'player1_move': move})
        elif match['player2'] == player_id:
            match_ref.update({'player2_move': move})
        else:
            return {'error': 'Not a participant'}
        
        # Check if both moves submitted
        match = match_ref.get().to_dict()
        
        if match['player1_move'] and match['player2_move']:
            result = self.resolve_turn(match_ref, match)
            return {'status': 'resolved', 'result': result}
        
        return {'status': 'waiting_for_opponent'}

    def resolve_turn(self, match_ref, match):
        p1_move = match['player1_move']
        p2_move = match['player2_move']
        p1_charges = match['player1_charges']
        p2_charges = match['player2_charges']
        
        # Apply charge costs
        costs = {'charge': -1, 'fireball': 1, 'iceball': 2, 'megaball': 5, 'shield': 0}
        p1_charges = max(0, p1_charges - costs.get(p1_move, 0))
        p2_charges = max(0, p2_charges - costs.get(p2_move, 0))
        
        # Determine winner
        winner = self.determine_winner(p1_move, p2_move)
        
        update_data = {
            'player1_charges': p1_charges,
            'player2_charges': p2_charges,
            'player1_move': None,
            'player2_move': None,
            'turn': match['turn'] + 1,
            'last_p1_move': p1_move,
            'last_p2_move': p2_move,
            'last_result': winner
        }
        
        if winner in ['player1', 'player2']:
            update_data['status'] = 'finished'
            update_data['winner'] = winner
        
        match_ref.update(update_data)
        
        return {
            'p1_move': p1_move,
            'p2_move': p2_move,
            'winner': winner,
            'p1_charges': p1_charges,
            'p2_charges': p2_charges
        }

    def determine_winner(self, p1, p2):
        if p1 == p2 and p1 != 'megaball':
            return 'continue'
        if p1 == 'megaball':
            return 'player1' if p2 != 'megaball' else 'continue'
        if p2 == 'megaball':
            return 'player2'
        if 'shield' in [p1, p2]:
            return 'continue'
        win_map = {'fireball': ['charge'], 'iceball': ['charge', 'fireball']}
        if p2 in win_map.get(p1, []):
            return 'player1'
        if p1 in win_map.get(p2, []):
            return 'player2'
        return 'continue'

    def get_game_state(self, data):
        match_id = data.get('matchId')
        player_id = data.get('playerId')
        
        match_doc = db.collection('matches').document(match_id).get()
        
        if not match_doc.exists:
            return {'error': 'Match not found'}
        
        match = match_doc.to_dict()
        
        is_player1 = match['player1'] == player_id
        opponent_submitted = (match['player2_move'] is not None) if is_player1 else (match['player1_move'] is not None)
        
        return {
            'myCharges': match['player1_charges'] if is_player1 else match['player2_charges'],
            'opponentCharges': match['player2_charges'] if is_player1 else match['player1_charges'],
            'turn': match['turn'],
            'status': match['status'],
            'winner': match.get('winner'),
            'lastResult': match.get('last_result'),
            'lastMyMove': match.get('last_p1_move') if is_player1 else match.get('last_p2_move'),
            'lastOpponentMove': match.get('last_p2_move') if is_player1 else match.get('last_p1_move'),
            'opponentSubmitted': opponent_submitted,
            'opponent': match['player2_username'] if is_player1 else match['player1_username']
        }

    def leave_queue(self, data):
        player_id = data.get('playerId')
        
        queue_ref = db.collection('matchmaking_queue')
        queue_entries = queue_ref.where('playerId', '==', player_id).stream()
        
        for doc in queue_entries:
            doc.reference.delete()
        
        return {'status': 'left'}
