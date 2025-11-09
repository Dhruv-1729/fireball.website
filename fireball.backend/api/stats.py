import json
import os
import firebase_admin
from firebase_admin import credentials, firestore
from http.server import BaseHTTPRequestHandler
from shared_logic.game import FireballQLearning, FireballGame

# --- Initialize Firebase ---
# It checks if it's already initialized to avoid errors on warm starts
if not firebase_admin._apps:
    try:
        # Get the service account JSON from the Vercel Environment Variable
        service_account_info = json.loads(os.environ.get('FIREBASE_SERVICE_ACCOUNT'))
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        print(f"Firebase Init Error: {e}")

db = firestore.client()

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Porting your logic from page_statistics()
            # 1. Get Unique Visitors
            visitors_ref = db.collection("unique_visitors").stream()
            unique_visitor_count = len(list(visitors_ref))
            
            # 2. Get AI vs Human Stats
            matches_ref = db.collection("ai_vs_human_matches").stream()
            matches = [doc.to_dict() for doc in matches_ref]
            
            total_games = len(matches)
            ai_wins = sum(1 for m in matches if m['winner'] == 'ai')
            total_turns = sum(m.get('turns', 0) for m in matches)
            
            win_rate = (ai_wins / total_games) * 100 if total_games > 0 else 0
            avg_length = total_turns / total_games if total_games > 0 else 0
            
            # 3. Get Recent 1v1 Matches
            matches_1v1_ref = db.collection("matches") \
                                .where("game_over", "==", True) \
                                .order_by("end_timestamp", direction=firestore.Query.DESCENDING) \
                                .limit(5) \
                                .stream()
            
            recent_matches_list = []
            for doc in matches_1v1_ref:
                match = doc.to_dict()
                usernames = match.get('player_usernames', {})
                winner_id = match.get('winner')
                winner_username = usernames.get(winner_id, "Draw")
                
                p1_id = match['players'][0]
                p2_id = match['players'][1]
                
                recent_matches_list.append({
                    "winner": winner_username,
                    "rounds": match.get('game_state', {}).get('turn', 1) - 1,
                    "p1": usernames.get(p1_id, "P1"),
                    "p2": usernames.get(p2_id, "P2"),
                })

            # Compile response
            stats = {
                "uniqueVisitors": unique_visitor_count,
                "aiWinRate": round(win_rate, 2),
                "aiGames": total_games,
                "avgMatchLength": round(avg_length, 2),
                "recentMatches": recent_matches_list
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())