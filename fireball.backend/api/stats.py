import json
import os
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

try:
    db = firestore.client()
except:
    db = None


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if not db:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Database not connected"}).encode())
            return

        try:
            # Get Unique Visitors
            try:
                visitors_ref = db.collection("unique_visitors").stream()
                unique_visitor_count = len(list(visitors_ref))
            except:
                unique_visitor_count = 0

            # Get AI vs Human Stats
            try:
                matches_ref = db.collection("ai_vs_human_matches").stream()
                matches = [doc.to_dict() for doc in matches_ref]

                total_games = len(matches)
                ai_wins = sum(1 for m in matches if m.get('winner') == 'ai')
                total_turns = sum(m.get('turns', 0) for m in matches)

                win_rate = (ai_wins / total_games) * 100 if total_games > 0 else 0
                avg_length = total_turns / total_games if total_games > 0 else 0
            except:
                total_games = 0
                win_rate = 0
                avg_length = 0

            # Get Recent 1v1 Matches
            recent_matches_list = []
            try:
                matches_1v1_ref = db.collection("matches").where("status", "==", "finished").limit(5).stream()

                for doc in matches_1v1_ref:
                    match = doc.to_dict()
                    winner = match.get('winner', '')
                    winner_name = match.get('player1_username') if winner == 'player1' else match.get('player2_username', 'Unknown')

                    recent_matches_list.append({
                        "winner": winner_name,
                        "rounds": match.get('turn', 1),
                        "p1": match.get('player1_username', 'P1'),
                        "p2": match.get('player2_username', 'P2'),
                    })
            except:
                pass

            stats = {
                "uniqueVisitors": unique_visitor_count,
                "aiWinRate": round(win_rate, 2),
                "aiGames": total_games,
                "avgMatchLength": round(avg_length, 2),
                "recentMatches": recent_matches_list
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(stats).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())