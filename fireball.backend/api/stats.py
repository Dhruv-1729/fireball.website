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

            # Get AI vs Human Games with full details
            ai_games = []
            try:
                matches_ref = db.collection("ai_vs_human_matches").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(100).stream()
                
                for doc in matches_ref:
                    match = doc.to_dict()
                    timestamp = match.get('timestamp')
                    ts_str = ''
                    if timestamp:
                        try:
                            ts_str = timestamp.strftime('%Y-%m-%d %H:%M')
                        except:
                            ts_str = str(timestamp)
                    
                    ai_games.append({
                        'id': doc.id,
                        'winner': match.get('winner', 'unknown'),
                        'turns': match.get('turns', 0),
                        'timestamp': ts_str,
                        'playerMoves': match.get('player_moves', []),
                        'aiMoves': match.get('ai_moves', [])
                    })
            except Exception as e:
                print(f"AI games error: {e}")

            # Calculate AI stats
            total_ai_games = len(ai_games)
            ai_wins = sum(1 for g in ai_games if g['winner'] == 'ai')
            total_turns = sum(g['turns'] for g in ai_games)
            win_rate = (ai_wins / total_ai_games) * 100 if total_ai_games > 0 else 0
            avg_length = total_turns / total_ai_games if total_ai_games > 0 else 0

            # Get Online/1v1 Games with full details
            online_games = []
            try:
                # Try multiple query strategies
                matches_1v1_ref = None
                query_error = None
                
                # Strategy 1: Try with finished_at ordering
                try:
                    matches_1v1_ref = db.collection("matches").where("status", "==", "finished").order_by("finished_at", direction=firestore.Query.DESCENDING).limit(100).stream()
                    # Try to get first result to see if query works
                    first_doc = next(matches_1v1_ref, None)
                    if first_doc:
                        # Reset iterator and process
                        matches_1v1_ref = db.collection("matches").where("status", "==", "finished").order_by("finished_at", direction=firestore.Query.DESCENDING).limit(100).stream()
                    else:
                        print("finished_at query returned no results")
                        matches_1v1_ref = None
                except Exception as e1:
                    query_error = str(e1)
                    print(f"finished_at query failed: {e1}")
                    matches_1v1_ref = None
                
                # Strategy 2: Fallback to just status filter, no ordering
                if not matches_1v1_ref:
                    try:
                        matches_1v1_ref = db.collection("matches").where("status", "==", "finished").limit(100).stream()
                        print(f"Using fallback query (status only). Previous error: {query_error}")
                    except Exception as e2:
                        print(f"Fallback query also failed: {e2}")
                        # Strategy 3: Last resort - get ALL matches and filter in Python
                        try:
                            all_matches = db.collection("matches").limit(200).stream()
                            for doc in all_matches:
                                match = doc.to_dict()
                                if match.get('status') == 'finished':
                                    timestamp = match.get('finished_at') or match.get('created_at')
                                    ts_str = ''
                                    if timestamp:
                                        try:
                                            ts_str = timestamp.strftime('%Y-%m-%d %H:%M')
                                        except:
                                            ts_str = str(timestamp)
                                    
                                    winner_name = match.get('player1_username') if match.get('winner') == 'player1' else match.get('player2_username', 'Unknown')
                                    
                                    online_games.append({
                                        'id': doc.id,
                                        'player1': match.get('player1_username', 'P1'),
                                        'player2': match.get('player2_username', 'P2'),
                                        'winner': winner_name,
                                        'turns': match.get('turn', 1),
                                        'timestamp': ts_str,
                                        'player1Moves': match.get('player1_moves', []),
                                        'player2Moves': match.get('player2_moves', [])
                                    })
                            print(f"Last resort query found {len(online_games)} finished matches")
                        except Exception as e3:
                            print(f"Even last resort failed: {e3}")
                
                # Process results from Strategy 1 or 2
                if matches_1v1_ref and len(online_games) == 0:
                    for doc in matches_1v1_ref:
                        match = doc.to_dict()
                        
                        # Try to get finished_at, fallback to created_at
                        timestamp = match.get('finished_at') or match.get('created_at')
                        ts_str = ''
                        if timestamp:
                            try:
                                ts_str = timestamp.strftime('%Y-%m-%d %H:%M')
                            except:
                                ts_str = str(timestamp)
                        
                        winner_name = match.get('player1_username') if match.get('winner') == 'player1' else match.get('player2_username', 'Unknown')
                        
                        online_games.append({
                            'id': doc.id,
                            'player1': match.get('player1_username', 'P1'),
                            'player2': match.get('player2_username', 'P2'),
                            'winner': winner_name,
                            'turns': match.get('turn', 1),
                            'timestamp': ts_str,
                            'player1Moves': match.get('player1_moves', []),
                            'player2Moves': match.get('player2_moves', [])
                        })
                
                print(f"Total online games found: {len(online_games)}")
            except Exception as e:
                print(f"Online games outer error: {e}")

            stats = {
                "uniqueVisitors": unique_visitor_count,
                "aiWinRate": round(win_rate, 2),
                "aiGames": total_ai_games,
                "avgMatchLength": round(avg_length, 2),
                "aiGamesList": ai_games,
                "onlineGamesList": online_games,
                "totalOnlineGames": len(online_games)
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