import json
import os
import hashlib
from datetime import datetime, timezone, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
from http.server import BaseHTTPRequestHandler

# PST timezone (UTC-8)
PST = timezone(timedelta(hours=-8))

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


def verify_admin_token(token):
    """Verify an admin session token."""
    if not db or not token:
        return False
    
    try:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        doc = db.collection('admin_sessions').document(token_hash).get()
        
        if not doc.exists:
            return False
        
        session = doc.to_dict()
        
        if not session.get('valid', False):
            return False
        
        # Check expiry
        expires_at = session.get('expires_at')
        if expires_at:
            if hasattr(expires_at, 'timestamp'):
                if datetime.now(timezone.utc).timestamp() > expires_at.timestamp():
                    return False
            elif isinstance(expires_at, datetime):
                if datetime.now(timezone.utc) > expires_at.replace(tzinfo=timezone.utc):
                    return False
        
        return True
    except Exception as e:
        print(f"Error verifying token: {e}")
        return False


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def do_GET(self):
        # Verify admin token
        auth_header = self.headers.get('Authorization', '')
        token = auth_header.replace('Bearer ', '') if auth_header.startswith('Bearer ') else ''
        
        if not verify_admin_token(token):
            self.send_response(401)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Unauthorized. Valid admin token required."}).encode())
            return
        
        if not db:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Database not connected"}).encode())
            return

        try:
            # Cutoff date: January 1, 2026 00:00:00 UTC
            cutoff_date = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            
            # Get Unique Visitors (efficiently) - no date filter as we want total unique
            try:
                unique_visitor_count = db.collection("unique_visitors").count().get()[0][0].value
            except:
                unique_visitor_count = 0

            # Get AI vs Human Games with full details (last 100 only, from 2026 onwards)
            ai_games = []
            try:
                matches_ref = db.collection("ai_vs_human_matches").where("timestamp", ">=", cutoff_date).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(100).stream()
                
                for doc in matches_ref:
                    match = doc.to_dict()
                    timestamp = match.get('timestamp')
                    ts_str = ''
                    if timestamp:
                        try:
                            # Convert to PST
                            if hasattr(timestamp, 'astimezone'):
                                pst_time = timestamp.astimezone(PST)
                            else:
                                pst_time = timestamp.replace(tzinfo=timezone.utc).astimezone(PST)
                            ts_str = pst_time.strftime('%Y-%m-%d %H:%M PST')
                        except:
                            ts_str = str(timestamp)
                    
                    ai_games.append({
                        'id': doc.id,
                        'winner': match.get('winner', 'unknown'),
                        'turns': match.get('turns', 0),
                        'timestamp': ts_str,
                        'playerMoves': match.get('player_moves', []),
                        'aiMoves': match.get('ai_moves', []),
                        'userId': match.get('user_id', 'N/A'),
                        'modelId': match.get('model_id', 'A')
                    })
            except Exception as e:
                print(f"AI games error: {e}")

            # Calculate AI stats
            display_ai_games = len(ai_games)
            ai_wins = sum(1 for g in ai_games if g['winner'] == 'ai')
            total_turns = sum(g['turns'] for g in ai_games)
            
            # Get TOTAL count and GLOBAL win rate (efficiently) - only from 2026 onwards
            total_ai_games = "ERROR"
            win_rate = "ERROR"
            try:
                # Try using count() aggregation first (requires index)
                total_ai_games = db.collection("ai_vs_human_matches").where("timestamp", ">=", cutoff_date).count().get()[0][0].value
                total_ai_wins = db.collection("ai_vs_human_matches").where("timestamp", ">=", cutoff_date).where("winner", "==", "ai").count().get()[0][0].value
                win_rate = (total_ai_wins / total_ai_games) * 100 if total_ai_games > 0 else 0
                print(f"Count query success: {total_ai_games} games, {total_ai_wins} wins")
            except Exception as count_err:
                print(f"Count query failed (likely missing index): {count_err}")
                # Fallback: fetch all documents and count in Python (more expensive but works)
                try:
                    all_games_ref = db.collection("ai_vs_human_matches").where("timestamp", ">=", cutoff_date).stream()
                    all_games_list = list(all_games_ref)
                    total_ai_games = len(all_games_list)
                    total_ai_wins = sum(1 for doc in all_games_list if doc.to_dict().get('winner') == 'ai')
                    win_rate = (total_ai_wins / total_ai_games) * 100 if total_ai_games > 0 else 0
                    print(f"Fallback count success: {total_ai_games} games, {total_ai_wins} wins")
                except Exception as fallback_err:
                    print(f"CRITICAL: Both count methods failed: {fallback_err}")
                    # Leave as "ERROR" - don't show misleading data
            
            avg_length = total_turns / display_ai_games if display_ai_games > 0 else 0

            # Get Online/1v1 Games with full details (from 2026 onwards)
            online_games = []
            games_to_delete = []  # Track games with N/A moves to delete
            try:
                # Get finished, terminated or disconnected matches from 2026 onwards
                target_statuses = ["finished", "terminated", "disconnected"]
                matches_ref = db.collection("matches").where("created_at", ">=", cutoff_date).where("status", "in", target_statuses).order_by("created_at", direction=firestore.Query.DESCENDING).limit(100).stream()
                
                # Check if it works (Firestore might complain about composite index for created_at + status)
                try:
                    results = list(matches_ref)
                except Exception as e:
                    print(f"Ordered query failed, falling back to simpler query: {e}")
                    # Fallback: get all from 2026 and filter by status in Python
                    matches_ref = db.collection("matches").where("created_at", ">=", cutoff_date).order_by("created_at", direction=firestore.Query.DESCENDING).limit(200).stream()
                    results = [doc for doc in matches_ref if doc.to_dict().get('status') in target_statuses][:100]

                for doc in results:
                    match = doc.to_dict()
                    
                    # Check if both players have empty/N/A moves
                    p1_moves = match.get('player1_moves', [])
                    p2_moves = match.get('player2_moves', [])
                    
                    # If both are empty/None/N/A, mark for deletion and skip
                    if (not p1_moves or p1_moves == ['N/A'] or p1_moves == []) and \
                       (not p2_moves or p2_moves == ['N/A'] or p2_moves == []):
                        games_to_delete.append(doc.id)
                        continue
                    
                    timestamp = match.get('finished_at') or match.get('terminated_at') or match.get('created_at')
                    ts_str = ''
                    if timestamp:
                        try:
                            # Convert to PST
                            if hasattr(timestamp, 'astimezone'):
                                pst_time = timestamp.astimezone(PST)
                            else:
                                pst_time = timestamp.replace(tzinfo=timezone.utc).astimezone(PST)
                            ts_str = pst_time.strftime('%Y-%m-%d %H:%M PST')
                        except:
                            ts_str = str(timestamp)
                    
                    winner = match.get('winner', match.get('status', 'unknown'))
                    winner_name = winner
                    if winner == 'player1': winner_name = match.get('player1_username', 'P1')
                    elif winner == 'player2': winner_name = match.get('player2_username', 'P2')

                    online_games.append({
                        'id': doc.id,
                        'player1': match.get('player1_username', 'P1'),
                        'player2': match.get('player2_username', 'P2'),
                        'winner': winner_name,
                        'turns': match.get('turn', 1),
                        'timestamp': ts_str,
                        'player1Moves': p1_moves if p1_moves else ['N/A'],
                        'player2Moves': p2_moves if p2_moves else ['N/A'],
                        'status': match.get('status', 'finished')
                    })
                
                # Delete games with N/A moves for both players
                for game_id in games_to_delete:
                    try:
                        db.collection("matches").document(game_id).delete()
                        print(f"Deleted invalid game: {game_id}")
                    except Exception as del_err:
                        print(f"Failed to delete game {game_id}: {del_err}")
                
                print(f"Total online games fetched: {len(online_games)}, deleted {len(games_to_delete)} invalid games")
            except Exception as e:
                print(f"Online games retrieval error: {e}")

            # Get total online games count (excluding deleted ones)
            total_online_count = "ERROR"
            try:
                total_online_count = db.collection("matches").where("created_at", ">=", cutoff_date).where("status", "==", "finished").count().get()[0][0].value
            except Exception as online_count_err:
                print(f"Online count query failed: {online_count_err}")
                # Fallback: fetch and count in Python
                try:
                    online_ref = db.collection("matches").where("created_at", ">=", cutoff_date).where("status", "==", "finished").stream()
                    total_online_count = sum(1 for _ in online_ref)
                except Exception as online_fallback_err:
                    print(f"CRITICAL: Online count fallback failed: {online_fallback_err}")
                    # Leave as "ERROR"

            stats = {
                "uniqueVisitors": unique_visitor_count,
                "aiWinRate": round(win_rate, 2) if isinstance(win_rate, (int, float)) else win_rate,
                "aiGames": total_ai_games,
                "avgMatchLength": round(avg_length, 2),
                "aiGamesList": ai_games,
                "onlineGamesList": online_games,
                "totalOnlineGames": total_online_count
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