"""
ML Management API
=================
Provides endpoints for:
- Checking ML training status
- Uploading initial model to Firebase
- Triggering training (when enabled)
- Viewing A/B test results
"""

import json
import os
import pickle
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from http.server import BaseHTTPRequestHandler
from collections import defaultdict

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

# ============ CONFIG CONSTANTS ============
TRAINING_ENABLED = True  # Master switch
GAMES_THRESHOLD_FOR_TRAINING = 200
AB_TEST_GAMES_REQUIRED = 25


def get_ml_config():
    """Get ML configuration from Firebase."""
    if not db:
        return None
    try:
        config_ref = db.collection('ml_config').document('main')
        doc = config_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            # Initialize default config
            default_config = {
                'training_enabled': TRAINING_ENABLED,
                'games_since_last_training': 0,
                'last_training_timestamp': None,
                'current_model_version': None,
                'challenger_model_version': None,
                'model_a_games': 0,
                'model_a_wins': 0,
                'model_b_games': 0,
                'model_b_wins': 0,
                'ab_test_active': False,
                'games_threshold': GAMES_THRESHOLD_FOR_TRAINING,
                'ab_test_games_required': AB_TEST_GAMES_REQUIRED
            }
            config_ref.set(default_config)
            return default_config
    except Exception as e:
        print(f"Error getting ML config: {e}")
        return None


def update_ml_config(updates):
    """Update ML configuration in Firebase."""
    if not db:
        return False
    try:
        config_ref = db.collection('ml_config').document('main')
        config_ref.update(updates)
        return True
    except Exception as e:
        print(f"Error updating ML config: {e}")
        return False


def save_model_to_firebase(model_data, version_name):
    """Save model bytes to Firebase as base64."""
    if not db:
        return False
    try:
        model_b64 = base64.b64encode(model_data).decode('utf-8')
        db.collection('ml_models').document(version_name).set({
            'model_data': model_b64,
            'created_at': firestore.SERVER_TIMESTAMP,
            'version': version_name
        })
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def upload_local_model():
    """Upload the local model.pkl to Firebase as v1_original."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
    try:
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        if save_model_to_firebase(model_data, 'v1_original'):
            update_ml_config({'current_model_version': 'v1_original'})
            return True, "Model uploaded successfully as v1_original"
        return False, "Failed to save to Firebase"
    except FileNotFoundError:
        return False, "model.pkl not found"
    except Exception as e:
        return False, str(e)


def get_training_data_stats():
    """Get statistics about available training data."""
    if not db:
        return {'error': 'Database not connected'}
    
    try:
        from datetime import datetime, timezone
        # Cutoff date: January 1, 2026 00:00:00 UTC
        cutoff_date = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        # Count AI vs Human games from 2026 onwards (single field query - no composite index needed)
        ai_game_count = 0
        ai_wins = 0
        try:
            ai_games = list(db.collection('ai_vs_human_matches').where('timestamp', '>=', cutoff_date).stream())
            ai_game_count = len(ai_games)
            ai_wins = sum(1 for g in ai_games if g.to_dict().get('winner') == 'ai')
        except Exception as ai_err:
            print(f"AI games query failed: {ai_err}")
            # Fallback: get all and filter in Python
            try:
                all_ai_games = list(db.collection('ai_vs_human_matches').stream())
                for doc in all_ai_games:
                    game = doc.to_dict()
                    ts = game.get('timestamp')
                    if ts:
                        try:
                            if hasattr(ts, 'timestamp'):
                                if ts.timestamp() >= cutoff_date.timestamp():
                                    ai_game_count += 1
                                    if game.get('winner') == 'ai':
                                        ai_wins += 1
                            elif hasattr(ts, 'replace'):
                                ts_utc = ts.replace(tzinfo=timezone.utc)
                                if ts_utc >= cutoff_date:
                                    ai_game_count += 1
                                    if game.get('winner') == 'ai':
                                        ai_wins += 1
                        except:
                            pass
            except Exception as fallback_err:
                print(f"AI games fallback also failed: {fallback_err}")
        
        # Count online games - avoid compound query, filter status in Python
        online_game_count = 0
        try:
            # First try single-field query on created_at
            online_games = list(db.collection('matches').where('created_at', '>=', cutoff_date).stream())
            # Filter by status in Python to avoid composite index requirement
            online_game_count = sum(1 for doc in online_games if doc.to_dict().get('status') == 'finished')
        except Exception as online_err:
            print(f"Online games query failed: {online_err}")
            # Fallback: get all and filter in Python
            try:
                all_matches = list(db.collection('matches').stream())
                for doc in all_matches:
                    match = doc.to_dict()
                    if match.get('status') != 'finished':
                        continue
                    ts = match.get('created_at')
                    if ts:
                        try:
                            if hasattr(ts, 'timestamp'):
                                if ts.timestamp() >= cutoff_date.timestamp():
                                    online_game_count += 1
                            elif hasattr(ts, 'replace'):
                                ts_utc = ts.replace(tzinfo=timezone.utc)
                                if ts_utc >= cutoff_date:
                                    online_game_count += 1
                        except:
                            pass
            except Exception as fallback_err:
                print(f"Online games fallback also failed: {fallback_err}")
        
        return {
            'ai_vs_human_games': ai_game_count,
            'ai_win_rate': round(ai_wins / ai_game_count * 100, 1) if ai_game_count > 0 else 0,
            'online_games': online_game_count,
            'total_games': ai_game_count + online_game_count
        }
    except Exception as e:
        print(f"CRITICAL: Error getting training stats: {e}")
        return {
            'ai_vs_human_games': 'ERROR',
            'ai_win_rate': 'ERROR',
            'online_games': 'ERROR',
            'total_games': 'ERROR',
            'error': str(e)
        }


def get_historical_win_rates(days=180):
    """Get historical win rate data aggregated by date for charting."""
    if not db:
        return []
    
    try:
        from datetime import datetime, timezone, timedelta
        
        # Cutoff date: January 1, 2026 00:00:00 UTC
        cutoff_date = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        
        # Get all AI games from Jan 1, 2026 onwards
        ai_games = list(db.collection('ai_vs_human_matches').where('timestamp', '>=', cutoff_date).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(2000).stream())
        
        # Group games by date
        date_stats = {}
        for doc in ai_games:
            game = doc.to_dict()
            timestamp = game.get('timestamp')
            if not timestamp:
                continue
            
            try:
                # Get the date string
                if hasattr(timestamp, 'date'):
                    date_key = timestamp.date().isoformat()
                else:
                    date_key = str(timestamp)[:10]
                
                if date_key not in date_stats:
                    date_stats[date_key] = {'total': 0, 'wins': 0}
                
                date_stats[date_key]['total'] += 1
                if game.get('winner') == 'ai':
                    date_stats[date_key]['wins'] += 1
            except:
                continue
        
        # Convert to list sorted by date
        result = []
        for date_str in sorted(date_stats.keys()):
            stats = date_stats[date_str]
            win_rate = round((stats['wins'] / stats['total']) * 100, 1) if stats['total'] > 0 else 0
            result.append({
                'date': date_str,
                'total': stats['total'],
                'wins': stats['wins'],
                'winRate': win_rate
            })
        
        return result
    except Exception as e:
        print(f"CRITICAL: Error getting historical win rates: {e}")
        return [{'error': str(e)}]


class handler(BaseHTTPRequestHandler):
    def verify_admin_token(self):
        """Verify the admin token from Authorization header."""
        import hashlib
        from datetime import datetime, timezone
        
        auth_header = self.headers.get('Authorization', '')
        token = auth_header.replace('Bearer ', '') if auth_header.startswith('Bearer ') else ''
        
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

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def do_GET(self):
        """GET request returns current ML status."""
        # Verify admin token
        if not self.verify_admin_token():
            self.send_response(401)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Unauthorized. Valid admin token required.'}).encode())
            return
        
        try:
            config = get_ml_config()
            training_stats = get_training_data_stats()
            historical_rates = get_historical_win_rates(days=180)
            
            response = {
                'config': {
                    'training_enabled': config.get('training_enabled', False) if config else False,
                    'games_since_last_training': config.get('games_since_last_training', 0) if config else 0,
                    'games_threshold': config.get('games_threshold', GAMES_THRESHOLD_FOR_TRAINING) if config else GAMES_THRESHOLD_FOR_TRAINING,
                    'current_model': config.get('current_model_version') if config else None,
                    'challenger_model': config.get('challenger_model_version') if config else None,
                    'ab_test_active': config.get('ab_test_active', False) if config else False,
                    'ab_test_games_required': config.get('ab_test_games_required', AB_TEST_GAMES_REQUIRED) if config else AB_TEST_GAMES_REQUIRED
                },
                'ab_test': {
                    'model_a_games': config.get('model_a_games', 0) if config else 0,
                    'model_a_wins': config.get('model_a_wins', 0) if config else 0,
                    'model_b_games': config.get('model_b_games', 0) if config else 0,
                    'model_b_wins': config.get('model_b_wins', 0) if config else 0
                },
                'training_data': training_stats,
                'master_switch': TRAINING_ENABLED,
                'historical_win_rates': historical_rates
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
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def do_POST(self):
        """POST request for actions."""
        # Verify admin token
        if not self.verify_admin_token():
            self.send_response(401)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Unauthorized. Valid admin token required.'}).encode())
            return
        
        try:
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len) if content_len > 0 else b'{}'
            data = json.loads(post_body) if post_body else {}
            
            action = data.get('action')
            
            if action == 'upload_original':
                # Upload local model.pkl to Firebase
                success, message = upload_local_model()
                response = {'success': success, 'message': message}
                
            elif action == 'enable_training':
                if not TRAINING_ENABLED:
                    response = {'success': False, 'message': 'Training is disabled by master switch in code'}
                else:
                    update_ml_config({'training_enabled': True})
                    response = {'success': True, 'message': 'Training enabled'}
                    
            elif action == 'disable_training':
                update_ml_config({'training_enabled': False})
                response = {'success': True, 'message': 'Training disabled'}
                
            elif action == 'reset_game_counter':
                update_ml_config({'games_since_last_training': 0})
                response = {'success': True, 'message': 'Game counter reset'}
                
            elif action == 'cancel_ab_test':
                config = get_ml_config()
                challenger = config.get('challenger_model_version') if config else None
                if challenger:
                    # Delete challenger model
                    db.collection('ml_models').document(challenger).delete()
                update_ml_config({
                    'challenger_model_version': None,
                    'ab_test_active': False,
                    'model_a_games': 0,
                    'model_a_wins': 0,
                    'model_b_games': 0,
                    'model_b_wins': 0
                })
                response = {'success': True, 'message': 'A/B test cancelled'}
            
            elif action == 'manually_conclude_ab_test':
                # Manually stop A/B test and choose winner
                winner = data.get('winner')  # 'A' or 'B'
                
                if winner not in ['A', 'B']:
                    response = {'success': False, 'error': 'Winner must be "A" or "B"'}
                else:
                    config = get_ml_config()
                    if not config or not config.get('ab_test_active'):
                        response = {'success': False, 'error': 'No active A/B test'}
                    else:
                        current_version = config.get('current_model_version')
                        challenger_version = config.get('challenger_model_version')
                        
                        if winner == 'B':
                            # Promote challenger
                            if current_version and current_version != 'v1_original':
                                db.collection('ml_models').document(current_version).delete()
                            
                            update_ml_config({
                                'current_model_version': challenger_version,
                                'challenger_model_version': None,
                                'ab_test_active': False,
                                'model_a_games': 0,
                                'model_a_wins': 0,
                                'model_b_games': 0,
                                'model_b_wins': 0
                            })
                            response = {
                                'success': True, 
                                'message': f'Model B ({challenger_version}) promoted to current model',
                                'winner': 'B'
                            }
                        else:  # winner == 'A'
                            # Keep current, delete challenger
                            if challenger_version:
                                db.collection('ml_models').document(challenger_version).delete()
                            
                            update_ml_config({
                                'challenger_model_version': None,
                                'ab_test_active': False,
                                'model_a_games': 0,
                                'model_a_wins': 0,
                                'model_b_games': 0,
                                'model_b_wins': 0
                            })
                            response = {
                                'success': True,
                                'message': f'Model A ({current_version}) remains as current model',
                                'winner': 'A'
                            }
                
            elif action == 'update_thresholds':
                updates = {}
                if 'games_threshold' in data:
                    updates['games_threshold'] = int(data['games_threshold'])
                if 'ab_test_games_required' in data:
                    updates['ab_test_games_required'] = int(data['ab_test_games_required'])
                if updates:
                    update_ml_config(updates)
                response = {'success': True, 'message': 'Thresholds updated', 'updates': updates}
                
            elif action == 'get_models':
                # List all models in Firebase
                models = []
                docs = db.collection('ml_models').stream()
                for doc in docs:
                    model_data = doc.to_dict()
                    models.append({
                        'version': doc.id,
                        'created_at': str(model_data.get('created_at', '')),
                        'q_table_size': model_data.get('q_table_size', 'unknown')
                    })
                response = {'success': True, 'models': models}
            
            elif action == 'upload_model':
                # Upload a new model from base64-encoded pickle data
                model_b64 = data.get('model_data')
                version_name = data.get('version_name')
                
                if not model_b64 or not version_name:
                    response = {'success': False, 'error': 'model_data and version_name required'}
                else:
                    try:
                        # Validate it's valid base64/pickle
                        model_bytes = base64.b64decode(model_b64)
                        pickle.loads(model_bytes)  # Validates pickle format
                        
                        # Save to Firebase
                        db.collection('ml_models').document(version_name).set({
                            'model_data': model_b64,
                            'created_at': firestore.SERVER_TIMESTAMP,
                            'version': version_name,
                            'q_table_size': len(pickle.loads(model_bytes))
                        })
                        response = {'success': True, 'message': f'Model {version_name} uploaded successfully'}
                    except Exception as e:
                        response = {'success': False, 'error': f'Invalid model data: {str(e)}'}
            
            elif action == 'start_ab_test':
                # Manually start A/B test with a specified challenger model
                # This bypasses the 200 game requirement
                challenger_version = data.get('challenger_version')
                
                if not challenger_version:
                    response = {'success': False, 'error': 'challenger_version required'}
                else:
                    # Verify the model exists
                    model_doc = db.collection('ml_models').document(challenger_version).get()
                    if not model_doc.exists:
                        response = {'success': False, 'error': f'Model {challenger_version} not found'}
                    else:
                        # Start A/B test
                        update_ml_config({
                            'challenger_model_version': challenger_version,
                            'ab_test_active': True,
                            'model_a_games': 0,
                            'model_a_wins': 0,
                            'model_b_games': 0,
                            'model_b_wins': 0
                        })
                        response = {
                            'success': True, 
                            'message': f'A/B test started with challenger: {challenger_version}',
                            'note': 'Bypassed 200 game requirement as requested'
                        }
                
            else:
                response = {'error': f'Unknown action: {action}', 'available_actions': [
                    'upload_original', 'enable_training', 'disable_training',
                    'reset_game_counter', 'cancel_ab_test', 'manually_conclude_ab_test',
                    'update_thresholds', 'get_models', 'upload_model', 'start_ab_test'
                ]}
            
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
            self.wfile.write(json.dumps({'error': str(e)}).encode())
