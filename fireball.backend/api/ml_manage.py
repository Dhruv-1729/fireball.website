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
TRAINING_ENABLED = False  # Master switch
GAMES_THRESHOLD_FOR_TRAINING = 200
AB_TEST_GAMES_REQUIRED = 15


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
        return {}
    
    try:
        # Count AI vs Human games
        ai_games = list(db.collection('ai_vs_human_matches').limit(1000).stream())
        ai_game_count = len(ai_games)
        ai_wins = sum(1 for g in ai_games if g.to_dict().get('winner') == 'ai')
        
        # Count online games
        online_games = list(db.collection('matches').where('status', '==', 'finished').limit(1000).stream())
        online_game_count = len(online_games)
        
        return {
            'ai_vs_human_games': ai_game_count,
            'ai_win_rate': round(ai_wins / ai_game_count * 100, 1) if ai_game_count > 0 else 0,
            'online_games': online_game_count,
            'total_games': ai_game_count + online_game_count
        }
    except Exception as e:
        print(f"Error getting training stats: {e}")
        return {}


def get_historical_win_rates(days=180):
    """Get historical win rate data aggregated by date for charting."""
    if not db:
        return []
    
    try:
        from datetime import datetime, timezone, timedelta
        
        # Get all AI games from the last N days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        ai_games = list(db.collection('ai_vs_human_matches').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(2000).stream())
        
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
        print(f"Error getting historical win rates: {e}")
        return []


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """GET request returns current ML status."""
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
                    data = doc.to_dict()
                    models.append({
                        'version': doc.id,
                        'created_at': str(data.get('created_at', '')),
                        'q_table_size': data.get('q_table_size', 'unknown')
                    })
                response = {'success': True, 'models': models}
                
            else:
                response = {'error': f'Unknown action: {action}', 'available_actions': [
                    'upload_original', 'enable_training', 'disable_training',
                    'reset_game_counter', 'cancel_ab_test', 'update_thresholds', 'get_models'
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
