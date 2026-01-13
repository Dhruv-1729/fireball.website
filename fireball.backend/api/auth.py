"""
Authentication API for Admin Panel
Handles secure server-side authentication.
- POST /api/auth?action=login - Verify admin password
- POST /api/auth?action=verify - Verify admin token
- GET /api/auth?action=config - Get public config (maintenance mode, etc.)
"""

import json
import os
import hashlib
import secrets
import time
from datetime import datetime, timezone, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


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


# ============ TOKEN MANAGEMENT ============

TOKEN_EXPIRY_HOURS = 24


def get_admin_password_hash():
    """Get the admin password hash from the database."""
    if not db:
        return None
    try:
        doc = db.collection('admin_config').document('auth').get()
        if doc.exists:
            return doc.to_dict().get('password_hash')
        return None
    except Exception as e:
        print(f"Error getting admin password: {e}")
        return None


def verify_password(input_hash):
    """
    Verify the provided password hash against the stored hash.
    The client sends SHA256(password), we compare it to the stored hash.
    """
    stored_hash = get_admin_password_hash()
    if not stored_hash:
        return False
    
    # Constant-time comparison to prevent timing attacks
    return secrets.compare_digest(input_hash.lower(), stored_hash.lower())


def create_admin_token():
    """Create a secure admin session token."""
    if not db:
        return None
    
    try:
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        # Store the token hash (not the token itself)
        db.collection('admin_sessions').document(token_hash).set({
            'created_at': firestore.SERVER_TIMESTAMP,
            'expires_at': datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRY_HOURS),
            'valid': True
        })
        
        return token
    except Exception as e:
        print(f"Error creating token: {e}")
        return None


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


def invalidate_admin_token(token):
    """Invalidate an admin session token."""
    if not db or not token:
        return False
    
    try:
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        db.collection('admin_sessions').document(token_hash).update({'valid': False})
        return True
    except Exception as e:
        print(f"Error invalidating token: {e}")
        return False


def cleanup_expired_tokens():
    """Clean up expired admin tokens."""
    if not db:
        return 0
    
    try:
        now = datetime.now(timezone.utc)
        expired = db.collection('admin_sessions').where('expires_at', '<', now).limit(50).stream()
        
        deleted = 0
        for doc in expired:
            doc.reference.delete()
            deleted += 1
        
        return deleted
    except Exception as e:
        print(f"Error cleaning up tokens: {e}")
        return 0


# ============ MAINTENANCE MODE ============

def get_site_config():
    """Get public site configuration including maintenance mode."""
    if not db:
        return {'maintenance_mode': False}
    
    try:
        doc = db.collection('site_config').document('main').get()
        if doc.exists:
            config = doc.to_dict()
            return {
                'maintenance_mode': config.get('maintenance_mode', False),
                'maintenance_message': config.get('maintenance_message', 'Site undergoing temporary maintenance. Please check back later.')
            }
        return {'maintenance_mode': False}
    except Exception as e:
        print(f"Error getting site config: {e}")
        return {'maintenance_mode': False}


def set_maintenance_mode(enabled, message=None):
    """Set maintenance mode status."""
    if not db:
        return False
    
    try:
        updates = {
            'maintenance_mode': enabled,
            'updated_at': firestore.SERVER_TIMESTAMP
        }
        if message:
            updates['maintenance_message'] = message
        
        db.collection('site_config').document('main').set(updates, merge=True)
        return True
    except Exception as e:
        print(f"Error setting maintenance mode: {e}")
        return False


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            action = params.get('action', [''])[0]
            
            response = {}
            
            if action == 'config':
                # Public endpoint - get site config including maintenance mode
                response = get_site_config()
            else:
                response = {'error': 'Unknown action'}
            
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
        try:
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            action = params.get('action', [''])[0]
            
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len) if content_len > 0 else b'{}'
            data = json.loads(post_body) if post_body else {}
            
            response = {}
            status_code = 200
            
            if action == 'login':
                # Verify password and create token
                password_hash = data.get('passwordHash', '')
                
                if verify_password(password_hash):
                    token = create_admin_token()
                    if token:
                        response = {
                            'success': True,
                            'token': token,
                            'expiresIn': TOKEN_EXPIRY_HOURS * 3600
                        }
                        
                        # Cleanup expired tokens occasionally
                        if secrets.randbelow(10) == 0:
                            cleanup_expired_tokens()
                    else:
                        response = {'error': 'Failed to create session'}
                        status_code = 500
                else:
                    response = {'error': 'Invalid password'}
                    status_code = 401
                    
            elif action == 'verify':
                # Verify existing token
                auth_header = self.headers.get('Authorization', '')
                token = auth_header.replace('Bearer ', '') if auth_header.startswith('Bearer ') else data.get('token', '')
                
                if verify_admin_token(token):
                    response = {'valid': True}
                else:
                    response = {'valid': False, 'error': 'Invalid or expired token'}
                    status_code = 401
                    
            elif action == 'logout':
                # Invalidate token
                auth_header = self.headers.get('Authorization', '')
                token = auth_header.replace('Bearer ', '') if auth_header.startswith('Bearer ') else data.get('token', '')
                
                invalidate_admin_token(token)
                response = {'success': True}
                
            elif action == 'set_maintenance':
                # Protected action - requires valid token
                auth_header = self.headers.get('Authorization', '')
                token = auth_header.replace('Bearer ', '') if auth_header.startswith('Bearer ') else ''
                
                if not verify_admin_token(token):
                    response = {'error': 'Unauthorized'}
                    status_code = 401
                else:
                    enabled = data.get('enabled', False)
                    message = data.get('message')
                    
                    if set_maintenance_mode(enabled, message):
                        response = {'success': True, 'maintenance_mode': enabled}
                    else:
                        response = {'error': 'Failed to update maintenance mode'}
                        status_code = 500
                        
            else:
                response = {'error': f'Unknown action: {action}'}
                status_code = 400
            
            self.send_response(status_code)
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
