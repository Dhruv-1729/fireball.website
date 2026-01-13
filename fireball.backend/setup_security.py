"""
Database Setup Script for Fireball Security
============================================

Run this script ONCE to set up the required database documents for the new security features.

This will:
1. Create the admin_config document with the password hash
2. Create the site_config document for maintenance mode

Prerequisites:
- Your FIREBASE_SERVICE_ACCOUNT environment variable must be set
- Or create a .env file with the service account JSON

Usage:
    python setup_security.py
"""

import os
import json
import sys
import re

# Try to load from .env.local if present (check both current dir and parent)
def load_env_file(path):
    """Load environment variables from a .env file."""
    if not os.path.exists(path):
        return False
    
    with open(path, 'r') as f:
        content = f.read()
    
    # Find FIREBASE_SERVICE_ACCOUNT using regex
    match = re.search(r'FIREBASE_SERVICE_ACCOUNT="({.*?})"', content, re.DOTALL)
    if match:
        os.environ['FIREBASE_SERVICE_ACCOUNT'] = match.group(1)
        return True
    
    return False

# Check various locations for .env.local
env_locations = [
    os.path.join(os.path.dirname(__file__), '.env.local'),
    os.path.join(os.path.dirname(__file__), '..', '.env.local'),
    '.env.local',
    '../.env.local'
]

for env_path in env_locations:
    if load_env_file(env_path):
        print(f"✓ Loaded environment from {env_path}")
        break

# Initialize Firebase
import firebase_admin
from firebase_admin import credentials, firestore

service_account_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT', '')
if not service_account_str:
    print("ERROR: FIREBASE_SERVICE_ACCOUNT environment variable not set!")
    print("Please set it to your Firebase service account JSON string.")
    sys.exit(1)

try:
    service_account_info = json.loads(service_account_str)
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✓ Connected to Firebase")
except Exception as e:
    print(f"ERROR: Failed to connect to Firebase: {e}")
    sys.exit(1)

# Admin password hash (SHA-256 of the password)
# This is the hash you provided: fa20e43cc591ce0772b90d46509f5ed031276dfe381de73c9b2089c1c29f5072
ADMIN_PASSWORD_HASH = 'fa20e43cc591ce0772b90d46509f5ed031276dfe381de73c9b2089c1c29f5072'

def setup_admin_config():
    """Set up the admin configuration with password hash."""
    try:
        doc_ref = db.collection('admin_config').document('auth')
        doc = doc_ref.get()
        
        if doc.exists:
            print("! admin_config/auth document already exists")
            existing = doc.to_dict()
            if existing.get('password_hash') == ADMIN_PASSWORD_HASH:
                print("  ✓ Password hash is correct")
            else:
                print("  → Updating password hash...")
                doc_ref.update({'password_hash': ADMIN_PASSWORD_HASH})
                print("  ✓ Password hash updated")
        else:
            doc_ref.set({
                'password_hash': ADMIN_PASSWORD_HASH,
                'created_at': firestore.SERVER_TIMESTAMP
            })
            print("✓ Created admin_config/auth document with password hash")
        
        return True
    except Exception as e:
        print(f"ERROR setting up admin config: {e}")
        return False

def setup_site_config():
    """Set up the site configuration for maintenance mode."""
    try:
        doc_ref = db.collection('site_config').document('main')
        doc = doc_ref.get()
        
        if doc.exists:
            print("! site_config/main document already exists")
            print("  ✓ Maintenance mode is currently:", doc.to_dict().get('maintenance_mode', False))
        else:
            doc_ref.set({
                'maintenance_mode': False,
                'maintenance_message': 'Site undergoing temporary maintenance. Please check back later.',
                'created_at': firestore.SERVER_TIMESTAMP
            })
            print("✓ Created site_config/main document (maintenance_mode: false)")
        
        return True
    except Exception as e:
        print(f"ERROR setting up site config: {e}")
        return False

def main():
    print("\n🔐 Fireball Security Setup\n")
    print("=" * 40)
    
    success = True
    
    if not setup_admin_config():
        success = False
    
    if not setup_site_config():
        success = False
    
    print("\n" + "=" * 40)
    
    if success:
        print("\n✅ Setup complete!")
        print("\nNext steps:")
        print("1. Deploy your backend to Vercel: vercel --prod")
        print("2. Go to your site's /admin page")
        print("3. Log in with your admin password")
        print("\nAdmin Features:")
        print("- View stats and game logs")
        print("- Toggle maintenance mode from Settings tab")
        print("- Session tokens expire after 24 hours")
    else:
        print("\n❌ Setup had errors. Please check the messages above.")
    
    print()

if __name__ == '__main__':
    main()
