import os
import json
import logging
import firebase_admin
from firebase_admin import credentials, firestore

with open('.env.local') as f:
    for line in f:
        if line.startswith('FIREBASE_SERVICE_ACCOUNT='):
            val = line.split('=', 1)[1].strip()
            if val.startswith('"') and val.endswith('"'): val = val[1:-1]
            os.environ['FIREBASE_SERVICE_ACCOUNT'] = val
            import ast
            os.environ['FIREBASE_SERVICE_ACCOUNT'] = ast.literal_eval('"' + os.environ['FIREBASE_SERVICE_ACCOUNT'] + '"')

print("Loading api.matchmaking")
import api.matchmaking as mm

# mock handler
class MockHandler:
    def __init__(self, method, action, data):
        self.method = method
        self.action = action
        self.data = data
        data['action'] = action
        self.post_body = json.dumps(data).encode('utf-8')
        self.headers = {'Content-Length': str(len(self.post_body))}
        self.rfile = self
    
    def read(self, length):
        return self.post_body
    
    def send_response(self, code):
        print(f"HTTP {code}")
    def send_header(self, k, v): pass
    def end_headers(self): pass
    def wfile_write(self, data):
        print(f"Response: {data.decode('utf-8')}")

    def test(self):
        h = mm.handler(None, None, None)
        h.rfile = self
        h.headers = self.headers
        h.wfile = type('WFile', (), {'write': self.wfile_write})()
        h.send_response = self.send_response
        h.send_header = self.send_header
        h.end_headers = self.end_headers
        h.do_POST()

class DBTest:
    def run(self):
        print("Testing check_match")
        # try find match
        pass

mh = MockHandler('POST', 'submit_move', {
    "matchId": "INVALID_ID_TEST",
    "playerId": "p1",
    "move": "charge"
})
try:
    mh.test()
except Exception as e:
    print(f"FAILED: {e}")
