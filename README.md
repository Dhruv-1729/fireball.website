<p align="center">
  <img src="fireball.frontend/banner.png" alt="Fireball" width="600"/>
</p>

<p align="center">
  <strong>A turn-based strategy game with a Q-Learning ML Bot.</strong><br/>
  Charge up, read your opponent, and throw fireballs. 
  "It's easy to learn, but hard to master."
</p>

<p align="center">
  <a href="https://fireballml.vercel.app">
    <img src="https://img.shields.io/badge/play_now-fireballml.vercel.app-orange?style=for-the-badge&logo=vercel&logoColor=white" alt="Play Now"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/react-18-61DAFB?style=flat-square&logo=react&logoColor=black"/>
  <img src="https://img.shields.io/badge/firebase-firestore-FFCA28?style=flat-square&logo=firebase&logoColor=black"/>
  <img src="https://img.shields.io/badge/deployed_on-vercel-000000?style=flat-square&logo=vercel&logoColor=white"/>
</p>

---

## What is Fireball?

Fireball is a turn based game similar to rock paper scissors, but removes the 100% luck aspect of it.

Both players pick a move each turn simultaneously. You need to **charge** energy to attack, but charging leaves you exposed. Shield blocks most attacks. Attacks cost energy. The mind games come from figuring out *when* your opponent is going to commit.

The AI is trained through Q-Learning with eligibility traces and self-play. It picks up on patterns; so if you're predictable, it'll punish you. 

## Moves

| Move | Cost | What it does |
|------|------|-------------|
| ⚡ **Charge** | Free (+1) | Gain 1 energy. You're vulnerable while doing this. |
| 🛡️ **Shield** | Free | Block all attacks except Megaball. |
| 🔥 **Fireball** | 1 energy | Kills a charging opponent. Blocked by Shield, beaten by Iceball. |
| ❄️ **Iceball** | 2 energy | Beats Fireball and Charge. Still blocked by Shield. |
| 💥 **Megaball** | 5 energy | Unblockable instant win. Only countered by another Megaball. |

You *have* to charge to win, but charging is the most dangerous thing you can do.


Everything runs on Vercel serverless functions. Game state lives in Firestore. No WebSocket server — the 1v1 system uses a polling-based approach with Firebase real-time listeners on the client for responsiveness.


## Game Modes

- **vs ML Model** — Play against the Q-Learning model. Server-side game sessions prevent client-side cheating.
- **1v1 Online** — Real-time matchmaking against other players. Heartbeat system tracks who's online, matches get auto-terminated if someone disconnects.


## Optimization

Early on in development the app was noticeably slow. The first move against the AI after a "cold" start took 5-6 seconds, and even warm requests were in the 500-800ms range because every single move triggered a Firebase round-trip to fetch the model. I've since gotten cold starts down to ~200ms and warm moves under 50ms. Here's what I changed.

### Moved the model to the filesystem

The original version stored the Q-learning model in Firebase and downloaded it on every request. That's an 8.59 MB gzip file being fetched over the network each time someone makes a move. Super annoying for anyone who's playing.

Now `model.pkl` ships with the Vercel deployment itself. Python module-level code runs once when the Lambda container spins up, so the model gets loaded into memory on the first request and just stays there until the container gets recycled (usually about 60 seconds of idle time). Every request after that reads from a Python dict in memory.

```python
ai_player, _model_loaded = _load_local_model()
_LOCAL_Q_TABLE = ai_player.q_table if _model_loaded else None
```

Cold starts went from 5-6s to ~200ms. Warm requests went from ~500ms to basically nothing (~3ms).

### Deferred Firebase initialization

Firebase Admin SDK takes 2-3 seconds to set up (credential parsing, gRPC channels, etc.). The old code ran `firebase_admin.initialize_app()` at the top of every module, meaning every cold start paid that cost even if the request never touched Firestore. The model move endpoint only needs Firebase for optional game logging, not for computing the actual move.

So now Firebase init is behind a `_get_db()` function with a lock. It only runs the first time something actually tries to write to Firestore. If a request comes in, gets the model move from the local model, and returns, Firebase never initializes at all.

```python
_db = None
_db_init_lock = threading.Lock()

def _get_db():
    global _db, _firebase_init_done
    if _db is not None:
        return _db
    with _db_init_lock:
        if _db is not None:
            return _db
        # initialize here, only once
```

This optimized another 2-3 seconds off cold starts for the machine learning endpoints.

### Non-blocking game logging

When a game ended, the server used to write the match result to Firestore and then send the response. Players were waiting an extra 100-300ms after winning/losing just so the server could finish a log write they don't care about.

Now game-over logging happens in a daemon thread. The response goes out immediately and the Firestore write happens in the background. Vercel keeps the container alive long enough for it to finish.

```python
def _log_game_data_async(turn_data, match_data=None, ...):
    def _do_log():
        db = _get_db()
        if db:
            db.collection('ai_game_turns').add(turn_data)
    threading.Thread(target=_do_log, daemon=True).start()
```

### No build step

The whole frontend is one `index.html` file. React and ReactDOM come from unpkg, Firebase client SDK comes from Google's CDN. No webpack, no Vite, no package.json. The total payload is about 120KB before compression.

I also skipped React Router. Navigation is just a `useState` that swaps which component renders. It's a game with like 4 screens, I didn't need a routing library for that.


### Compressed model artifact

`model.pkl` is gzip-compressed (9MB on disk, ~40MB / 383K Q-states when decompressed). The loader checks for gzip magic bytes and decompresses automatically:

```python
def load_from_bytes(self, data):
    import gzip
    if data.startswith(b'\x1f\x8b'):
        data = gzip.decompress(data)
    loaded = pickle.loads(data)
```

Smaller artifact means faster deploys and faster cold-start disk reads. It also keeps things under Vercel's deployment size limits.

### Sparse Q-table

The Q-table uses `defaultdict(lambda: defaultdict(float))`, so only states the model actually visited during training take up memory. There's no pre-allocated matrix for the full state space. With 383K states this matters on Vercel's serverless containers where memory is limited.

Each API endpoint (`play_ai.py`, `game_session.py`) loads its own copy at module level, but since Vercel routes them to separate Lambda containers they don't share memory anyway.


## License

Do whatever you want with it. If you build something cool on top of it, I'd love to hear about it 😀
