<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fireball AI</title>
    <!-- Load Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Load React & ReactDOM -->
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <!-- Load Babel to transpile JSX -->
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <!-- Load Lucide Icons -->
    <script src="https://unpkg.com/lucide-react@0.294.0/dist/umd/lucide-react.js"></script>

    <style>
        /* Simple loading spinner */
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gray-950 text-gray-200 font-sans">
    
    <!-- This is the root element where our React app will be mounted -->
    <div id="root"></div>

    <!-- This is our React application -->
    <script type="text/babel">
        // Get React functions from the window
        const { useState, useEffect, createContext, useContext, useMemo } = React;
        const { createRoot } = ReactDOM;
        // Get Icons from the window
        const { Rocket, Users, BarChart3, Shield, Zap, Power, Bomb, Flame, Snowflake, Brain, Swords, X } = lucide-react;

        // --- Core Game Logic (Ported from Python) ---
        // We need this logic on the frontend to display legal moves and game state
        // without waiting for the backend for everything.
        const GameLogic = {
            getLegalMoves: (charges) => {
                const moves = ["charge", "shield"];
                if (charges >= 1) moves.push("fireball");
                if (charges >= 2) moves.push("iceball");
                if (charges >= 5) moves.push("megaball");
                return moves;
            },
            getMoveCost: (move) => {
                const costs = { "charge": -1, "fireball": 1, "iceball": 2, "megaball": 5 };
                return costs[move] || 0;
            },
            determineWinner: (p1, p2) => {
                if (p1 === p2 && p1 !== "megaball") return "continue";
                if (p1 === "megaball") return p2 !== "megaball" ? "player1" : "continue";
                if (p2 === "megaball") return "player2";
                if (p1 === "shield" || p2 === "shield") return "continue";
                
                const winMap = {
                    "fireball": ["charge"],
                    "iceball": ["charge", "fireball"]
                };
                
                if (winMap.fireball.includes(p2)) return "player1";
                if (winMap.iceball.includes(p2)) return "player1";

                if (winMap.fireball.includes(p1)) return "player2";
                if (winMap.iceball.includes(p1)) return "player2";
                
                return "continue";
            }
        };

        // --- React Components ---

        // App Header
        function Header() {
            // This is the gradient you requested: #29f2a2 to #46c5fc
            return (
                <header class="w-full p-4">
                    <div class="w-full max-w-4xl mx-auto flex justify-between items-center">
                        <div class="flex items-center space-x-3">
                            <div class="p-2 bg-gray-800 rounded-lg">
                                <Rocket class="w-8 h-8 bg-gradient-to-r from-[#29f2a2] to-[#46c5fc] text-transparent bg-clip-text" />
                            </div>
                            <h1 class="text-3xl font-bold text-white">Fireball AI</h1>
                        </div>
                    </div>
                </header>
            );
        }

        // Main App Navigation
        function App() {
            // Use state to manage which page is active
            const [page, setPage] = useState("playAI"); // "playAI", "play1v1", "stats"

            return (
                <div class="min-h-screen w-full p-4 md:p-8 flex flex-col items-center">
                    <Header />
                    
                    {/* Navigation Tabs */}
                    <nav class="flex space-x-2 bg-gray-900 p-2 rounded-lg my-6">
                        <NavButton
                            icon={Brain}
                            label="Play vs AI"
                            isActive={page === "playAI"}
                            onClick={() => setPage("playAI")}
                        />
                        <NavButton
                            icon={Swords}
                            label="Play 1v1"
                            isActive={page === "play1v1"}
                            onClick={() => setPage("play1v1")}
                        />
                        <NavButton
                            icon={BarChart3}
                            label="Stats"
                            isActive={page === "stats"}
                            onClick={() => setPage("stats")}
                        />
                    </nav>

                    {/* Page Content */}
                    <main class="w-full max-w-3xl">
                        {page === "playAI" && <PlayAIPage />}
                        {page === "play1v1" && <Play1v1Page />}
                        {page === "stats" && <StatsPage />}
                    </main>
                </div>
            );
        }

        function NavButton({ icon: Icon, label, isActive, onClick }) {
            return (
                <button
                    onClick={onClick}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-all duration-200 ${
                        isActive
                            ? 'bg-gradient-to-r from-[#29f2a2] to-[#46c5fc] text-gray-950 font-semibold shadow-lg'
                            : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                    }`}
                >
                    <Icon className="w-5 h-5" />
                    <span>{label}</span>
                </button>
            );
        }

        // --- Page: Play vs AI ---
        function PlayAIPage() {
            const [game, setGame] = useState({ playerCharges: 0, aiCharges: 0, gameOver: false, winner: null });
            const [history, setHistory] = useState([]);
            const [loading, setLoading] = useState(false);
            const [error, setError] = useState(null);

            const legalMoves = GameLogic.getLegalMoves(game.playerCharges);

            const handlePlayMove = async (move) => {
                setLoading(true);
                setError(null);
                
                // ** BACKEND CALL (TODO) **
                // This is where you would call your Vercel serverless function
                try {
                    /*
                    const response = await fetch('/api/play_ai', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            playerMove: move,
                            playerCharges: game.playerCharges,
                            aiCharges: game.aiCharges,
                            // You might need to send history for the AI state
                            moveHistory: history.map(h => h.player),
                            oppMoveHistory: history.map(h => h.ai),
                        }),
                    });

                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    
                    const data = await response.json();
                    
                    // data = { 
                    //   playerCharges: 0, aiCharges: 1, 
                    //   aiMove: "charge", result: "continue", 
                    //   gameOver: false, winner: null
                    // }

                    setGame({
                        playerCharges: data.playerCharges,
                        aiCharges: data.aiCharges,
                        gameOver: data.gameOver,
                        winner: data.winner
                    });
                    
                    setHistory([
                        { player: move, ai: data.aiMove, result: data.result },
                        ...history
                    ]);
                    */
                    
                    // --- MOCK DATA (Remove when backend is live) ---
                    // Simulate a delay
                    await new Promise(res => setTimeout(res, 500));
                    
                    setError("The 'Play vs AI' backend is not yet connected. Please follow the instructions in Part 2 to build the AI backend.");
                    // --- End Mock Data ---

                } catch (err) {
                    setError('Failed to fetch AI response. Is the API running?');
                    console.error(err);
                }
                setLoading(false);
            };

            const handlePlayAgain = () => {
                setGame({ playerCharges: 0, aiCharges: 0, gameOver: false, winner: null });
                setHistory([]);
                setError(null);
            };

            return (
                <div class="bg-gray-900 p-6 rounded-lg shadow-xl border border-gray-700">
                    <h2 class="text-2xl font-bold mb-4">Play against the AI</h2>
                    <ChargeDisplay playerCharges={game.playerCharges} oppCharges={game.aiCharges} />
                    
                    {error && (
                        <div class="my-4 p-3 bg-red-900 border border-red-700 text-red-200 rounded-lg">
                            <p class="font-semibold">Backend Error</p>
                            <p>{error}</p>
                        </div>
                    )}
                    
                    {game.gameOver ? (
                        <EndGameDisplay winner={game.winner} onPlayAgain={handlePlayAgain} />
                    ) : (
                        <MoveSelector
                            legalMoves={legalMoves}
                            onMoveSelect={handlePlayMove}
                            disabled={loading}
                        />
                    )}
                    
                    <HistoryDisplay history={history} />
                </div>
            );
        }

        // --- Page: 1v1 Fireball ---
        function Play1v1Page() {
            // This component would have much more complex state:
            // username, matchId, playerIndex (p1 or p2), gameState, etc.
            const [username, setUsername] = useState("");
            const [matchId, setMatchId] = useState(null);
            const [error, setError] = useState(null);
            const [loading, setLoading] = useState(false);
            
            const handleFindMatch = async () => {
                if (!username) {
                    setError("Please enter a username.");
                    return;
                }
                setLoading(true);
                setError(null);
                
                // ** BACKEND CALL (TODO) **
                // This would call your matchmaking API
                try {
                    /*
                    const response = await fetch('/api/matchmaking', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ username }),
                    });
                    const data = await response.json();
                    // data = { matchId: "...", status: "pending" }
                    setMatchId(data.matchId);
                    */
                    
                    // --- MOCK DATA (Remove when backend is live) ---
                    await new Promise(res => setTimeout(res, 1000));
                    setError("The '1v1 Matchmaking' backend is not yet connected. Please follow the instructions to build the Firestore-based API.");
                    // --- End Mock Data ---
                    
                } catch (err) {
                    setError("Failed to find match.");
                }
                setLoading(false);
            };
            
            // This is a simplified view. A real 1v1 page would
            // need polling (useEffect with interval) to check game state.
            return (
                <div class="bg-gray-900 p-6 rounded-lg shadow-xl border border-gray-700">
                    <h2 class="text-2xl font-bold mb-4">Play 1v1 vs Another Player</h2>
                    
                    {error && (
                        <div class="my-4 p-3 bg-red-900 border border-red-700 text-red-200 rounded-lg">
                            <p class="font-semibold">Backend Error</p>
                            <p>{error}</p>
                        </div>
                    )}
                    
                    {!matchId ? (
                        <div class="flex flex-col space-y-4">
                            <input
                                type="text"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                placeholder="Enter your username"
                                className="p-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-[#29f2a2]"
                            />
                            <button
                                onClick={handleFindMatch}
                                disabled={loading}
                                className="flex justify-center items-center p-4 bg-gradient-to-r from-[#29f2a2] to-[#46c5fc] text-gray-950 font-bold rounded-lg text-lg hover:opacity-90 transition-opacity disabled:opacity-50"
                            >
                                {loading ? <div className="loader" /> : "Find Match"}
                            </button>
                        </div>
                    ) : (
                        <div class="text-center">
                            <h3 class="text-xl">Match Found!</h3>
                            <p class="text-gray-400">Match ID: {matchId}</p>
                            <p class="my-4">Waiting for player...</p>
                            {/* In a real app, this is where the game board would appear */}
                        </div>
                    )}
                </div>
            );
        }

        // --- Page: Statistics ---
        function StatsPage() {
            const [stats, setStats] = useState(null);
            const [loading, setLoading] = useState(true);
            const [error, setError] = useState(null);

            useEffect(() => {
                const fetchStats = async () => {
                    setLoading(true);
                    setError(null);
                    try {
                        // ** BACKEND CALL (TODO) **
                        // This is where you would call your stats API
                        /*
                        const response = await fetch('/api/stats');
                        if (!response.ok) {
                            throw new Error('Failed to fetch stats');
                        }
                        const data = await response.json();
                        setStats(data);
                        */
                        
                        // --- MOCK DATA (Remove when backend is live) ---
                        await new Promise(res => setTimeout(res, 800));
                        setStats({
                            uniqueVisitors: 1234,
                            aiWinRate: 42.5,
                            aiGames: 567,
                            avgMatchLength: 8.2,
                            recentMatches: [
                                { winner: "Player123", rounds: 7, p1: "Player123", p2: "BotKiller" },
                                { winner: "AI", rounds: 5, p1: "Human", p2: "AI" },
                                { winner: "Dhruv", rounds: 12, p1: "Dhruv", p2: "TestUser" },
                            ]
                        });
                        // --- End Mock Data ---
                        
                    } catch (err) {
                        setError(err.message);
                    }
                    setLoading(false);
                };
                
                // fetchStats(); // Uncomment this to run on load
                setError("The 'Stats' backend is not yet connected. Please follow the instructions to build the Firestore-based API.");
                setLoading(false); // Remove this when fetchStats is live
            }, []);

            if (loading) {
                return <div class="flex justify-center items-center p-10"><div class="loader" /></div>;
            }
            
            if (error) {
                 return (
                    <div class="my-4 p-3 bg-red-900 border border-red-700 text-red-200 rounded-lg">
                        <p class="font-semibold">Backend Error</p>
                        <p>{error}</p>
                    </div>
                );
            }

            if (!stats) {
                return <p>No stats available.</p>;
            }

            return (
                <div class="bg-gray-900 p-6 rounded-lg shadow-xl border border-gray-700">
                    <h2 class="text-2xl font-bold mb-6">Live Statistics</h2>
                    
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <StatCard label="Unique Visitors" value={stats.uniqueVisitors} />
                        <StatCard label="AI Win Rate" value={`${stats.aiWinRate}%`} />
                        <StatCard label="Avg. Match" value={`${stats.avgMatchLength} rounds`} />
                    </div>
                    
                    <h3 class="text-xl font-semibold mb-4">Recent Matches</h3>
                    <div class="space-y-3">
                        {stats.recentMatches.map((match, i) => (
                            <div key={i} class="p-4 bg-gray-800 rounded-lg flex justify-between items-center">
                                <div>
                                    <span class="font-bold text-white">{match.winner}</span>
                                    <span class="text-gray-400"> won in {match.rounds} rounds</span>
                                </div>
                                <span class="text-sm text-gray-500">{match.p1} vs {match.p2}</span>
                            </div>
                        ))}
                    </div>
                </div>
            );
        }
        
        function StatCard({ label, value }) {
            return (
                <div class="bg-gray-800 p-4 rounded-lg border border-gray-700">
                    <p class="text-sm text-gray-400 mb-1">{label}</p>
                    <p class="text-3xl font-bold text-white">{value}</p>
                </div>
            );
        }

        // --- Reusable Game Components ---

        function ChargeDisplay({ playerCharges, oppCharges, p1Label = "Your", p2Label = "Opponent" }) {
            return (
                <div class="flex justify-around items-center bg-gray-950 p-4 rounded-lg mb-6 border border-gray-700">
                    <div class="text-center">
                        <p class="text-lg font-semibold text-[#46c5fc]">{p1Label} Charges</p>
                        <p class="text-5xl font-bold text-white">{playerCharges}</p>
                    </div>
                    <div class="h-16 w-px bg-gray-700"></div>
                    <div class="text-center">
                        <p class="text-lg font-semibold text-[#29f2a2]">{p2Label} Charges</p>
                        <p class="text-5xl font-bold text-white">{oppCharges}</p>
                    </div>
                </div>
            );
        }

        function MoveSelector({ legalMoves, onMoveSelect, disabled }) {
            const moveIcons = {
                charge: { icon: Zap, color: "text-yellow-400" },
                shield: { icon: Shield, color: "text-blue-400" },
                fireball: { icon: Flame, color: "text-red-500" },
                iceball: { icon: Snowflake, color: "text-cyan-300" },
                megaball: { icon: Bomb, color: "text-purple-400" },
            };

            return (
                <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {legalMoves.map(move => {
                        const { icon: Icon, color } = moveIcons[move];
                        return (
                            <button
                                key={move}
                                onClick={() => onMoveSelect(move)}
                                disabled={disabled}
                                className={`flex flex-col items-center justify-center p-6 bg-gray-800 rounded-lg border border-gray-700 text-white font-semibold
                                    hover:bg-gray-700 hover:border-gray-600 transition-all duration-200
                                    focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-[#29f2a2]
                                    disabled:opacity-50 disabled:cursor-not-allowed`}
                            >
                                <Icon className={`w-10 h-10 ${color} mb-2`} />
                                {move.charAt(0).toUpperCase() + move.slice(1)}
                            </button>
                        );
                    })}
                </div>
            );
        }

        function EndGameDisplay({ winner, onPlayAgain }) {
            const isPlayerWin = winner === 'player1' || winner === 'human';
            
            return (
                <div class="text-center p-6">
                    <h3 class={`text-4xl font-bold ${isPlayerWin ? 'text-[#29f2a2]' : 'text-red-500'}`}>
                        {isPlayerWin ? "You Won!" : "You Lost!"}
                    </h3>
                    <button
                        onClick={onPlayAgain}
                        className="mt-6 px-6 py-3 bg-gradient-to-r from-[#29f2a2] to-[#46c5fc] text-gray-950 font-bold rounded-lg text-lg hover:opacity-90 transition-opacity"
                    >
                        Play Again
                    </button>
                </div>
            );
        }

        function HistoryDisplay({ history }) {
            if (history.length === 0) return null;
            
            return (
                <div class="mt-8">
                    <h3 class="text-xl font-semibold mb-4">Turn History</h3>
                    <div class="space-y-3 max-h-60 overflow-y-auto p-4 bg-gray-950 rounded-lg border border-gray-700">
                        {history.map((turn, i) => (
                            <div key={i} class="flex justify-between items-center text-sm p-3 bg-gray-800 rounded">
                                <div>
                                    <p>You played <span class="font-bold text-[#46c5fc]">{turn.player}</span></p>
                                    <p>Opponent played <span class="font-bold text-[#29f2a2]">{turn.ai}</span></p>
                                </div>
                                <span class="font-semibold text-gray-300">{turn.result}</span>
                            </div>
                        ))}
                    </div>
                </div>
            );
        }


        // --- Mount the App ---
        const root = createRoot(document.getElementById('root'));
        root.render(<App />);

    </script>
</body>
</html>
