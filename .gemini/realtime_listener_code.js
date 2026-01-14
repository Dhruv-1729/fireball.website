// Helper function to setup Firebase real-time listeners or fall back to polling
const startPolling = (mId) => {
    // Try to use Firebase real-time listener first
    if (window.firestoreDb) {
        try {
            // Import the necessary Firebase functions
            import('https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js')
                .then(({ doc, onSnapshot }) => {
                    console.log('✅ Setting up real-time listener for match:', mId);

                    const matchRef = doc(window.firestoreDb, 'matches', mId);

                    // Set up real-time listener
                    const unsubscribe = onSnapshot(matchRef,
                        (docSnapshot) => {
                            if (docSnapshot.exists()) {
                                const match = docSnapshot.data();

                                // Determine if current player is player1
                                const isPlayer1 = match.player1 === playerId;
                                const opponentSubmitted = (match.player2_move !== null) ? isPlayer1 : (match.player1_move !== null);

                                // Determine if current player won
                                const winner = match.winner;
                                let didIWin = null;
                                if (winner) {
                                    didIWin = (isPlayer1 && winner === 'player1') || (!isPlayer1 && winner === 'player2');
                                }

                                const newGameState = {
                                    myCharges: isPlayer1 ? match.player1_charges : match.player2_charges,
                                    opponentCharges: isPlayer1 ? match.player2_charges : match.player1_charges,
                                    turn: match.turn,
                                    status: match.status,
                                    winner: match.winner,
                                    didIWin: didIWin,
                                    lastResult: match.last_result,
                                    lastMyMove: isPlayer1 ? match.last_p1_move : match.last_p2_move,
                                    lastOpponentMove: isPlayer1 ? match.last_p2_move : match.last_p1_move,
                                    opponentSubmitted: opponentSubmitted,
                                    opponent: isPlayer1 ? match.player2_username : match.player1_username
                                };

                                setGameState(newGameState);

                                // Update history if there's a new move
                                if (newGameState.lastResult && newGameState.lastMyMove) {
                                    setHistory(prev => {
                                        if (prev.length === 0 || prev[0].you !== newGameState.lastMyMove || prev[0].opp !== newGameState.lastOpponentMove) {
                                            return [{ you: newGameState.lastMyMove, opp: newGameState.lastOpponentMove }, ...prev.slice(0, 9)];
                                        }
                                        return prev;
                                    });
                                }

                                // If match finished, show end overlay
                                if (newGameState.status === 'finished') {
                                    setShowEndOverlay(true);
                                }
                            }
                        },
                        (error) => {
                            console.error('❌ Firebase listener error:', error);
                            // Fall back to polling on error
                            console.log('⚠️  Falling back to polling due to listener error');
                            startPollingFallback(mId);
                        }
                    );

                    // Store unsubscribe function to clean up later
                    setPollInterval(unsubscribe);

                    // Do an initial fetch
                    fetchGameState(mId);
                })
                .catch(err => {
                    console.warn('⚠️  Failed to set up Firebase listener:', err);
                    startPollingFallback(mId);
                });
        } catch (e) {
            console.warn('⚠️  Firebase listener setup failed:', e);
            startPollingFallback(mId);
        }
    } else {
        // Firebase not available, use polling
        console.log('ℹ️  Firebase not available, using polling');
        startPollingFallback(mId);
    }
};

const startPollingFallback = (mId) => {
    console.log('🔄 Using polling fallback');
    const interval = setInterval(() => fetchGameState(mId), 1500);
    setPollInterval(interval);
    fetchGameState(mId);
};

const fetchGameState = async (mId) => {
    try {
        const res = await fetch('/api/matchmaking', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'get_game_state', matchId: mId, playerId })
        });

        const data = await res.json();
        setGameState(data);

        if (data.lastResult && data.lastMyMove) {
            setHistory(prev => {
                if (prev.length === 0 || prev[0].you !== data.lastMyMove || prev[0].opp !== data.lastOpponentMove) {
                    return [{ you: data.lastMyMove, opp: data.lastOpponentMove }, ...prev.slice(0, 9)];
                }
                return prev;
            });
        }

        if (data.status === 'finished') {
            // For real-time listeners, unsubscribe; for polling, clear interval
            if (typeof pollInterval === 'function') {
                pollInterval(); // unsubscribe
            } else {
                clearInterval(pollInterval);
            }
            setShowEndOverlay(true);
        }
    } catch (e) {
        console.error('Fetch state error:', e);
    }
};
