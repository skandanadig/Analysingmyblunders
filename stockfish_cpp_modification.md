# Modifying Stockfish's C++ Search Loop

This guide explains how to modify Stockfish's C++ source code to expose a "depth-trajectory" (the internal evaluation score at every progressive search depth ply) during a live execution. 

This is incredibly useful for Meta-Learner pipelines because the *volatility* of an engine's score across search depths is often a strong indicator of a highly complex or tactical position.

## The Goal
When Stockfish evaluates a position, it uses **Iterative Deepening**. It searches to depth 1, then depth 2, then depth 3, and so on. We want to capture the evaluation score at the end of each of these depth iterations and output a running history (trajectory) of these scores to standard output via the UCI protocol.

---

## Step-by-Step Modification

### 1. Locate the Iterative Deepening Loop
Open `search.cpp` in the Stockfish source code. You are looking for the main search function executed by threads, specifically `Thread::search()`. 

Inside this function, you will find the Iterative Deepening loop, which looks similar to this:

```cpp
while (++rootDepth < MAX_PLY && !Threads.stop && !(Limits.depth && rootDepth > Limits.depth)) {
    // ... search logic for the current depth ...
}
```

### 2. Introduce a Trajectory Container
Before the iterative deepening loop begins, declare a container to hold the evaluation scores. This will persist and grow as the depth increases.

```cpp
// Add this just before the `while (++rootDepth ...)` loop:
std::vector<int> scoreTrajectory;
```

### 3. Capture the Score at the End of the Iteration
Inside the iterative deepening loop, after the search for the current `rootDepth` completes (but before the loop circles back around), capture the best score. 

Stockfish stores the moves evaluated at the root in a `rootMoves` vector, sorted by score. The best move is currently sitting at `rootMoves[0]`.

```cpp
// Add this near the bottom of the iterative deepening loop body:

// Ensure we have a valid score and the search wasn't aborted midway
if (!Threads.stop) {
    // Internal score of the best move for the current depth
    Value currentScore = rootMoves[0].score;
    
    // We convert the internal 'Value' to Centipawns using Stockfish's internal UCI conversion
    int cpScore = currentScore * 100 / UCI::NormalizeEval; 
    
    scoreTrajectory.push_back(cpScore);
}
```

### 4. Format and Stream the Trajectory
We need to format this vector into a text string so our Python bridge can parse it. We can intercept where Stockfish prints the root move info and append our custom string.

```cpp
// Format the trajectory as a comma-separated string
std::string trajString = "";
for (size_t i = 0; i < scoreTrajectory.size(); ++i) {
    trajString += std::to_string(scoreTrajectory[i]);
    if (i != scoreTrajectory.size() - 1) {
        trajString += ",";
    }
}

// Stream to standard output alongside the standard info line
sync_cout << "info string trajectory " << trajString << sync_endl;
```

### 5. Hooking into the Exact Print Statement
In modern Stockfish versions, the UCI output for the principal variation is handled deep inside the iteration. If you want to merge it directly onto the main `info depth X` line instead of a separate `info string` line, you can find the print statements in `search.cpp` (or `uci.cpp` depending on the version) and append to it:

```cpp
sync_cout << "info depth " << rootDepth 
          << /* ... standard stockfish output ... */
          << " trajectory " << trajString   // <--- Add this custom token
          << sync_endl;
```

---

## Summary of the Expected Output
Once compiled, when you send a `go depth 20` command to your modified Stockfish binary from Python, the output will look something like this:

```text
info depth 1 score cp 45 time 2 nodes 300 ... trajectory 45
info depth 2 score cp 32 time 5 nodes 1500 ... trajectory 45,32
info depth 3 score cp 50 time 12 nodes 4000 ... trajectory 45,32,50
...
info depth 20 score cp 40 time 4000 nodes 12000000 ... trajectory 45,32,50,...,40
```

### Modifying the Python Bridge (`stockfish_bridge.py`)
To use this new feature, you would modify the parser in `_go_and_get_score` in your Python script to look for the `trajectory` keyword in the output string, split it by commas, and feed that array directly into your PyTorch dataset as an additional powerful external metric!
