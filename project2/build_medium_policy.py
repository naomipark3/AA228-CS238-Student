import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# ---------- helpers ----------
def decode_pos_vel(s):
    """Decode state index to (pos, vel). Both 0-indexed."""
    s0 = int(s) - 1
    pos = s0 % 500
    vel = s0 // 500
    return pos, vel

def load_medium_csv(path):
    """Load CSV data properly."""
    df = pd.read_csv(path)
    for c in ["s", "a", "r", "sp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["s", "a", "r", "sp"]).copy()
    df = df.astype({"s": int, "a": int, "r": float, "sp": int})
    return df

def infer_goal_states(df, reward_threshold=50000):
    """Identify goal states from rewards."""
    high_reward = df[df["r"] >= reward_threshold]
    goal_next_states = set(high_reward["sp"].unique())
    
    # Get goal position
    if len(goal_next_states) > 0:
        positions = [decode_pos_vel(s)[0] for s in goal_next_states]
        goal_pos = int(np.median(positions))
    else:
        goal_pos = 469  # Fallback based on analysis
    
    return goal_next_states, goal_pos

def build_terminal_mask(n_states, goal_states, goal_pos):
    """Mark states as terminal based on position."""
    term_flag = np.zeros(n_states + 1, dtype=bool)
    
    # Mark known goal states
    for s in goal_states:
        if 1 <= s <= n_states:
            term_flag[s] = True
    
    # Mark all states with position >= goal_pos - 3 as terminal
    # (generous to account for discretization errors)
    pos_threshold = max(goal_pos - 3, 465)
    for s in range(1, n_states + 1):
        pos, _ = decode_pos_vel(s)
        if pos >= pos_threshold:
            term_flag[s] = True
    
    return term_flag

def fitted_q_iteration(
    df,
    n_states=50000,
    n_actions=7,
    gamma=1.0,
    iters=100,
    terminal_flag=None,
):
    """
    Fitted Q-iteration with proper handling of terminal states.
    """
    if terminal_flag is None:
        terminal_flag = np.zeros(n_states + 1, dtype=bool)

    # Pre-compute grouped data
    grouped = df.groupby(["s", "a"])
    sa_data = {}
    for (s, a), grp in grouped:
        rewards = grp["r"].values.astype(np.float64)
        next_states = grp["sp"].values.astype(np.int64)
        sa_data[(s, a)] = (rewards, next_states)

    # Initialize Q with reasonable estimates
    Q = np.zeros((n_states + 1, n_actions + 1), dtype=np.float64)
    
    # Run value iteration
    for it in range(iters):
        Q_old = Q.copy()
        
        for (s, a), (rewards, next_states) in sa_data.items():
            # Get max Q-value for each next state
            max_next_q = Q[next_states, 1:n_actions+1].max(axis=1)
            
            # Zero out terminal states
            max_next_q[terminal_flag[next_states]] = 0.0
            
            # Bellman update
            targets = rewards + gamma * max_next_q
            Q[s, a] = targets.mean()
        
        # Check convergence
        max_diff = np.abs(Q - Q_old).max()
        
        if (it + 1) % 20 == 0:
            print(f"  Iteration {it+1}: max change = {max_diff:.2f}")
        
        if max_diff < 1e-3 and it > 20:
            print(f"  Converged at iteration {it+1}")
            break
    
    return Q

def get_smart_default_action(df, visited_states):
    """
    Choose a smart default action for unvisited states.
    Prefer action 4 (no acceleration) as it's safest.
    """
    # Action 4 has reward 0 and is safest
    return 4

def propagate_values_to_neighbors(Q, df, n_states, n_actions):
    """
    For unvisited states, interpolate from nearby visited states.
    This helps with generalization.
    """
    visited = set(df["s"].unique())
    Q_prop = Q.copy()
    
    # Simple propagation: for each unvisited state, find closest visited state
    # and copy its policy (this is a simplified version)
    for s in range(1, n_states + 1):
        if s not in visited:
            # Get position and velocity
            pos, vel = decode_pos_vel(s)
            
            # Find nearby states that were visited
            nearby_states = []
            for dp in range(-2, 3):
                for dv in range(-2, 3):
                    neighbor_pos = pos + dp
                    neighbor_vel = vel + dv
                    if 0 <= neighbor_pos < 500 and 0 <= neighbor_vel < 100:
                        neighbor_s = 1 + neighbor_pos + 500 * neighbor_vel
                        if neighbor_s in visited:
                            nearby_states.append(neighbor_s)
            
            if nearby_states:
                # Average Q-values from nearby states
                neighbor_qs = Q[nearby_states, 1:n_actions+1]
                Q_prop[s, 1:n_actions+1] = neighbor_qs.mean(axis=0)
    
    return Q_prop

def dump_policy(Q, df, out_path, n_states=50000, n_actions=7, use_propagation=False):
    """Extract greedy policy from Q-values."""
    visited = set(df["s"].unique())
    
    # Optionally propagate values
    if use_propagation:
        print("Propagating Q-values to unvisited states...")
        Q = propagate_values_to_neighbors(Q, df, n_states, n_actions)
    
    default_action = get_smart_default_action(df, visited)
    
    lines = []
    unvisited_count = 0
    action_counts = Counter()
    
    for s in range(1, n_states + 1):
        q_values = Q[s, 1:n_actions+1]
        
        if np.all(q_values == 0):
            # Unvisited state
            best_action = default_action
            unvisited_count += 1
        else:
            # Greedy selection
            best_action = int(np.argmax(q_values) + 1)
        
        lines.append(str(best_action))
        action_counts[best_action] += 1
    
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Unvisited states: {unvisited_count}/{n_states} ({100*unvisited_count/n_states:.1f}%)")
    print(f"Default action: {default_action}")
    print(f"Action distribution: {dict(sorted(action_counts.items()))}")
    
    return out_path

# ---------- main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="medium.csv")
    ap.add_argument("--out", default="medium.policy")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--propagate", action="store_true", help="Propagate Q-values to unvisited states")
    args = ap.parse_args()

    print("="*70)
    print("MEDIUM PROBLEM - OPTIMIZED Q-LEARNING")
    print("="*70)
    
    # Load data
    df = load_medium_csv(args.csv)
    print(f"Loaded {len(df):,} transitions")
    print(f"Unique states: {df['s'].nunique()}")
    print(f"Unique (s,a) pairs: {df.groupby(['s', 'a']).ngroups}")
    
    n_actions = 7
    n_states = 50000

    # Identify goal states
    goal_states, goal_pos = infer_goal_states(df)
    print(f"\nGoal states: {len(goal_states)}")
    print(f"Goal position index: {goal_pos}")
    
    # Build terminal mask
    term_flag = build_terminal_mask(n_states, goal_states, goal_pos)
    print(f"Total terminal states: {term_flag.sum()}")

    # Run Q-learning
    print(f"\nRunning Q-iteration ({args.iters} max iterations)...")
    Q = fitted_q_iteration(
        df=df,
        n_states=n_states,
        n_actions=n_actions,
        gamma=1.0,
        iters=args.iters,
        terminal_flag=term_flag,
    )
    
    # Statistics
    visited = set(df["s"].unique())
    visited_q_vals = [Q[s, 1:n_actions+1].max() for s in visited]
    print(f"\nQ-value stats (visited states):")
    print(f"  Mean: {np.mean(visited_q_vals):.2f}")
    print(f"  Max: {np.max(visited_q_vals):.2f}")
    
    # Extract policy
    print("\nExtracting policy...")
    out_path = dump_policy(Q, df, args.out, n_states, n_actions, use_propagation=args.propagate)
    
    print(f"\n✓ Wrote policy to: {out_path}")