import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

# ---------- State encoding/decoding ----------
def decode_state(s):
    """Decode state XXYYZZ -> (x, y, z) where x∈[1,30], y∈[1,20], z∈[1,20]"""
    s_str = f"{s:06d}"
    x = int(s_str[0:2])
    y = int(s_str[2:4])
    z = int(s_str[4:6])
    return x, y, z

def encode_state(x, y, z):
    """Encode (x, y, z) -> state XXYYZZ"""
    return x * 10000 + y * 100 + z

def is_valid_state(s):
    """Check if state is valid (within grid bounds)"""
    x, y, z = decode_state(s)
    return 1 <= x <= 30 and 1 <= y <= 20 and 1 <= z <= 20

# ---------- Data loading ----------
def load_large_csv(path):
    """Load CSV data."""
    df = pd.read_csv(path)
    for c in ["s", "a", "r", "sp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["s", "a", "r", "sp"]).copy()
    df = df.astype({"s": int, "a": int, "r": float, "sp": int})
    return df

# ---------- Model building ----------
def build_ml_model(df, n_actions=9):
    """
    Build maximum likelihood model from data.
    Returns: transition_counts, reward_sums, state_action_counts, visited_states
    """
    # Use dictionaries for sparse storage
    transition_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # N[s][a][sp]
    reward_sums = defaultdict(lambda: defaultdict(float))      # ρ[s][a]
    state_action_counts = defaultdict(lambda: defaultdict(int)) # N[s][a]
    
    print("Building maximum likelihood model from data...")
    for _, row in df.iterrows():
        s, a, r, sp = int(row['s']), int(row['a']), row['r'], int(row['sp'])
        
        transition_counts[s][a][sp] += 1
        reward_sums[s][a] += r
        state_action_counts[s][a] += 1
    
    visited_states = set(df['s'].unique())
    
    print(f"  Observed {len(visited_states)} unique states")
    print(f"  Observed {len(df)} transitions")
    
    return transition_counts, reward_sums, state_action_counts, visited_states

def get_transition_probs(s, a, transition_counts, state_action_counts):
    """Get transition probabilities for (s, a) pair"""
    if s in state_action_counts and a in state_action_counts[s]:
        n_sa = state_action_counts[s][a]
        probs = {}
        for sp, count in transition_counts[s][a].items():
            probs[sp] = count / n_sa
        return probs
    return None

def get_reward(s, a, reward_sums, state_action_counts):
    """Get expected reward for (s, a) pair"""
    if s in state_action_counts and a in state_action_counts[s]:
        n_sa = state_action_counts[s][a]
        return reward_sums[s][a] / n_sa
    return 0.0

# ---------- Value iteration ----------
def value_iteration(
    all_states,
    n_actions,
    transition_counts,
    reward_sums,
    state_action_counts,
    visited_states,
    gamma=0.95,
    max_iters=100,
    theta=1e-4
):
    """
    Run value iteration on the learned model.
    """
    print(f"\nRunning value iteration (max {max_iters} iterations)...")
    
    # Initialize value function
    V = defaultdict(float)  # Default value is 0
    
    # For unvisited states, we'll compute values on-demand
    # Focus iterations on visited states and their neighbors
    
    for iteration in range(max_iters):
        V_old = V.copy()
        max_delta = 0
        
        # Update all visited states
        for s in visited_states:
            if not is_valid_state(s):
                continue
                
            # Compute Q-values for all actions
            q_values = []
            for a in range(1, n_actions + 1):
                # Get expected reward
                r = get_reward(s, a, reward_sums, state_action_counts)
                
                # Get transition probabilities
                trans_probs = get_transition_probs(s, a, transition_counts, state_action_counts)
                
                if trans_probs is not None:
                    # Use learned transition model
                    expected_future = sum(prob * V_old[sp] for sp, prob in trans_probs.items())
                else:
                    # No data for this action - assume it stays in place
                    expected_future = V_old[s]
                
                q_value = r + gamma * expected_future
                q_values.append(q_value)
            
            # Update value function
            V[s] = max(q_values)
            max_delta = max(max_delta, abs(V[s] - V_old[s]))
        
        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}: max delta = {max_delta:.6f}")
        
        # Check convergence
        if max_delta < theta:
            print(f"  Converged at iteration {iteration + 1}")
            break
    
    return V

# ---------- Policy extraction ----------
def extract_policy(
    all_states,
    n_actions,
    V,
    transition_counts,
    reward_sums,
    state_action_counts,
    visited_states,
    gamma=0.95
):
    """
    Extract greedy policy from value function.
    For unvisited states, use heuristics based on the grid structure.
    """
    print("\nExtracting policy...")
    
    policy = {}
    
    # Default action based on observed data
    # Actions 5-9 mostly do nothing, so default to action that moves
    # Actions 1,2,3,4 are the primary movement actions
    
    # Analyze which actions are most rewarding
    action_rewards = defaultdict(list)
    for s in visited_states:
        for a in range(1, n_actions + 1):
            if s in state_action_counts and a in state_action_counts[s]:
                r = get_reward(s, a, reward_sums, state_action_counts)
                action_rewards[a].append(r)
    
    # Compute average reward per action
    avg_action_rewards = {}
    for a in range(1, n_actions + 1):
        if action_rewards[a]:
            avg_action_rewards[a] = np.mean(action_rewards[a])
        else:
            avg_action_rewards[a] = 0.0
    
    print(f"  Average rewards by action: {avg_action_rewards}")
    
    # Default action for completely unvisited states
    default_action = max(avg_action_rewards, key=avg_action_rewards.get)
    
    # First pass: compute policy for all visited states
    visited_policy = {}
    for s in visited_states:
        if not is_valid_state(s):
            continue
        
        # Compute greedy action based on learned model
        q_values = []
        for a in range(1, n_actions + 1):
            r = get_reward(s, a, reward_sums, state_action_counts)
            trans_probs = get_transition_probs(s, a, transition_counts, state_action_counts)
            
            if trans_probs is not None:
                expected_future = sum(prob * V[sp] for sp, prob in trans_probs.items())
            else:
                expected_future = V[s]  # Assume stays in place
            
            q_value = r + gamma * expected_future
            q_values.append(q_value)
        
        best_action = np.argmax(q_values) + 1
        visited_policy[s] = best_action
    
    # Second pass: assign policy for all states
    unvisited_count = 0
    
    for s in all_states:
        if not is_valid_state(s):
            # Invalid state - use default
            policy[s] = default_action
            continue
        
        if s in visited_policy:
            # Visited state - use computed policy
            policy[s] = visited_policy[s]
        else:
            # Unvisited state - use nearest neighbor heuristic
            unvisited_count += 1
            
            # Try to find a nearby visited state
            x, y, z = decode_state(s)
            best_neighbor = None
            best_value = float('-inf')
            
            # Check immediate neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if 1 <= nx <= 30 and 1 <= ny <= 20 and 1 <= nz <= 20:
                            neighbor = encode_state(nx, ny, nz)
                            if neighbor in visited_policy and V[neighbor] > best_value:
                                best_neighbor = neighbor
                                best_value = V[neighbor]
            
            if best_neighbor is not None:
                # Copy policy from best neighbor
                policy[s] = visited_policy[best_neighbor]
            else:
                # No nearby visited state - use default
                policy[s] = default_action
    
    print(f"  Unvisited states: {unvisited_count:,} / {len(all_states):,}")
    print(f"  Default action: {default_action}")
    
    # Show policy distribution
    from collections import Counter
    policy_dist = Counter(policy.values())
    print(f"  Policy distribution: {dict(sorted(policy_dist.items()))}")
    
    return policy

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_file", default="data/large.csv")
    parser.add_argument("--out", default="large.policy")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.95)
    args = parser.parse_args()
    
    print("="*70)
    print("LARGE PROBLEM - MODEL-BASED RL")
    print("="*70)
    
    # Load data
    df = load_large_csv(args.input_file)
    print(f"Loaded {len(df):,} transitions")
    
    n_actions = 9
    n_states = 302020
    
    # Build maximum likelihood model
    transition_counts, reward_sums, state_action_counts, visited_states = build_ml_model(df, n_actions)
    
    # Generate all valid states
    print("\nGenerating all valid states...")
    all_states = []
    for x in range(1, 31):
        for y in range(1, 21):
            for z in range(1, 21):
                s = encode_state(x, y, z)
                all_states.append(s)
    
    # Pad to exactly 302020 states
    while len(all_states) < n_states:
        all_states.append(len(all_states) + 1)
    
    print(f"Total states to generate policy for: {len(all_states):,}")
    
    # Run value iteration
    V = value_iteration(
        all_states,
        n_actions,
        transition_counts,
        reward_sums,
        state_action_counts,
        visited_states,
        gamma=args.gamma,
        max_iters=args.iters
    )
    
    # Extract policy
    policy = extract_policy(
        all_states,
        n_actions,
        V,
        transition_counts,
        reward_sums,
        state_action_counts,
        visited_states,
        gamma=args.gamma
    )
    
    # Write policy file
    print(f"\nWriting policy to {args.out}...")
    with open(args.out, "w") as f:
        for s in range(1, n_states + 1):
            f.write(f"{policy.get(s, 1)}\n")
    
    print(f"\n✓ Wrote policy to: {args.out}")

if __name__ == "__main__":
    main()