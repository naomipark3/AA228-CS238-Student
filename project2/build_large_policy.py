"""
Build a model of large.csv, solve it with value iteration, and write large.policy.

How to run: python .\build_large_policy.py --in data/large.csv --out <policy file name>.policy
"""

import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from problem_configs import LARGE
from datetime import datetime

cfg = LARGE
n_states = cfg["S"]
n_actions = cfg["A"]
gamma_default = cfg["GAMMA"]

#Stage encoding and decoding based on csv structure:
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

#Load data using pandas:
def load_large_csv(path):
    """Load CSV data."""
    df = pd.read_csv(path)
    for c in ["s", "a", "r", "sp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["s", "a", "r", "sp"]).copy()
    df = df.astype({"s": int, "a": int, "r": float, "sp": int})
    return df

#Build Maximum Likelihood Model-Based RL (see 16.1, 16.2 from Algorithms for Validation)
def build_ml_model(df, n_actions=n_actions):
    """
    Build maximum likelihood model from data.
    Returns: transition_counts, reward_sums, state_action_counts, visited_states
    """
    #dictionaries for *sparse storage*
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
    
    print(f"Observed {len(visited_states)} unique states")
    print(f"Observed {len(df)} transitions")
    
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

#value iteration function
def value_iteration(
    all_states,
    n_actions,
    transition_counts,
    reward_sums,
    state_action_counts,
    visited_states,
    gamma=gamma_default,
    max_iters=100,
    theta=1e-4
):
    """
    Run value iteration on the learned model.
    """
    print(f"Running value iteration (max {max_iters} iterations)...")
    
    #initialize value function
    V = defaultdict(float)  #**default value is 0
    
    #for unvisited states, we will compute values ON-DEMAND
    #Then, focus iterations on visited states and their neighbors
    
    for iteration in range(max_iters):
        V_old = V.copy()
        max_delta = 0
        
        #update all visited states
        for s in visited_states:
            if not is_valid_state(s):
                continue
                
            #compute Q-values for all actions
            q_values = []
            for a in range(1, n_actions + 1):
                #get expected reward
                r = get_reward(s, a, reward_sums, state_action_counts)
                
                #get transition probabilities
                trans_probs = get_transition_probs(s, a, transition_counts, state_action_counts)
                
                if trans_probs is not None:
                    #use learned transition model
                    expected_future = sum(prob * V_old[sp] for sp, prob in trans_probs.items())
                else:
                    #no data for this action, so we assume it stays in place
                    expected_future = V_old[s]
                
                q_value = r + gamma * expected_future
                q_values.append(q_value)
            
            #update value function
            V[s] = max(q_values)
            max_delta = max(max_delta, abs(V[s] - V_old[s]))
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}: max delta = {max_delta:.6f}")
        
        # Check convergence
        if max_delta < theta:
            print(f"Converged at iteration {iteration + 1}")
            break
    
    return V

#Policy extraction
def extract_policy(
    all_states,
    n_actions,
    V,
    transition_counts,
    reward_sums,
    state_action_counts,
    visited_states,
    gamma=gamma_default
):
    """
    At this point, we will extract greedy policy from value function.
    For unvisited states, we will use heuristics based on the grid structure
    we detected in the large.csv.
    """
    policy = {}
    
    #default action based on observed data: Actions 5-9 mostly do nothing, so default to action that moves.
    #Actions 1,2,3,4 are the primary movement actions
    
    #analyze which actions are most rewarding as follows:
    action_rewards = defaultdict(list)
    for s in visited_states:
        for a in range(1, n_actions + 1):
            if s in state_action_counts and a in state_action_counts[s]:
                r = get_reward(s, a, reward_sums, state_action_counts)
                action_rewards[a].append(r)
    
    #compute average reward per action
    avg_action_rewards = {}
    for a in range(1, n_actions + 1):
        if action_rewards[a]:
            avg_action_rewards[a] = np.mean(action_rewards[a])
        else:
            avg_action_rewards[a] = 0.0
    
    print(f"average rewards by action: {avg_action_rewards}")
    
    #default action for completely unvisited states
    default_action = max(avg_action_rewards, key=avg_action_rewards.get)
    
    #in first pass, we compute policy for all visited states
    visited_policy = {}
    for s in visited_states:
        if not is_valid_state(s):
            continue
        
        #compute greedy action based on learned model
        q_values = []
        for a in range(1, n_actions + 1):
            r = get_reward(s, a, reward_sums, state_action_counts)
            trans_probs = get_transition_probs(s, a, transition_counts, state_action_counts)
            
            if trans_probs is not None:
                expected_future = sum(prob * V[sp] for sp, prob in trans_probs.items())
            else:
                expected_future = V[s]  #assume stays in place
            
            q_value = r + gamma * expected_future
            q_values.append(q_value)
        
        best_action = np.argmax(q_values) + 1
        visited_policy[s] = best_action
    
    #In second pass, we assign policy for all states
    unvisited_count = 0
    
    for s in all_states:
        if not is_valid_state(s):
            #invalid state, so use default
            policy[s] = default_action
            continue
        
        if s in visited_policy:
            #visited state --> use computed policy
            policy[s] = visited_policy[s]
        else:
            #unvisited state --> use nearest neighbor heuristic
            unvisited_count += 1
            
            #we will try to find a nearby visited state:
            x, y, z = decode_state(s)
            best_neighbor = None
            best_value = float('-inf')
            
            #check immediate neighbors:
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
                #copy policy from best neighbor as follows:
                policy[s] = visited_policy[best_neighbor]
            else:
                #no nearby visited state --> use default
                policy[s] = default_action
    
    print(f"Unvisited states: {unvisited_count:,} / {len(all_states):,}")
    print(f"Default action: {default_action}")
    
    #need to show policy distributions:
    policy_dist = Counter(policy.values())
    print(f"Policy distribution: {dict(sorted(policy_dist.items()))}")
    
    return policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_file", default="data/large.csv")
    parser.add_argument("--out", default="large.policy")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=gamma_default)
    args = parser.parse_args()

    start_time = datetime.now()
    
    #load data using pandas:
    df = load_large_csv(args.input_file)
    
    #build maximum likelihood model by calling our model function
    transition_counts, reward_sums, state_action_counts, visited_states = build_ml_model(df, n_actions)
    
    #generate all valid states
    all_states = []
    for x in range(1, 31):
        for y in range(1, 21):
            for z in range(1, 21):
                s = encode_state(x, y, z)
                all_states.append(s)
    
    #pad to EXACTLY 302020 states
    while len(all_states) < n_states:
        all_states.append(len(all_states) + 1)
    
    print(f"Total states to generate policy for: {len(all_states):,}")
    
    #run value iteration
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
    
    #extract policy as follows:
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
    
    #write policy file to specified destination:
    with open(args.out, "w") as f:
        for s in range(1, n_states + 1):
            f.write(f"{policy.get(s, 1)}\n")
    
    print(f"Wrote policy to: {args.out}")

    end_time = datetime.now()  # <-- stop timing here
    print(f"Total runtime: {end_time - start_time} "
          f"({(end_time - start_time).total_seconds():.3f} s)")

if __name__ == "__main__":
    main()