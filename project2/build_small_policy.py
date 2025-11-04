#!/usr/bin/env python3
"""
Build a model of small.csv, solve it by value iteration, and write small.policy.

How to run: python .\build_small_policy.py --in data/small.csv --out <policy file name>.policy
"""

import argparse
import csv
from collections import defaultdict
from typing import List, Tuple
import numpy as np
from problem_configs import SMALL
from constants import ALPHA, EPS, MAX_ITERS


#problem constants (from the project description)
cfg = SMALL
S, A, GAMMA = cfg["S"], cfg["A"], cfg["GAMMA"]

def load_transitions(path: str) -> List[Tuple[int, int, float, int]]:
    """
    Read CSV rows. NOTE: States/actions are 1-indexed in the file.
    Returns a list of 0-indexed tuples in the format(s, a, r, sp).
    """
    trips = []
    with open(path, "r", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or len(row) < 4:
                continue
            s, a, r, sp = row[:4]
            try:
                s = int(float(s))
                a = int(float(a))
                r = float(r)
                sp = int(float(sp))
            except ValueError:
                #skip non-numeric rows if a header/blank line appears
                continue
            assert 1 <= s <= S and 1 <= sp <= S, f"State out of range in row: {row}"
            assert 1 <= a <= A, f"Action out of range in row: {row}"
            trips.append((s - 1, a - 1, r, sp - 1))
    if not trips:
        raise ValueError("No valid rows parsed from CSV.")
    return trips


def estimate_model(trips: List[Tuple[int, int, float, int]],
                   alpha: float = ALPHA) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    In this function, we estimate P(s'|s,a) and expected reward R_sa(s,a) from samples.
    Reward is tied to (s,a,sp); we store the mean per (s,a,sp) and then we
    take the expectation under P for each (s,a).

    Return:
    P: (S, A, S) transition probabilities
    R_sa: (S, A) expected one-step reward
    Nsa: (S, A) counts per (s,a)
    """
    N = np.zeros((S, A, S), dtype=np.float64)        #counts for (s,a,sp)
    R_sasp = np.zeros((S, A, S), dtype=np.float64)   #mean reward per (s,a,sp)
    seen = np.zeros((S, A, S), dtype=bool)

    for s, a, r, sp in trips:
        N[s, a, sp] += 1.0
        if not seen[s, a, sp]:
            R_sasp[s, a, sp] = r
            seen[s, a, sp] = True
        else:
            #average in case duplicates/noise; spec says deterministic per (s,a,sp)
            cnt = N[s, a, sp]
            R_sasp[s, a, sp] = (R_sasp[s, a, sp] * (cnt - 1) + r) / cnt

    Nsa = N.sum(axis=2)                 # (S,A)
    P = np.zeros_like(N)                # (S,A,S)

    #light smoothing. Add alpha mass to self-loop for stability on sparse pairs
    for s in range(S):
        for a in range(A):
            denom = Nsa[s, a] + alpha
            if denom == 0.0:
                # Unseen (s,a): conservative self-loop with zero reward handled later
                P[s, a, s] = 1.0
            else:
                P[s, a, :] = N[s, a, :] / denom
                P[s, a, s] += alpha / denom

    # Expected reward for (s,a) = sum_{s'} P(s'|s,a) * R(s,a,s')
    R_sa = (P * R_sasp).sum(axis=2)
    return P, R_sa, Nsa


def value_iteration(P: np.ndarray,
                    R_sa: np.ndarray,
                    gamma: float = GAMMA,
                    eps: float = EPS,
                    max_iters: int = MAX_ITERS) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Standard value iteration on a learned tabular model.

    Return:
    V: (S,) value function
    pi: (S,) greedy actions (0-indexed)
    iters: number of iterations executed
    """
    V = np.zeros(S, dtype=np.float64)
    iters = 0
    for it in range(1, max_iters + 1):
        #Q(s,a) = R(s,a) + gamma * sum_{s'} P(s'|s,a) V(s')
        Q = R_sa + gamma * np.tensordot(P, V, axes=([2], [0]))  #output has shape (S,A)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < eps:
            V = V_new
            iters = it
            break
        V = V_new
    else:
        iters = max_iters

    pi = Q.argmax(axis=1)  #0-indexed action per state
    return V, pi, iters


def write_policy(pi: np.ndarray, out_path: str) -> None:
    """
    This function writes the deterministic policy, 1 action (1..A) per line, total S lines.
    """
    assert pi.shape == (S,)
    with open(out_path, "w", newline="") as f:
        for s in range(S):
            f.write(f"{int(pi[s]) + 1}\n")


def main():
    parser = argparse.ArgumentParser()
    #run on csv and write to output .policy file:
    parser.add_argument("--in", dest="in_path", required=True, help="Path to small.csv")
    parser.add_argument("--out", dest="out_path", default="small.policy",
                        help="Output policy file path (default: small.policy)")
    args = parser.parse_args()

    trips = load_transitions(args.in_path)
    P, R_sa, Nsa = estimate_model(trips, alpha=ALPHA)
    V, pi, iters = value_iteration(P, R_sa, gamma=GAMMA, eps=EPS, max_iters=MAX_ITERS)
    write_policy(pi, args.out_path)

    seen_sa_frac = float((Nsa > 0).mean())
    min_seen = int(Nsa[Nsa > 0].min()) if (Nsa > 0).any() else 0
    max_seen = int(Nsa.max())
    num_unseen = int((Nsa == 0).sum())
    print(f"samples: {len(trips)}")
    print(f"states seen: {len(np.unique([s for s,_,_,_ in trips]))} / {S}")
    print(f"actions seen: {len(np.unique([a for _,a,_,_ in trips]))} / {A}")
    print(f"value-iteration iterations: {iters}")
    print(f"(s,a) coverage: {seen_sa_frac:.3f} (unseen pairs: {num_unseen})")
    print(f"min/ max counts over seen (s,a): {min_seen} / {max_seen}")
    print(f"wrote policy to: {args.out_path}")


if __name__ == "__main__":
    main()
