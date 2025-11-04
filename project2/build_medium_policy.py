import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# ---------- helpers ----------
def decode_pos_vel(s):
    s0 = int(s) - 1
    pos = s0 % 500
    vel = s0 // 500
    return pos, vel

def load_medium_csv(path):
    raw = pd.read_csv(path, header=None)
    # tolerate an accidental header row
    raw = raw.rename(columns={0:"s",1:"a",2:"r",3:"sp"})
    for c in ["s","a","r","sp"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    df = raw.dropna(subset=["s","a","r","sp"]).copy()
    df = df.astype({"s":int, "a":int, "r":float, "sp":int})
    return df

def infer_terminals(df, mode="reward_spike", top_frac=0.001):
    """
    Returns:
      terminal_states: set of s' considered terminal for bootstrap=0
      pos_goal_idx: optional integer (pos index cutoff) for explainability/debug
    """
    n = len(df)
    # very high rewards are typically terminals in MountainCar
    thr = df["r"].quantile(1.0 - top_frac)
    high = df[df["r"] >= thr]
    if high.empty:
        return set(), None

    # look at where those land in pos space
    high_pos = [decode_pos_vel(int(sp))[0] for sp in high["sp"].values]
    pos_counts = Counter(high_pos)
    if not pos_counts:
        return set(), None
    # a robust "goal pos" guess: 95th percentile of those high-reward positions
    pos_goal_idx = int(np.percentile(high_pos, 95))

    if mode == "reward_spike":
        # terminal = any s' seen in a high-reward row
        terminal_states = set(high["sp"].values.tolist())
    elif mode == "goal_position":
        # terminal = any s' whose position is at/above the inferred goal pos
        terminal_states = set(
            sp for sp in df["sp"].values if decode_pos_vel(sp)[0] >= pos_goal_idx
        )
    elif mode == "hybrid":
        # union of both
        reward_term = set(high["sp"].values.tolist())
        goal_term = set(sp for sp in df["sp"].values if decode_pos_vel(sp)[0] >= pos_goal_idx)
        terminal_states = reward_term | goal_term
    else:
        raise ValueError("mode must be one of: reward_spike | goal_position | hybrid")

    return terminal_states, pos_goal_idx

def fitted_q_iteration(df, n_states=50000, n_actions=7, gamma=1.0, iters=25,
                       terminal_states=None):
    """
    Offline fitted tabular Q-iteration:
      Q_new(s,a) = average_{(s,a)->(r,sp) in data} [ r + (0 if terminal(sp) else max_a' Q(sp,a')) ]
    """
    if terminal_states is None:
        terminal_states = set()
    term_flag = np.zeros(n_states+1, dtype=bool)
    for t in terminal_states:
        if 1 <= t <= n_states:
            term_flag[t] = True

    # group indices for each (s,a)
    # (materializing once keeps it fast)
    grouped = df.groupby(["s","a"])
    groups = {key: grp[["r","sp"]].values for key, grp in grouped}

    Q = np.zeros((n_states+1, n_actions+1), dtype=np.float32)  # 1-based

    for _ in range(iters):
        Q_new = Q.copy()
        for (s,a), arr in groups.items():
            # arr: Nx2 of [r, sp]
            rs = arr[:,0].astype(np.float32)
            sps = arr[:,1].astype(np.int64)
            max_next = Q[sps, :].max(axis=1)
            max_next[term_flag[sps]] = 0.0
            targets = rs + gamma * max_next
            Q_new[s, a] = float(targets.mean())
        Q = Q_new
    return Q

def dump_policy(Q, out_path, n_states=50000):
    # choose argmax; if ties / all-zero row, fall back to action 1
    A = Q.shape[1] - 1
    policy_lines = []
    for s in range(1, n_states+1):
        qs = Q[s, 1:A+1]
        if np.all(qs == 0):
            policy_lines.append("1")
        else:
            policy_lines.append(str(int(np.argmax(qs) + 1)))
    with open(out_path, "w") as f:
        f.write("\n".join(policy_lines))
    return out_path

# ---------- main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/medium.csv")
    ap.add_argument("--out", default="medium.policy")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--term_mode", choices=["reward_spike","goal_position","hybrid"], default="hybrid")
    ap.add_argument("--top_frac", type=float, default=0.001, help="fraction of rewards treated as 'very high'")
    args = ap.parse_args()

    df = load_medium_csv(args.csv)
    # sanity: ensure actions range is 1..7 (if fewer seen, still emit 7-way Q)
    n_actions = max(7, int(df["a"].max()))

    terminal_states, pos_goal_idx = infer_terminals(df, mode=args.term_mode, top_frac=args.top_frac)

    Q = fitted_q_iteration(
        df=df,
        n_states=50000,
        n_actions=n_actions,
        gamma=1.0,           # undiscounted
        iters=args.iters,
        terminal_states=terminal_states
    )

    out_path = dump_policy(Q, args.out, n_states=50000)

    # brief report
    print(f"wrote {out_path}")
    print(f"samples: {len(df):,} | actions_seen: {sorted(df['a'].unique().tolist())}")
    if pos_goal_idx is not None:
        print(f"inferred goal pos index ≈ {pos_goal_idx} (0..499)")