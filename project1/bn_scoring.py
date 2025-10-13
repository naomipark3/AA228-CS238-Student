
from typing import Dict, List, Tuple
import pandas as pd
from math import lgamma
from itertools import product

def load_discrete_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in df.columns:
        df[c] = df[c].astype(int)
    return df

def cardinalities(df: pd.DataFrame):
    return {col: int(df[col].max()) for col in df.columns}

def all_parent_configs(df: pd.DataFrame, parents: List[str], r: Dict[str, int]):
    if not parents:
        return [()]
    from itertools import product
    ranges = [range(1, r[p] + 1) for p in parents]
    return list(product(*ranges))

def counts_for_node_given_parents(df: pd.DataFrame, node: str, parents: List[str], r: Dict[str, int]):
    result = {cfg: {k: 0 for k in range(1, r[node] + 1)} for cfg in all_parent_configs(df, parents, r)}
    if not parents:
        vc = df[node].value_counts()
        for k, cnt in vc.items():
            result[()][int(k)] = int(cnt)
        return result
    grouped = df.groupby(parents + [node]).size().reset_index(name='count')
    for _, row in grouped.iterrows():
        cfg = tuple(int(row[p]) for p in parents)
        k = int(row[node])
        result[cfg][k] = int(row['count'])
    return result

def bayesian_score_dirichlet_uniform(df: pd.DataFrame, graph: Dict[str, List[str]]) -> float:
    r = cardinalities(df)
    score = 0.0
    for node in df.columns:
        parents = graph.get(node, [])
        parents = [p for p in parents if p in df.columns and p != node]
        cfg_counts = counts_for_node_given_parents(df, node, parents, r)
        r_i = r[node]
        alpha_ij = r_i
        for cfg, k_counts in cfg_counts.items():
            N_ij = sum(k_counts.values())
            score += lgamma(alpha_ij) - lgamma(alpha_ij + N_ij)
            for k in range(1, r_i + 1):
                N_ijk = k_counts.get(k, 0)
                score += lgamma(1 + N_ijk) - lgamma(1)
    return float(score)
