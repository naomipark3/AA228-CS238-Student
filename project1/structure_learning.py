import itertools
import networkx as nx
from bn_scoring import load_discrete_data, bayesian_score_dirichlet_uniform

def score_dag(df, dag):
    """Convert DAG into a structure of child:[parents] (i.e. map each child to its list of parents) and then compute the Bayesian score."""
    parent_map = {n: [] for n in dag.nodes}
    for u, v in dag.edges:
        parent_map[v].append(u)
    return bayesian_score_dirichlet_uniform(df, parent_map)

def neighbors(dag, max_parents=3):
    """Generate all DAGs that differ by one edge (add/remove/reverse)."""
    nodes = list(dag.nodes)
    existing = set(dag.edges)

    #try removing each edge as follows: 
    for u, v in list(dag.edges):
        g2 = dag.copy()
        g2.remove_edge(u, v)
        yield g2

    #try adding each possible edge (if not already present) as follows:
    for u, v in itertools.permutations(nodes, 2):
        if (u, v) in existing or (v, u) in existing:
            continue
        g2 = dag.copy()
        g2.add_edge(u, v)
        if nx.is_directed_acyclic_graph(g2) and len(list(g2.predecessors(v))) <= max_parents:
            yield g2

    #try reversing existing edges as follows:
    for u, v in list(dag.edges):
        g2 = dag.copy()
        g2.remove_edge(u, v)
        g2.add_edge(v, u)
        if nx.is_directed_acyclic_graph(g2) and len(list(g2.predecessors(u))) <= max_parents:
            yield g2

def hill_climb(df, max_iters=200, max_parents=3):
    """Greedy hill climbing using the Bayesian score."""
    cols = list(df.columns)
    dag = nx.DiGraph()
    dag.add_nodes_from(cols)

    best_score = score_dag(df, dag)
    improved = True
    iteration = 0

    while improved and iteration < max_iters:
        improved = False
        iteration += 1
        best_neighbor, best_neighbor_score = None, best_score

        for g2 in neighbors(dag, max_parents=max_parents):
            s2 = score_dag(df, g2)
            if s2 > best_neighbor_score:
                best_neighbor_score = s2
                best_neighbor = g2

        if best_neighbor is not None:
            dag = best_neighbor
            best_score = best_neighbor_score
            improved = True
            print(f"Iteration {iteration}: improved score = {best_score:.2f}")

    return dag, best_score
