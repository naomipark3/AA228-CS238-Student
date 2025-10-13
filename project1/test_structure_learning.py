from bn_scoring import load_discrete_data
from structure_learning import hill_climb

df = load_discrete_data("data/large.csv")
dag, best_score = hill_climb(df, max_iters=100, max_parents=3)

print("Final best score:", best_score)
print("Edges:", list(dag.edges))