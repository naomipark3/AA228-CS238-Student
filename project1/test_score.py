from bn_scoring import load_discrete_data, bayesian_score_dirichlet_uniform

def empty_graph(df):
    return {c: [] for c in df.columns}

def simple_graph(df):
    cols = list(df.columns)
    return {c: ([] if c != cols[-1] else cols[:-1]) for c in cols}

for name in ["data/small.csv", "data/medium.csv", "data/large.csv"]:
    try:
        df = load_discrete_data(name)
    except FileNotFoundError:
        print(f"{name}: missing")
        continue
    print(name, df.shape)
    print("empty:", bayesian_score_dirichlet_uniform(df, empty_graph(df)))
    # print("simple:", bayesian_score_dirichlet_uniform(df, simple_graph(df)))