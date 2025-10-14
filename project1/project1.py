import sys
import matplotlib.pyplot as plt
import networkx as nx
from bn_scoring import load_discrete_data
from structure_learning import hill_climb
import time


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def compute(infile, outfile):
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
    df = load_discrete_data(infile) #load dataset
    columns = list(df.columns)

    idx2names = {i: col for i, col in enumerate(columns)} #maps integer node IDs to variable names (i.e. 0 --> "age")
    names2idx = {v: k for k, v in idx2names.items()} #maps variable names to integer node IDs ("age" --> 0)

    start_time = time.time()

    dag, best_score = hill_climb(df, max_iters=100, max_parents=3) #run structure learning algo and learn structure of DAG
    end_time = time.time()
    runtime = end_time - start_time

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(dag, seed=0)
    nx.draw_networkx_nodes(dag, pos, node_size=900)
    nx.draw_networkx_labels(dag, pos, font_size=9)
    nx.draw_networkx_edges(dag, pos, arrows=True, arrowstyle='-|>', arrowsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outfile.replace('.gph', '.png'), dpi=200)
    # plt.show()  #interactive window

    dag_indexed = nx.relabel_nodes(dag, names2idx, copy=True) #convert names to node IDs
    write_gph(dag_indexed, idx2names, outfile) #convert node IDs back to names

    print(f"Structure algorithm finished running. Best score = {best_score:.2f}")
    print(f"Runtime = {runtime:.2f} seconds")
    print(f"Graph written to {outfile}.")
    print(f"Edges: {list(dag.edges())}")

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
