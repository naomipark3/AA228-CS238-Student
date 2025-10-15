# viz_gph_to_dot.py
import sys

def gph_to_dot(gph_path, dot_path):
    with open(gph_path) as f, open(dot_path, "w") as out:
        out.write('digraph G {\n')
        out.write('  rankdir=LR;\n')                      # left→right layering
        out.write('  graph [splines=ortho, nodesep=0.4, ranksep=0.6];\n')
        out.write('  node  [shape=box, style=rounded, fontname="Helvetica"];\n')
        out.write('  edge  [arrowsize=0.8];\n')
        for line in f:
            if not line.strip(): 
                continue
            u, v = [s.strip() for s in line.split(",")]
            out.write(f'  "{u}" -> "{v}";\n')
        out.write('}\n')

if __name__ == "__main__":
    gph_to_dot(sys.argv[1], sys.argv[2])