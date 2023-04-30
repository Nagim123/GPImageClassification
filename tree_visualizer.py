import graphviz
from deap import gp

def visualize_tree(tree: gp.PrimitiveTree):
    _, edges, labels = gp.graph(tree)

    dot = 'digraph G {\n'
    dot += 'node [shape=circle, style=filled, color=lightblue];\n'
    for i, label in labels.items():
        dot += 'node{} [label="{}"];\n'.format(i, label)

    for edge in edges:
        dot += 'node{} -> node{};\n'.format(edge[0], edge[1])

    dot += '}'

    graph = graphviz.Source(dot)
    graph.render('tree')