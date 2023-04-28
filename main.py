from deap import gp
import gp_operators as ops
from gp_terminals.gp_image import GPImage
from gp_terminals.gp_cutshape import GPCutshape
from gp_terminals.gp_filter import GPFilter
from gp_terminals.gp_point import GPPoint
from gp_terminals.gp_size import GPSize

pset = gp.PrimitiveSetTyped("test", [float, float, GPImage, GPFilter, GPPoint, GPCutshape, GPSize], float)
pset.addPrimitive(ops.add, [float, float], float)
pset.addPrimitive(ops.sub, [float, float], float)
pset.addPrimitive(ops.conv, [GPImage, GPFilter], GPImage)
pset.addPrimitive(ops.agg_max, [GPImage, GPPoint, GPSize, GPCutshape], float)
pset.addPrimitive(ops.agg_mean, [GPImage, GPPoint, GPSize, GPCutshape], float)


expr = gp.genFull(pset, min_=2, max_=5)
tree = gp.PrimitiveTree(expr)


nodes, edges, labels = gp.graph(tree)

import graphviz
dot = 'digraph G {\n'
dot += 'node [shape=circle, style=filled, color=lightblue];\n'
for i, label in labels.items():
    dot += 'node{} [label="{}"];\n'.format(i, label)

for edge in edges:
    dot += 'node{} -> node{};\n'.format(edge[0], edge[1])

dot += '}'

graph = graphviz.Source(dot)
graph.render('tree')

#test_fn = gp.compile(tree, pset)
#print(test_fn(1,2))