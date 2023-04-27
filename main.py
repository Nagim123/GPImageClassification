from deap import gp
import gp_operators as ops

pset = gp.PrimitiveSet("test", 6)
pset.addPrimitive(ops.add, [float, float], float)
pset.addPrimitive(ops.sub, [float, float], float)

expr = gp.genFull(pset, min_=2, max_=5)
tree = gp.PrimitiveTree(expr)

### Graphviz Section ###
import pygraphviz as pgv

nodes, edges, labels = gp.graph(expr)

g = pgv.AGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
g.layout(prog="dot")

for i in nodes:
    n = g.get_node(i)
    n.attr["label"] = labels[i]

g.draw("tree.pdf")