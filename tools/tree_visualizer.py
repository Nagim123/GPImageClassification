import graphviz
from deap import gp
import sys

sys.setrecursionlimit(10000000)
ind = 0

def tree_parse(content: str, edges, labels):
    global ind
    b1 = content.find('(')
    if b1 == -1:
        labels[ind] = content
        ind += 1
        return ind - 1
    b2 = content.rfind(')')
    arg_content = content[b1+1:b2]
    open_brackets = 0
    c_data = ""
    childs = []
    for i in range(len(arg_content)):
        if c_data == "ARG0":
            childs.append(c_data)
            c_data = ""
            continue
        if arg_content[i] == ',' or arg_content[i] == ' ':
            continue
        if arg_content[i] == '(':
            open_brackets += 1
        elif arg_content[i] == ')':
            open_brackets -= 1
            if open_brackets == 0:
               c_data += arg_content[i]
               childs.append(c_data)
               c_data = ""
               continue
        c_data += arg_content[i]
        
    labels[ind] = content[:b1]
    c_ind = ind
    ind += 1
    if(labels[c_ind] in ["GPFilter", "GPImage", "GPPercentage", "GPPercentageSize", "ARG0"]):
        return c_ind
    if(labels[c_ind] == "GPCutshape"):
        labels[c_ind] = f"GPCutshape({arg_content})"        
        return c_ind
    for child in childs:
        edges.append((c_ind, tree_parse(child, edges, labels)))
    return c_ind

def visualize_tree(tree: str):
    global ind
    
    edges = []
    labels = {}
    ind = 0
    tree_parse(tree, edges, labels)

    dot = 'digraph G {\n'
    dot += 'node [shape=circle, style=filled, color=lightblue];\n'
    for i, label in labels.items():
        dot += 'node{} [label="{}"];\n'.format(i, label)

    for edge in edges:
        dot += 'node{} -> node{};\n'.format(edge[0], edge[1])

    dot += '}'

    graph = graphviz.Source(dot)
    graph.render('tree', 'outputs/')

visualize_tree(open("outputs/best_result_tree.txt", "r").read())