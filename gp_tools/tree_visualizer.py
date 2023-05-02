import shutil
import matplotlib.pyplot as plt
import graphviz
from gp_terminals.gp_image import GPImage
import sys
from tools.tree_runner import run_tree
import os
import ast

sys.setrecursionlimit(10000000)
index = 0

if os.path.exists('../outputs/tmp'):
    shutil.rmtree('../outputs/tmp')
os.mkdir('../outputs/tmp')


class Node:
    def __init__(self, parent):
        self.content = None
        self.parent = parent
        self.children = []
        self.image = False
        self.result = None
        self.dim = ""


def draw_array(data, name):
    fig, ax = plt.subplots()
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    table = ax.table(cellText=data, loc='center', bbox=(0, 0, 1, 1), cellLoc='center')
    table.set_fontsize(40)
    plt.savefig(f'../outputs/{name}', dpi=20)


def tree_parse(content: str, node, nodes, inp):
    left = content.find('(')
    right = content.rfind(')')
    nodes.append(node)
    node.result = run_tree(inp, content)
    node.content = content[:left] if left != -1 else content
    content = content[left + 1: right] if left != -1 else content
    if content[1:].isdigit():
        node.content = content
        return
    if node.content == 'GPFilter':
        table = ast.literal_eval(content.replace('np.array(', '').replace(')', ''))
        global index
        name = f'./tmp/{index}.png'
        draw_array(table, name)
        node.content = name
        node.image = True
        index += 1
    elif 'GPPercent' in node.content:
        x, y = ast.literal_eval(content)
        x, y = round(x, 3), round(y, 3)
        node.content = f'{node.content}({x}, {y})'
        length = len(node.content)
        node.content = node.content[:length//2] + '\n' + node.content[length//2:]
    elif 'GP' in node.content:
        node.content = f'{node.content}({content})'
    elif node.content == 'ARG0':
        node.content = '../outputs/vis.png'
        node.image = True
    else:
        parentheses, start = 0, 0
        children = []
        for i in range(len(content)):
            if not parentheses and content[i] == ',':
                children.append(content[start: i])
                start = i + 2
            elif content[i] == '(':
                parentheses += 1
            elif content[i] == ')':
                parentheses -= 1
        children.append(content[start:])
        for child in children:
            child_node = Node(node)
            node.children.append(child_node)
            tree_parse(child, child_node, nodes, inp)


def visualize_tree(tree: str):
    nodes = []
    tree_parse(tree, Node(None), nodes, GPImage(plt.imread('../outputs/vis.png')))

    for node in nodes[::-1]:
        if node.content == 'conv' or node.content == 'pool':
            shape = node.result.pixel_data.shape
            node.dim = f'{shape[0]}x{shape[1]}'
            node.image = True
            global index
            index += 1
            node.content = f'./tmp/{index}.png'
            plt.imsave(f'../outputs/tmp/{index}.png', node.result.pixel_data, cmap='gray')
        elif 'GP' not in node.content and not node.image and not node.content[1:].isdigit():
            node.content += '\n' + str(node.result)




    dot = 'digraph G {\n'
    for node in nodes:
        if node.image:
            dot += 'node{} [label="{}" fontcolor=blue image="{}" shape=rectangle width=1 height=1 imagescale=true];\n'\
                .format(id(node), node.dim, node.content)
        else:
            dot += 'node{} [label="{}" shape=circle style=filled color=lightblue width=1];\n'\
                .format(id(node), node.content)

        if node.parent:
            dot += 'node{} -> node{};\n'.format(id(node.parent), id(node))
    dot += '}'

    graph = graphviz.Source(dot)
    graph.render('tree', '../outputs/')


tree_str = open("../outputs/best_result_tree.txt", "r").readline()
visualize_tree(tree_str)
