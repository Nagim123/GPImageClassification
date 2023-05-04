import argparse
import ast
import os
import shutil
import sys

import graphviz
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from gp_terminals.gp_image import GPImage
from gp_tools.tree_runner import run_tree


class Node:
    """
    The class represents a node of a tree (GP chromosome) in a format
    that is used to draw the tree
    """

    def __init__(self, parent: "Node"):
        """
        @param parent: The parent node of the current node
        """

        self.content = None
        self.parent = parent
        self.children = []
        self.image = False
        self.result = None
        self.dim = ""


def draw_array(data, name: str) -> None:
    """
    Draws an array-like object as a table. Saves it with the name 'name' to outputs

    @param data: array-like object to draw
    @param name: the name of the file to save the picture
    """
    fig, ax = plt.subplots()
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    table = ax.table(cellText=data, loc="center", bbox=(0, 0, 1, 1), cellLoc="center")
    table.set_fontsize(40)
    plt.savefig(f"outputs/{name}", dpi=20)


class GPTreeVisualizer:
    """
    The class is used to plot the tree and its decisions as a picture
    """

    def __init__(self, tree: str):
        """
        @param tree: the tree to draw. Represented by a set of nested parentheses
        """
        self.tree = tree

    def __call__(self, image: GPImage) -> None:
        """
        The method draws the tree and its decisions on a given image

        @param image: the given image
        """
        self.image = image

        sys.setrecursionlimit(10000000)
        if os.path.exists("outputs/tmp"):
            shutil.rmtree("outputs/tmp")
        os.mkdir("outputs/tmp")

        self.im_path = "outputs/tmp/0.png"
        plt.imsave(self.im_path, self.image.pixel_data, cmap="gray")

        self.index = 1
        self.nodes = []

        self.visualize_tree()

        shutil.rmtree("outputs/tmp")

    def visualize_tree(self) -> None:
        """
        The core of tree visualization. Prepares the string that graphviz
        will take as an input. Renders the image using graphviz
        """
        self.tree_parse(self.tree, Node(None))

        for node in self.nodes[::-1]:
            if node.content == "conv" or node.content == "pool":
                shape = node.result.pixel_data.shape
                node.dim = f"{shape[0]}x{shape[1]}"
                node.image = True
                node.content = f"./tmp/{self.index}.png"
                if shape == (1, 1) and .99 < node.result.pixel_data[0, 0] < 1.01:
                    plt.imsave(
                        f"outputs/tmp/{self.index}.png",
                        node.result.pixel_data,
                        cmap="gray",
                        vmin=0.0,
                        vmax=1.0
                    )
                else:
                    plt.imsave(
                        f"outputs/tmp/{self.index}.png",
                        np.interp(node.result.pixel_data, (node.result.pixel_data.min(), node.result.pixel_data.max()), (0, 1)),
                        cmap="gray"
                    )
                self.index += 1

            elif (
                "GP" not in node.content
                and not node.image
                and not node.content.isdigit()
                and not node.content[1:].isdigit()
            ):
                node.content += "\n" + str(node.result)

        dot = "digraph G {\n"

        for node in self.nodes:
            if node.image:
                dot += (
                    'node{} [label="{}" fontcolor=red image="{}" shape=rectangle width=1 height=1 '
                    "imagescale=true];\n".format(id(node), node.dim, node.content)
                )
            else:
                dot += 'node{} [label="{}" shape=circle style=filled color=lightblue width=1];\n'.format(
                    id(node), node.content
                )

            if node.parent:
                dot += "node{} -> node{};\n".format(id(node.parent), id(node))

        dot += "}"

        graph = graphviz.Source(dot)
        graph.render("tree", "outputs/")

    def tree_parse(self, content: str, node: "Node"):
        """
        Recursively parses the tree by nodes

        @param content: the content of the node and its subtree
        @param node: the current node
        """
        left = content.find("(")
        right = content.rfind(")")
        self.nodes.append(node)

        # Get the result of the node's subtree
        node.result = run_tree(self.image, content)

        # Split the node's content to arguments and action
        node.content = content[:left] if left != -1 else content
        content = content[left + 1 : right] if left != -1 else content

        # If the node is a number
        if content[1:].isdigit() or content.isdigit():
            node.content = content

        # If the node is a filter
        elif node.content == "GPFilter":
            table = ast.literal_eval(content.replace("np.array(", "").replace(")", ""))
            name = f"./tmp/{self.index}.png"
            draw_array(table, name)
            node.content = name
            node.image = True
            self.index += 1

        # If the node is a GPPercentage or GPPercentageSize object
        elif "GPPercent" in node.content:
            x, y = ast.literal_eval(content)
            x, y = round(x, 3), round(y, 3)
            node.content = f"{node.content}({x}, {y})"
            length = len(node.content)
            node.content = (
                node.content[: length // 2] + "\n" + node.content[length // 2 :]
            )

        # If the node is another object starting with GP
        elif "GP" in node.content:
            node.content = f"{node.content}({content})"

        # If the node is the input image
        elif node.content == "ARG0":
            node.content = self.im_path
            node.image = True

        # If the node is not terminal
        else:
            parentheses, start = 0, 0
            children = []

            # Parse arguments
            for i in range(len(content)):
                if not parentheses and content[i] == ",":
                    children.append(content[start:i])
                    start = i + 2

                elif content[i] == "(":
                    parentheses += 1

                elif content[i] == ")":
                    parentheses -= 1

            children.append(content[start:])

            # Run recursively for all arguments
            for child in children:
                child_node = Node(node)
                node.children.append(child_node)
                self.tree_parse(child, child_node)


tree_str = open("outputs/best_result_tree.txt", "r").readline()
drawer = GPTreeVisualizer(tree_str)

parser = argparse.ArgumentParser(description="Visualize the tree on the example image")

parser.add_argument("path", type=str, help="Path to file for visualizing")

args = parser.parse_args()
path_to_file = args.path
image = GPImage(np.array(Image.open(path_to_file).convert('L')))

drawer(image)
