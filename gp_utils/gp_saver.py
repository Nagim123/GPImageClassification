from gp_structures.gp_tree import GPTree

def save_gp_tree(gptree : GPTree, name: str):
    output_file = open(f"outputs/classifiers/best_result_tree_{name}.txt", 'w')
    output_file.write(str(gptree.tree))
    output_file.close()