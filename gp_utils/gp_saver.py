from gp_structures.gp_tree import GPTree

def save_gp_tree(gptree : GPTree):
    output_file = open("outputs/best_result_tree.txt", 'w')
    output_file.write(str(gptree.tree))
    output_file.close()