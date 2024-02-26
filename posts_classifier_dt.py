import math
import copy
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

class TreeNode():
    def __init__(self, node_data, is_weighted):
        self.split_number = None
        self.has_best_word = None
        self.is_weighted = is_weighted
        self.node_data = node_data
        self.node_info = eval_info(node_data)
        self.point_est = eval_point_est(node_data)
        self.children = []

    def get_node_info(self):
        """
        Returns the amount of information within the node
        """
        return self.node_info
    
    def get_node_data(self):
        """
        Returns the data of the node
        """
        return self.node_data
    
    def get_split_number(self):
        return self.split_number
    
    def set_split_number(self, i_node):
        self.split_number = i_node

    def set_has_best_word(self, has_word:bool):
        self.has_best_word = has_word

    def get_has_best_word(self):
        return self.has_best_word
    
    def add_child(self, node):
        self.children.append(node)

    def set_child_with_word(self, node):
        self.child_with_word = node

    def set_child_without_word(self, node):
        self.child_without_word = node

    def get_point_est(self):
        return self.point_est

    def split_node(self, word: str, is_weighted: bool):
        """
        splits the data of a node based on the "word" feature and returns two new nodes

        :param word: The word feature used for dividing data
        :type word: String
        """
        node_data = self.get_node_data()
        docs_with_word = node_data.loc[node_data[word]==True, :]
        docs_without_word = node_data.loc[node_data[word]==False, :]
        if len(docs_with_word):
            node1 = TreeNode(docs_with_word, is_weighted)
            node1.set_has_best_word(True)
        else:
            node1 = None
        if len(docs_without_word):
            node2 = TreeNode(docs_without_word, is_weighted)
            node2.set_has_best_word(False)
        else:
            node2 = None
        return (node1, node2)

    def set_best_word(self, is_weighted:bool):
        """
        Returns the best word which results in the highest information gain be dividing the node data using that word

        :param is_weighted: Specifies whetehr weighted information gain formula is used
        :type is_weighted: Boolean 
        """
        words = self.list_words()
        best_gain = 0
        best_word = None
        if words:
            for word in words:
                info_gain = self.eval_info_gain(word, is_weighted)
                if info_gain > best_gain:
                    best_word = word
                    best_gain = info_gain
        self.best_word = best_word
        self.best_gain = best_gain
    
    def eval_info_gain(self, word: str, is_weighted: bool):
        """
        Evaluates information gain by classifying node data using "word" feature

        :param word: The word feature used for classification
        :type word: String

        :param is_weighted: Specifies whether to use avarage information gain or weighted information gain
        :type is_weighted: Boolean
        """
        node_info = self.get_node_info()
        node_copy = copy.deepcopy(self)
        node1, node2 = node_copy.split_node(word, self.is_weighted)
        if node1 and node2:
            n1 = len(node1.get_node_data())
            n2 = len(node2.get_node_data())
            n = n1 + n2
            node1_info = node1.get_node_info()
            node2_info = node2.get_node_info() 
            if is_weighted:
                info_gain = node_info - (n1/n*node1_info + n2/n*node2_info)
            else:
                info_gain = node_info - (node1_info + node2_info)/2
        else:
            info_gain = 0
        return info_gain

    def get_best_word(self):
        return self.best_word
    
    def get_best_gain(self):
        return self.best_gain
    
    def list_words(self):
        """
        Returns list of available words in the node
        """
        node_data = self.get_node_data()
        node_data = node_data.iloc[:, 1:-1]
        node_data.loc['Total', :] = node_data.sum()
        self.words = set(node_data.columns[node_data.loc['Total']!=0])
        return self.words

class TreeLearner():
    def __init__(self, is_weighted: bool):
        self.is_weighted = is_weighted
        self.priority_leaves = []
        self.root_node = None
    
    def train(self, train_data: pd.DataFrame):
        """
        Starts training of the tree using the priority list of leaves
        """
        n_iter = 0
        self.root_node = TreeNode(train_data, self.is_weighted)
        self.root_node.set_best_word(self.is_weighted)
        self.priority_leaves.append(self.root_node)
        while n_iter < 100 or not self.priority_leaves:
            n_iter += 1
            # From the priority list choose the node of the highest priority
            self.priority_leaves.sort(key=lambda x: x.best_gain, reverse=True)
            best_node = self.priority_leaves[0]
            if best_node.best_gain > 0:
                best_node.set_split_number(n_iter)
                node1, node2 = best_node.split_node(best_node.best_word, self.is_weighted)
                node1.set_best_word(self.is_weighted)
                node2.set_best_word(self.is_weighted)
                best_node.add_child(node1)
                best_node.add_child(node2)
                self.priority_leaves += [node1, node2]
                self.priority_leaves.remove(best_node)
            else:
                break

    def get_root_node(self):
        return self.root_node

    def plot(self):
    #     fig = plt.figure(figsize=(8, 6))
    #     plot_node(fig, self.root_node, 0.5, 0.9)
    #     plt.axis("off")
    #     plt.show()

    # Create a directed graph
        G = nx.DiGraph()
        create_graph(G, self.root_node)
        # Draw the tree
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        nx.draw(G, pos, with_labels=False, node_size=1000, font_size=10, font_color="black")
        nx.draw_networkx_labels(G, pos, labels={n: f"split:{G.nodes[n]['num']}, est:{G.nodes[n]['mode']}, p:{G.nodes[n]['p']:.2f}" for n in G.nodes()})
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G.edges[u, v]["have"] for u, v in G.edges()})

        plt.axis("off")
        plt.show()



def data_to_wide(doc_words, doc_labs):
    doc_words['Availability'] = 1
    wide_data = (
        pd.pivot_table(doc_words, index="docID", columns="wordID", values="Availability").fillna(0)
        .astype(bool)
        .join(doc_labs, on="docID", how="right").fillna(0)
        .reset_index(drop=True)
    )
    return wide_data

def eval_point_est(df: pd.DataFrame):
    # find the mode and if there are multiple modes return the first one in the list
    mode = list(df["Class"].mode(dropna=True))[0]
    
    # evaluate the probablity of the mode
    p = df["Class"].value_counts()[mode]/len(df)
    return (mode, p)

def eval_info(labeled_data):
    n_data = len(labeled_data)
    count_1 = len(labeled_data.loc[labeled_data["Class"] == 1, :])
    count_2 = n_data - count_1
    p1 = count_1/n_data
    p2 = 1 - p1
    if p1*p2 == 0:
        info = 1
    else:
        info = - (p1*math.log2(p1) + p2*math.log2(p2))
    return info

# def print_tree(node, indent="  "):
#     have_str = "have" if node.get_has_best_word() else "not have"
#     print(indent + f"node {node.get_split_number()}, {have_str} {node.get_best_word()}, class est:{node.get_point_est()}")
#     for child in node.children:
#         print_tree(child, indent)

# def plot_node(fig, node, x, y):
#     """
#     Plots the decision tree
#     """
#     have_str = "have" if node.get_has_best_word() else "not have"
#     v_spacing = 0.1
#     h_spacing = 0.1
#     fig.text(x,
#               y,
#               f"node {node.get_split_number()}, {have_str} {node.get_best_word()}, class est:{node.get_point_est()}",
#               ha="center",
#               va="center")
#     for i, child in enumerate(node.children):
#         plot_node(fig, child, x + (i - 0.5) * h_spacing, y - v_spacing)

def create_graph(G, node):
    global words_list
    G.add_node(id(node), num=node.get_split_number(), mode=node.get_point_est()[0], p=node.get_point_est()[1])
    for child in node.children:
        have_str = "have" if child.get_has_best_word() else "not have"
        G.add_edge(id(node), id(child), have=f"{have_str} {words_list[node.get_best_word()]}")
        create_graph(G, child)

# def plot_node(node, depth, pos):
#     """
#     Plots the decision tree
#     """
#     # Define spacing between nodes
#     horizontal_spacing = 2
#     vertical_spacing = 2

#     # Calculate x position based on depth
#     x = pos * horizontal_spacing
#     # Calculate y position based on depth
#     y = -depth * vertical_spacing

#     plt.text(x, y, str(id(node)), ha="center", va="center")

#     # Plot children recursively
#     if node.children:
#         child_count = len(node.children)
#         # Spread children out evenly
#         start_pos = x - ((child_count - 1) * horizontal_spacing) / 2
#         for i, child in enumerate(node.children):
#             plot_node(child, depth + 1, start_pos + i * horizontal_spacing)
        
def read_words(path):
    all_words = []
    with open(path, mode="r") as file:
        for line in file:
            all_words.append(line.strip())
    return all_words
        
            

def main():
    # read training data
    doc_words = pd.read_csv("./dataset/trainData.txt",
                              sep=" ",
                              names=['docID', 'wordID'],
                              dtype={'docID':int, 'wordID':int})
    doc_labs = pd.read_csv("./dataset/trainLabel.txt",
                             sep=" ",
                             names=['Class'],
                             dtype={'Class':int})
    doc_labs.index.set_names("docID", inplace=True)
    global words_list
    words_list = read_words("./dataset/words.txt")
    wide_data = data_to_wide(doc_words, doc_labs)
    sample_data = wide_data.sample(n=100)
    tree_learner = TreeLearner(is_weighted=True)
    tree_learner.train(sample_data)
    # root_node = tree_learner.get_root_node()
    # print_tree(root_node)
    tree_learner.plot()

    # read test data
    # doc_word_df = pd.read_csv("./dataset/testData.txt",
    #                           sep=" ",
    #                           names=['docID', 'wordID'],
    #                           dtype={'docID':int, 'wordID':int})
    # doc_lab_df = pd.read_csv("./dataset/testLabel.txt",
    #                          sep=" ",
    #                          names=['Class'],
    #                          dtype={'Class':int})
    # doc_lab_df.index.set_names("docID", inplace=True)

if __name__ == "__main__":
    main()
