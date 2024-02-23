import math
import pandas as pd

class TreeNode():
    def __init__(self, node_data):
        self.node_data = node_data
        self.node_info = eval_info(node_data)

    def divide(self, word: str):
        """
        Divides the data of a node based on the "word" feature

        :param word: The word feature used for dividing data
        :type word: String
        """
        docs_with_word = self.node_data.loc[self.node_data[word]==1, :]
        docs_without_word = self.node_data.loc[self.node_data[word]==0, :]
        return (TreeNode(docs_with_word), TreeNode(docs_without_word))
    
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

    def eval_info_gain(self, word: str, is_weighted: bool):
        """
        Evaluates information gain by classifying node data using "word" feature

        :param word: The word feature used for classification
        :type word: String

        :param is_weighted: Specifies whether to use avarage information gain or weighted information gain
        :type is_weighted: Boolean
        """
        node_info = self.get_node_info()
        node1, node2 = self.divide(word)
        n1 = len(node1.get_node_data())
        n2 = len(node2.get_node_data())
        n = n1 + n2
        node1_info = node1.get_node_info()
        node2_info = node2.get_node_info() 
        if is_weighted:
            info_gain = node_info - (n1/n*node1_info + n2/n*node2_info)
        else:
            info_gain = node_info - (node1_info + node2_info)/2
        return info_gain
    
    def get_word_list(self):
        """
        Returns list of available words in the node
        """
        node_data = self.get_node_data()
        node_data = node_data.iloc[:, 1:-1]
        node_data.loc['Total'] = node_data.sum()
        return list(node_data.columns[node_data.loc['Total']!=0])

    
    def find_best_word(self, is_weighted:bool):
        """
        Returns the best word which results in the highest information gain be dividing the node data using that word

        :param is_weighted: Specifies whetehr weighted information gain formula is used
        :type is_weighted: Boolean 
        """
        words = self.get_word_list()
        best_gain = 0
        best_word = ""
        for word in words:
            info_gain = self.eval_info_gain(word, is_weighted)
            if info_gain > best_gain:
                best_word = word
                best_gain = info_gain
        return (best_word, best_gain)
        
def dt_trainer(doc_words):
    rem_info = 1

def data_to_wide(doc_words, doc_labs):
    doc_words['Availability'] = 1
    wide_data = pd.pivot_table(doc_words, index="docID", columns="wordID", values="Availability").fillna(0)
    wide_data = wide_data.join(doc_labs, on="docID", how="right").fillna(0)
    wide_data.reset_index(drop=True, inplace=True)
    return wide_data

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
    wide_data = data_to_wide(doc_words, doc_labs)
    sample_data = wide_data.sample(n=100)
    main_node = TreeNode(sample_data)
    bsetword, bestgain = main_node.find_best_word(True)
    print(bsetword, bestgain)

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
