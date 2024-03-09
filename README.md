# Decision Tree Classifier
This Python script implements a decision tree classifier using a custom algorithm. The decision tree is constructed based on information gain, and it provides methods for training the model, making predictions, evaluating accuracy, and visualizing the decision tree graph.

## Dependencies
* numpy (for numerical computations)
* pandas (for data manipulation)
* matplotlib (for data visualization)
* networkx (for graph creation)
* tqdm (for progress bar)
Ensure you have installed these dependencies before running the script.

## Usage
To use the decision tree classifier, follow these steps:

1. Prepare Data: Prepare your dataset in the required format. The dataset should include:

- Document-word associations
- Document classes (labels)
- Word-to-word ID mappings
- Data Parsing: Use the parse_data function to parse your dataset files.

2. Instantiate Model: Create an instance of the TreeModel class.
3. Training: Fit the model using the fit method by providing the document-word associations, document classes, maximum number of splits, and whether to use weighted or non-weighted information gain.
4. Prediction: Use the predict method to predict the class for a list of words in a document.
5. Accuracy Evaluation: Evaluate the accuracy of the model using the eval_accuracy method.
6. Visualization: Visualize the decision tree graph using the plot_graph method.

## Example
```python
Copy code
# Parse data
train_doc_words, train_doc_classes, id_words = parse_data("./dataset/trainData.txt",
                                               "./dataset/trainLabel.txt",
                                               "./dataset/words.txt")

# Instantiate and train model
tree_model = TreeModel()
tree_model.fit(train_doc_words, train_doc_classes, max_splits=100, weighted=True)

# Visualize decision tree graph
tree_model.plot_graph(id_words, max_splits=10)

# Evaluate accuracy
train_accuracy = tree_model.eval_accuracy(train_doc_words, train_doc_classes, max_splits=100)
print(f"Training Accuracy: {100*train_accuracy:.2f}%")

# Make predictions
test_doc_words, test_doc_classes, _ = parse_data("./dataset/testData.txt",
                                               "./dataset/testLabel.txt",
                                               "./dataset/words.txt")
test_accuracy = tree_model.eval_accuracy(test_doc_words, test_doc_classes, max_splits=100)
print(f"Testing Accuracy: {100*test_accuracy:.2f}%")
```
## Files
TreeClassifier.py: Contains the decision tree classifier implementation.
dataset/: Directory containing dataset files (train and test data, labels, word mappings).
## Note
Ensure that your dataset files are correctly formatted and accessible to the script.
Adjust parameters such as max_splits according to your dataset and requirements.
Experiment with both weighted and non-weighted methods for information gain to observe different model behaviors.
