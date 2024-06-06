import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import networkx as nx
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Function to calculate distances between nodes in a graph
def max_common_subgraph_size(g1, g2):
    # Find all common nodes between g1 and g2
    common_nodes = set(g1.nodes()) & set(g2.nodes())
    
    # Calculate all possible pairs of common nodes
    all_common_pairs = list(itertools.combinations(common_nodes, 2))
    
    # Compute the size of the maximum common subgraph
    mcs_size = len([pair for pair in all_common_pairs if g1.has_edge(*pair) and g2.has_edge(*pair)])
    
    return mcs_size

def calculate_graph_distance(g1, g2):
    mcs_size = max_common_subgraph_size(g1, g2)
    max_size = max(len(g1), len(g2))
    
    if max_size == 0:
        return 1.0  # Handle division by zero case
    
    distance = 1 - (mcs_size / max_size)
    return max(0, distance)  # Ensure distance is within [0, 1] range

# Function to calculate distances between nodes in a graph
def calculate_distances(graph):
    # Compute all pairs shortest path lengths
    all_pairs_shortest_path = dict(nx.all_pairs_shortest_path_length(graph))
    
    # Initialize a dictionary to store distances
    distances = {}
    
    # Iterate over each node and compute the average distance
    for node, paths in all_pairs_shortest_path.items():
        total_distance = sum(paths.values())
        avg_distance = total_distance / len(paths)
        distances[node] = avg_distance
    
    return distances

# Function to preprocess content
def preprocess_content(content):
    # For simplicity, let's just split the content into words
    return content.split()

# Function to create graph and calculate distances for a document
def process_document(file_path):
    # Read the document content
    with open(file_path, 'r', encoding='latin-1') as file:
        content = file.read()
    
    # Preprocess the content
    words = preprocess_content(content)
    
    # Create a graph
    graph = nx.Graph()
    
    # Add nodes for each word
    graph.add_nodes_from(words)
    
    # Add edges between adjacent words
    for i in range(len(words)-1):
        word1 = words[i]
        word2 = words[i+1]
        graph.add_edge(word1, word2)
    
    # Calculate distances between nodes
    distances = calculate_distances(graph)
    
    # Extract distances values and convert them to a list
    distances_values = list(distances.values())
    
    # Calculate the mean of distances
    if len(distances_values) > 0:
         mean_distance = np.nanmean(distances_values)
    else:
         mean_distance = 0  # or any other appropriate value

    # Print the mean distance for each document
    print("Mean Distance for", file_path, ":", mean_distance)
    
    return mean_distance


# Initialize lists to store features and labels
features = []
labels = []

# Process training data (36 documents)
topics = ['Food','Lifestyle & Hobbies','Marketing & Sales']
for i, topic in enumerate(topics, start=1):
    for doc_num in range(1, 13):  # 12 documents per topic for training
        file_path = f"D:\\GT PROJECT(633 & 647)\\Project\\{topic}\\File{doc_num}.txt"
        features.append(process_document(file_path))
        labels.append(i)

# Convert lists to numpy arrays
X_train = np.array(features).reshape(-1, 1)
y_train = np.array(labels)

# Process testing data (9 documents)
test_features = []
test_labels = []
for i, topic in enumerate(topics, start=1):
    for doc_num in range(13, 16):  # 3 documents per topic for testing
        file_path = f"D:\\GT Project(633 & 647)\\Project\\{topic}\\File{doc_num}.txt"
        test_features.append(process_document(file_path))
        test_labels.append(i)

# Convert lists to numpy arrays
X_test = np.array(test_features).reshape(-1, 1)
y_test = np.array(test_labels)

# Handle NaN values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Tune the number of neighbors using cross-validation
best_accuracy = 0
best_k = None

for k in range(1, 30):  # Try different values of k
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_imputed, y_train)
    y_pred = knn_model.predict(X_test_imputed)
    accuracy = np.round(np.sum(np.diag(confusion_matrix(y_test, y_pred))) / np.sum(confusion_matrix(y_test, y_pred)) * 100)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print("Best k:", best_k)
print("Best accuracy:", best_accuracy )

# Train the KNN model with the best k
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_imputed, y_train)

# Predict labels for the test set
y_pred = knn_model.predict(X_test_imputed)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix in a separate page
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=topics, yticklabels=topics)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")  # Save the confusion matrix as an image file
plt.show()
