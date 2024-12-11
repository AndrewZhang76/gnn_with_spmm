import numpy as np

class CoraDataset:
    """
    A simple class for the Cora dataset.
    Processes the data from .content and .cites files.
    Returns:
        - X: Feature matrix (np.array of shape [num_nodes, num_features])
        - y: Labels (np.array of shape [num_nodes])
        - adjacency_matrix: Dense adjacency matrix (np.array of shape [num_nodes, num_nodes])
    """

    def __init__(self, content_file: str, cites_file: str):
        """
        Initializes the dataset with provided file paths.

        Args:
            content_file (str): Path to the .content file.
            cites_file (str): Path to the .cites file.
        """
        self.content_file = content_file
        self.cites_file = cites_file

        # These will store the processed data
        self.X = None
        self.y = None
        self.adjacency_matrix = None
        self.label_mapping = None

        # Load and process the data
        self._load_data()

    def _load_data(self):
        """Loads and processes node features, labels, and edges."""
        nodes = []
        labels = []
        features = []

        # Read the .content file
        with open(self.content_file, 'r') as f:
            for line in f:
                elements = line.strip().split('\t')
                nodes.append(elements[0])
                features.append([float(x) for x in elements[1:-1]])
                labels.append(elements[-1])

        # Encode labels into integers
        self.label_mapping = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        encoded_labels = [self.label_mapping[label] for label in labels]

        # Convert features and labels to NumPy arrays
        self.X = np.array(features, dtype=np.float32)
        self.y = np.array(encoded_labels, dtype=np.int64)

        # Build the adjacency matrix
        num_nodes = len(nodes)
        node_index = {node: idx for idx, node in enumerate(nodes)}
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        with open(self.cites_file, 'r') as f:
            for line in f:
                src, dst = line.strip().split('\t')
                if src in node_index and dst in node_index:
                    i, j = node_index[src], node_index[dst]
                    self.adjacency_matrix[i, j] = 1
                    self.adjacency_matrix[j, i] = 1  # Assume undirected graph

    def get_data(self):
        """Returns the full dataset: features, labels, and adjacency matrix."""
        return self.X, self.y, self.adjacency_matrix

    def get_label_mapping(self):
        """Returns the mapping from original labels to encoded integers."""
        return self.label_mapping