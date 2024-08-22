import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

class GraphConverter:
    def __init__(self, file_path, delimiter=";", directed=False):
        """
        Initializes the GraphConverter.

        Args:
            file_path (str): Path to the CSV file containing graph data.
            delimiter (str, optional): Delimiter used in the CSV file (default is ';').
            directed (bool, optional): Whether to create a directed or undirected graph (default is False, undirected).
        """
        self.file_path = file_path
        self.delimiter = delimiter
        self.directed = directed
        self.graph_type = nx.DiGraph if directed else nx.Graph

    def create_graph(self, remove_leaves=True):
        """
        Creates and returns a NetworkX graph from the CSV data.

        Args:
            remove_leaves (bool, optional): Whether to remove leaf nodes (default is False).
        
        Returns:
            nx.Graph or nx.DiGraph: The created graph.
        """

        df = pd.read_csv(self.file_path, delimiter=self.delimiter)
        G = self.graph_type()

        # Check for required columns
        if not set(["edge_from", "edge_to"]).issubset(df.columns):
            raise ValueError("CSV file must contain 'edge_from' and 'edge_to' columns.")

        for _, row in df.iterrows():
            G.add_edge(row["edge_from"], row["edge_to"])

        if remove_leaves:
            leaf_nodes = [node for node in G.nodes() if G.degree(node) == 1]
            G.remove_nodes_from(leaf_nodes)

        return G
    
    def get_edge_matrix(self, graph):
        node_to_index = {node: i for i, node in enumerate(graph.nodes())}
        edge_index_np = np.array([(node_to_index[u], node_to_index[v]) for u, v in graph.edges()]).T
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        return edge_index
    
    def visualize_graph(self, graph, layout=None, figsize=(8, 6)):
        """
        Visualizes the given NetworkX graph using matplotlib.

        Args:
            graph (nx.Graph or nx.DiGraph): The graph to visualize.
            layout (function, optional): A NetworkX layout function (e.g., nx.spring_layout). If None, a default layout will be used.
            figsize (tuple, optional): Size of the figure (default is (8, 6)).
        """
        if layout is None:
            # Automatically choose a suitable layout based on the graph type
            if self.directed:
                layout = nx.kamada_kawai_layout(graph)
            else:
                layout = nx.spring_layout(graph)
        
        plt.figure(figsize=figsize)
        nx.draw(
            graph,
            pos=layout,
            with_labels=True,
            node_color="lightblue",
            node_size=500,
            font_size=8,
        )
        plt.title("Graph Visualization")
        plt.show()