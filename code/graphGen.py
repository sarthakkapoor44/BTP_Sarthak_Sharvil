from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def generate_random_connected_adjacency(num_servers: int, 
                                       edge_probability: float = 0.3,
                                       min_degree: int = 1,
                                       seed: Optional[int] = None) -> Dict[int, List[int]]:
    """
    Generate a random connected graph adjacency list.
    
    Uses Erdős-Rényi model with connectivity guarantee via MST.
    
    Parameters
    ----------
    num_servers : int
        Number of servers (nodes) in the graph
    edge_probability : float, optional
        Probability of edge creation between any two nodes (default: 0.3)
        Higher values create denser graphs
    min_degree : int, optional
        Minimum degree for each node (default: 1)
        Ensures no isolated nodes
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    Dict[int, List[int]]
        Adjacency list where keys are server IDs and values are lists of neighbors
    
    Examples
    --------
    >>> adj = generate_random_connected_adjacency(5, edge_probability=0.4)
    >>> print(adj)
    {0: [1, 3], 1: [0, 2, 4], 2: [1, 3], 3: [0, 2, 4], 4: [1, 3]}
    
    >>> # For a sparser graph
    >>> adj = generate_random_connected_adjacency(10, edge_probability=0.2, seed=42)
    """
    if num_servers < 2:
        raise ValueError("Number of servers must be at least 2")
    
    if not 0 <= edge_probability <= 1:
        raise ValueError("edge_probability must be between 0 and 1")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Step 1: Create a Minimum Spanning Tree to ensure connectivity
    # Generate random edge weights for all possible edges
    edges = []
    for i in range(num_servers):
        for j in range(i + 1, num_servers):
            weight = np.random.random()
            edges.append((weight, i, j))
    
    # Sort edges by weight (Kruskal's algorithm)
    edges.sort()
    
    # Use Union-Find to build MST
    parent = list(range(num_servers))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    # Build adjacency from MST edges
    adjacency = {i: [] for i in range(num_servers)}
    mst_edges = []
    
    for weight, u, v in edges:
        if union(u, v):
            adjacency[u].append(v)
            adjacency[v].append(u)
            mst_edges.append((u, v))
            if len(mst_edges) == num_servers - 1:
                break
    
    # Step 2: Add random edges based on edge_probability
    for i in range(num_servers):
        for j in range(i + 1, num_servers):
            # Skip if edge already exists (from MST)
            if j in adjacency[i]:
                continue
            
            # Add edge with given probability
            if np.random.random() < edge_probability:
                adjacency[i].append(j)
                adjacency[j].append(i)
    
    # Step 3: Ensure minimum degree requirement
    for node in range(num_servers):
        current_degree = len(adjacency[node])
        if current_degree < min_degree:
            # Find potential neighbors not already connected
            potential_neighbors = [n for n in range(num_servers) 
                                 if n != node and n not in adjacency[node]]
            
            # Randomly select additional neighbors
            num_to_add = min(min_degree - current_degree, len(potential_neighbors))
            new_neighbors = np.random.choice(potential_neighbors, num_to_add, replace=False)
            
            for neighbor in new_neighbors:
                adjacency[node].append(neighbor)
                adjacency[neighbor].append(node)
    
    # Step 4: Sort neighbor lists for consistency
    for node in adjacency:
        adjacency[node] = sorted(adjacency[node])
    
    return adjacency


def visualize_adjacency(adjacency: Dict[int, List[int]], 
                       title: str = "Network Topology",
                       figsize: Tuple[int, int] = (10, 8),
                       node_size: int = 800,
                       save_path: Optional[str] = None):
    G = nx.Graph()
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Use spring layout for nice positioning
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    
    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=node_size, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)
    
    # Add statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
    diameter = nx.diameter(G) if nx.is_connected(G) else float('inf')
    
    info_text = f"Nodes: {num_nodes} | Edges: {num_edges} | Avg Degree: {avg_degree:.2f} | Diameter: {diameter}"
    plt.text(0.5, 0.02, info_text, ha='center', transform=plt.gcf().transFigure,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    plt.show()


# # Example usage
# print("✓ Random connected graph generator defined")
# print("\nUsage:")
# print("  # Generate random connected graph")
# print("  adj = generate_random_connected_adjacency(num_servers=10, edge_probability=0.3)")
# print("  ")
# print("  # Visualize it")
# print("  visualize_adjacency(adj, title='My Network')")
# print("  ")
# print("  # Use in config")
# print("  config.override(adjacency=adj)")