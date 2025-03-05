import networkx as nx
import matplotlib.pyplot as plt

def is_eulerian(G):
    """Check if a graph is Eulerian (all degrees even and connected)."""
    return nx.is_connected(G) and all(d % 2 == 0 for _, d in G.degree())

def main():
    # Step 1: Create two disconnected Eulerian components
    G = nx.Graph()

    # First Eulerian component (Cycle on 4 vertices)
    G1_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    G.add_edges_from(G1_edges)

    # Second Eulerian component (Cycle on 6 vertices)
    G2_edges = [(4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 4)]
    G.add_edges_from(G2_edges)

    # Check initial properties
    print("Initial Graph Properties:")
    print("- Connected:", nx.is_connected(G))
    print("- Is Eulerian:", is_eulerian(G))
    
    # Verify degrees before adding edges
    print("\nInitial Vertex Degrees:")
    for node, deg in G.degree():
        print(f" Vertex {node}: degree {deg}")

    # Step 2: Add 4 edges to make the graph Eulerian
    additional_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]  # 4 edges between components
    G.add_edges_from(additional_edges)

    # Check final properties
    print("\nAfter Adding 4 Edges:")
    print("- Connected:", nx.is_connected(G))
    print("- Is Eulerian:", is_eulerian(G))

    # Verify degrees after adding edges
    print("\nFinal Vertex Degrees:")
    for node, deg in G.degree():
        print(f" Vertex {node}: degree {deg}")

    # Step 3: Draw Graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=800)
    plt.title("Eulerian Graph After Adding 4 Edges")
    
    # Save and show the plot
    filename = "eulerian_connected_graph.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"\nGraph image saved as '{filename}'.")

if __name__ == "__main__":
    main()
