from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

class UnionFind:
    """Union-Find data structure with path compression and union by rank."""
    
    def __init__(self, n: int):
        """
        Initialize Union-Find with n elements.
        
        Args:
            n: Number of elements
        """
        self.parent = list(range(n))  # Parent array
        self.rank = [0] * n          # Rank array for union by rank
        self.size = [1] * n          # Size of each component
        self.count = n               # Number of components
    
    def find(self, x: int) -> int:
        """
        Find the root of element x with path compression.
        
        Time Complexity: O(α(n)) amortized, where α is the inverse Ackermann function
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        Union the sets containing x and y.
        Returns True if x and y were in different sets, False otherwise.
        
        Time Complexity: O(α(n)) amortized
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        self.count -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """
        Check if x and y are in the same set.
        
        Time Complexity: O(α(n)) amortized
        """
        return self.find(x) == self.find(y)
    
    def get_component_size(self, x: int) -> int:
        """
        Get the size of the component containing x.
        
        Time Complexity: O(α(n)) amortized
        """
        return self.size[self.find(x)]
    
    def get_components(self) -> List[List[int]]:
        """
        Get all components.
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        components = defaultdict(list)
        for i in range(len(self.parent)):
            components[self.find(i)].append(i)
        return list(components.values())

def find_connected_components(edges: List[List[int]], n: int) -> List[List[int]]:
    """
    Find all connected components in an undirected graph.
    
    Example:
    Input: edges = [[0,1], [1,2], [3,4]], n = 5
    Output: [[0,1,2], [3,4]] (Two components)
    
    Time Complexity: O(E * α(n)) where E is number of edges
    Space Complexity: O(n)
    """
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.get_components()

def detect_cycle(edges: List[List[int]], n: int) -> bool:
    """
    Detect if an undirected graph has a cycle.
    
    Example:
    Input: edges = [[0,1], [1,2], [2,0]], n = 3
    Output: True (Cycle: 0-1-2-0)
    
    Time Complexity: O(E * α(n))
    Space Complexity: O(n)
    """
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):  # If already connected, cycle exists
            return True
    return False

def find_minimum_spanning_tree(edges: List[List[int]], n: int, 
                             weights: Optional[List[int]] = None) -> List[List[int]]:
    """
    Find minimum spanning tree using Kruskal's algorithm.
    
    Example:
    Input: edges = [[0,1], [1,2], [2,0]], weights = [1,2,3], n = 3
    Output: [[0,1], [1,2]] (MST with weight 3)
    
    Time Complexity: O(E * log E) where E is number of edges
    Space Complexity: O(n)
    """
    if weights is None:
        weights = [1] * len(edges)
    
    # Sort edges by weight
    sorted_edges = sorted(zip(edges, weights), key=lambda x: x[1])
    
    uf = UnionFind(n)
    mst = []
    
    for (u, v), _ in sorted_edges:
        if uf.union(u, v):
            mst.append([u, v])
            if len(mst) == n - 1:  # MST has n-1 edges
                break
    
    return mst

def find_critical_edges(edges: List[List[int]], n: int, 
                       weights: Optional[List[int]] = None) -> List[List[int]]:
    """
    Find critical edges in a graph (edges that must be in MST).
    
    Example:
    Input: edges = [[0,1], [1,2], [2,0]], weights = [1,2,3], n = 3
    Output: [[0,1]] (Critical edge with weight 1)
    
    Time Complexity: O(E * log E)
    Space Complexity: O(n)
    """
    if weights is None:
        weights = [1] * len(edges)
    
    # Sort edges by weight
    sorted_edges = sorted(zip(edges, weights), key=lambda x: x[1])
    
    # Find MST weight
    uf = UnionFind(n)
    mst_weight = 0
    for (u, v), w in sorted_edges:
        if uf.union(u, v):
            mst_weight += w
    
    # Find critical edges
    critical_edges = []
    for i, ((u, v), w) in enumerate(sorted_edges):
        # Try excluding this edge
        uf = UnionFind(n)
        new_weight = 0
        for j, ((x, y), weight) in enumerate(sorted_edges):
            if i != j and uf.union(x, y):
                new_weight += weight
        
        # If excluding this edge increases MST weight, it's critical
        if new_weight > mst_weight or not uf.connected(u, v):
            critical_edges.append([u, v])
    
    return critical_edges

# Example usage
if __name__ == "__main__":
    def test_union_find():
        """Test the union-find implementations."""
        # Test cases
        n = 5
        edges = [[0,1], [1,2], [3,4], [2,3]]
        weights = [1, 2, 3, 4]
        
        print("\nTesting Union-Find Operations:")
        print(f"Graph with {n} vertices and edges: {edges}")
        print(f"Edge weights: {weights}")
        
        # Test connected components
        components = find_connected_components(edges, n)
        print("\nConnected Components:")
        for i, component in enumerate(components):
            print(f"Component {i+1}: {component}")
        
        # Test cycle detection
        has_cycle = detect_cycle(edges, n)
        print("\nCycle Detection:")
        print(f"Graph has cycle: {has_cycle}")
        
        # Test minimum spanning tree
        mst = find_minimum_spanning_tree(edges, n, weights)
        print("\nMinimum Spanning Tree:")
        print(f"Edges: {mst}")
        
        # Test critical edges
        critical = find_critical_edges(edges, n, weights)
        print("\nCritical Edges:")
        print(f"Edges: {critical}")
        
        # Print explanation
        print("\nExplanation:")
        print("1. Union-Find Data Structure:")
        print("   - Path compression and union by rank")
        print("   - O(α(n)) amortized time for operations")
        print("2. Connected Components:")
        print("   - Uses union-find to find all components")
        print("   - O(E * α(n)) time complexity")
        print("3. Cycle Detection:")
        print("   - Detects cycles during union operations")
        print("   - Same time complexity")
        print("4. Minimum Spanning Tree:")
        print("   - Uses Kruskal's algorithm")
        print("   - O(E * log E) time complexity")
        print("5. Critical Edges:")
        print("   - Finds edges that must be in MST")
        print("   - O(E * log E) time complexity")
        print()
    
    # Run tests
    test_union_find() 