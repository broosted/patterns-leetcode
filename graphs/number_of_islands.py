from typing import List, Deque, Tuple
from collections import deque

def num_islands_dfs(grid: List[List[str]]) -> int:
    """
    Find the number of islands in a 2D grid using DFS.
    An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
    
    Example:
    Input:
        1 1 0 0 0
        1 1 0 0 0
        0 0 1 0 0
        0 0 0 1 1
    Output: 3
    
    Time Complexity: O(m * n) where m and n are the dimensions of the grid
    Space Complexity: O(m * n) for the recursion stack
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(r: int, c: int) -> None:
        """DFS helper function to mark visited cells."""
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] != '1'):
            return
        
        # Mark current cell as visited
        grid[r][c] = '0'
        
        # Explore all four directions
        dfs(r + 1, c)  # down
        dfs(r - 1, c)  # up
        dfs(r, c + 1)  # right
        dfs(r, c - 1)  # left
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    
    return count

def num_islands_bfs(grid: List[List[str]]) -> int:
    """
    Find the number of islands in a 2D grid using BFS.
    
    Time Complexity: O(m * n)
    Space Complexity: O(min(m, n)) for the queue
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def bfs(r: int, c: int) -> None:
        """BFS helper function to mark visited cells."""
        queue: Deque[Tuple[int, int]] = deque([(r, c)])
        grid[r][c] = '0'  # Mark as visited
        
        while queue:
            curr_r, curr_c = queue.popleft()
            
            # Check all four directions
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_r, new_c = curr_r + dr, curr_c + dc
                
                if (0 <= new_r < rows and 0 <= new_c < cols and 
                    grid[new_r][new_c] == '1'):
                    grid[new_r][new_c] = '0'  # Mark as visited
                    queue.append((new_r, new_c))
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                bfs(r, c)
    
    return count

def num_islands_union_find(grid: List[List[str]]) -> int:
    """
    Find the number of islands using Union-Find (Disjoint Set) data structure.
    
    Time Complexity: O(m * n * α(m * n)) where α is the inverse Ackermann function
    Space Complexity: O(m * n)
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    
    class UnionFind:
        def __init__(self, size: int):
            self.parent = list(range(size))
            self.rank = [0] * size
            self.count = size
        
        def find(self, x: int) -> int:
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x: int, y: int) -> None:
            root_x = self.find(x)
            root_y = self.find(y)
            
            if root_x == root_y:
                return
            
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            
            self.count -= 1
    
    # Initialize Union-Find
    uf = UnionFind(rows * cols)
    water_count = 0
    
    # First pass: count water cells and initialize Union-Find
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '0':
                water_count += 1
    
    # Second pass: union adjacent land cells
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                curr = r * cols + c
                
                # Check right and down neighbors
                for dr, dc in [(0, 1), (1, 0)]:
                    new_r, new_c = r + dr, c + dc
                    if (new_r < rows and new_c < cols and 
                        grid[new_r][new_c] == '1'):
                        uf.union(curr, new_r * cols + new_c)
    
    return uf.count - water_count

# Example usage
if __name__ == "__main__":
    def create_test_grid() -> List[List[str]]:
        """Create a test grid."""
        return [
            ['1', '1', '0', '0', '0'],
            ['1', '1', '0', '0', '0'],
            ['0', '0', '1', '0', '0'],
            ['0', '0', '0', '1', '1']
        ]
    
    def create_test_grid_2() -> List[List[str]]:
        """Create another test grid."""
        return [
            ['1', '1', '1', '1', '0'],
            ['1', '1', '0', '1', '0'],
            ['1', '1', '0', '0', '0'],
            ['0', '0', '0', '0', '0']
        ]
    
    # Test cases
    test_grids = [
        (create_test_grid(), "Test Grid 1"),
        (create_test_grid_2(), "Test Grid 2"),
        ([], "Empty Grid"),
        ([[]], "Empty Row Grid"),
    ]
    
    for grid, name in test_grids:
        print(f"\nTesting {name}:")
        
        # Create copies of the grid for each method
        grid_dfs = [row[:] for row in grid] if grid else []
        grid_bfs = [row[:] for row in grid] if grid else []
        grid_uf = [row[:] for row in grid] if grid else []
        
        # Test DFS approach
        result1 = num_islands_dfs(grid_dfs)
        print("\nDFS Approach:")
        print(f"Number of islands: {result1}")
        
        # Test BFS approach
        result2 = num_islands_bfs(grid_bfs)
        print("\nBFS Approach:")
        print(f"Number of islands: {result2}")
        
        # Test Union-Find approach
        result3 = num_islands_union_find(grid_uf)
        print("\nUnion-Find Approach:")
        print(f"Number of islands: {result3}")
        
        # Print explanation
        if grid and grid[0]:
            print("\nExplanation:")
            print("1. DFS Approach:")
            print("   - Use depth-first search to explore each island")
            print("   - Mark visited cells by changing '1' to '0'")
            print("   - Time Complexity: O(m * n), Space Complexity: O(m * n)")
            print("2. BFS Approach:")
            print("   - Use breadth-first search to explore each island")
            print("   - More efficient space complexity for wide islands")
            print("   - Time Complexity: O(m * n), Space Complexity: O(min(m, n))")
            print("3. Union-Find Approach:")
            print("   - Use disjoint set data structure to track connected components")
            print("   - More efficient for dynamic updates")
            print("   - Time Complexity: O(m * n * α(m * n))")
        print() 