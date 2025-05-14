from typing import List, Dict, Set, Deque
from collections import defaultdict, deque

def can_finish_dfs(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    Check if it's possible to finish all courses using DFS to detect cycles.
    
    Example:
    Input: num_courses = 4, prerequisites = [[1,0], [2,0], [3,1], [3,2]]
    Output: True
    Explanation: One valid course order is [0,1,2,3]
    
    Time Complexity: O(V + E) where V is the number of courses and E is the number of prerequisites
    Space Complexity: O(V + E)
    """
    # Build adjacency list
    graph: Dict[int, List[int]] = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # States: 0 = unvisited, 1 = visiting, 2 = visited
    visited = [0] * num_courses
    
    def has_cycle(course: int) -> bool:
        """DFS helper to detect cycles."""
        if visited[course] == 1:  # Cycle detected
            return True
        if visited[course] == 2:  # Already visited
            return False
        
        visited[course] = 1  # Mark as visiting
        
        # Visit all neighbors
        for neighbor in graph[course]:
            if has_cycle(neighbor):
                return True
        
        visited[course] = 2  # Mark as visited
        return False
    
    # Check each course for cycles
    for course in range(num_courses):
        if has_cycle(course):
            return False
    
    return True

def can_finish_bfs(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    Check if it's possible to finish all courses using BFS (Kahn's algorithm).
    
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    """
    # Build adjacency list and in-degree count
    graph: Dict[int, List[int]] = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Initialize queue with courses having no prerequisites
    queue: Deque[int] = deque()
    for course in range(num_courses):
        if in_degree[course] == 0:
            queue.append(course)
    
    # Process courses
    count = 0
    while queue:
        course = queue.popleft()
        count += 1
        
        # Reduce in-degree for all neighbors
        for neighbor in graph[course]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return count == num_courses

def find_order(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Find a valid course order if possible, otherwise return an empty list.
    
    Example:
    Input: num_courses = 4, prerequisites = [[1,0], [2,0], [3,1], [3,2]]
    Output: [0,1,2,3] or [0,2,1,3]
    
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    """
    # Build adjacency list and in-degree count
    graph: Dict[int, List[int]] = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Initialize queue with courses having no prerequisites
    queue: Deque[int] = deque()
    for course in range(num_courses):
        if in_degree[course] == 0:
            queue.append(course)
    
    # Process courses
    result = []
    while queue:
        course = queue.popleft()
        result.append(course)
        
        # Reduce in-degree for all neighbors
        for neighbor in graph[course]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == num_courses else []

def find_all_orders(num_courses: int, prerequisites: List[List[int]]) -> List[List[int]]:
    """
    Find all possible valid course orders using backtracking.
    
    Time Complexity: O(V! * E) where V is the number of courses
    Space Complexity: O(V + E)
    """
    # Build adjacency list and in-degree count
    graph: Dict[int, List[int]] = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    result = []
    visited = [False] * num_courses
    
    def backtrack(path: List[int], in_degree_copy: List[int]) -> None:
        """Backtracking helper to find all valid orders."""
        if len(path) == num_courses:
            result.append(path[:])
            return
        
        for course in range(num_courses):
            if not visited[course] and in_degree_copy[course] == 0:
                # Choose
                visited[course] = True
                path.append(course)
                
                # Update in-degree for neighbors
                for neighbor in graph[course]:
                    in_degree_copy[neighbor] -= 1
                
                # Explore
                backtrack(path, in_degree_copy)
                
                # Unchoose
                visited[course] = False
                path.pop()
                for neighbor in graph[course]:
                    in_degree_copy[neighbor] += 1
    
    backtrack([], in_degree[:])
    return result

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        (4, [[1,0], [2,0], [3,1], [3,2]], "Valid Schedule"),
        (2, [[1,0], [0,1]], "Invalid Schedule (Cycle)"),
        (3, [[1,0], [2,1]], "Simple Valid Schedule"),
        (1, [], "Single Course"),
        (0, [], "No Courses"),
    ]
    
    for num_courses, prerequisites, name in test_cases:
        print(f"\nTesting {name}:")
        print(f"Number of courses: {num_courses}")
        print(f"Prerequisites: {prerequisites}")
        
        # Test DFS approach
        result1 = can_finish_dfs(num_courses, prerequisites)
        print("\nDFS Approach:")
        print(f"Can finish all courses: {result1}")
        
        # Test BFS approach
        result2 = can_finish_bfs(num_courses, prerequisites)
        print("\nBFS Approach:")
        print(f"Can finish all courses: {result2}")
        
        # Test finding a valid order
        result3 = find_order(num_courses, prerequisites)
        print("\nFinding a Valid Order:")
        print(f"Valid order: {result3}")
        
        # Test finding all valid orders
        if num_courses <= 5:  # Limit to small inputs due to factorial complexity
            result4 = find_all_orders(num_courses, prerequisites)
            print("\nFinding All Valid Orders:")
            print(f"Number of valid orders: {len(result4)}")
            if result4:
                print("First few orders:")
                for order in result4[:3]:
                    print(f"  {order}")
        
        # Print explanation
        print("\nExplanation:")
        print("1. DFS Approach:")
        print("   - Use depth-first search to detect cycles")
        print("   - Three states for each node: unvisited, visiting, visited")
        print("   - Cycle exists if we encounter a 'visiting' node")
        print("2. BFS Approach (Kahn's Algorithm):")
        print("   - Use topological sort to find a valid order")
        print("   - Start with nodes having no incoming edges")
        print("   - Remove edges as we process nodes")
        print("3. Finding a Valid Order:")
        print("   - Modified BFS to return the actual order")
        print("   - Returns empty list if no valid order exists")
        print("4. Finding All Valid Orders:")
        print("   - Use backtracking to find all possible orders")
        print("   - Only practical for small inputs due to factorial complexity")
        print() 