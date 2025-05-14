from typing import List, Dict, Set, Tuple
from collections import defaultdict, deque

def can_finish_courses(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    Check if it's possible to finish all courses given prerequisites.
    Uses topological sort with cycle detection.
    
    Example:
    Input: num_courses = 4, prerequisites = [[1,0], [2,0], [3,1], [3,2]]
    Output: True (Valid order: 0 -> 1 -> 3, 0 -> 2 -> 3)
    
    Time Complexity: O(V + E) where V is number of courses and E is number of prerequisites
    Space Complexity: O(V + E)
    """
    # Build adjacency list
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Initialize queue with courses having no prerequisites
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    courses_taken = 0
    
    # Process courses
    while queue:
        course = queue.popleft()
        courses_taken += 1
        
        # Reduce in-degree for dependent courses
        for dependent in graph[course]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    
    return courses_taken == num_courses

def find_course_order(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Find a valid order to take all courses given prerequisites.
    Returns empty list if no valid order exists.
    
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    """
    # Build adjacency list
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Initialize queue with courses having no prerequisites
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    result = []
    
    # Process courses
    while queue:
        course = queue.popleft()
        result.append(course)
        
        # Reduce in-degree for dependent courses
        for dependent in graph[course]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    
    return result if len(result) == num_courses else []

def find_course_order_dfs(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Find a valid course order using DFS-based topological sort.
    
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    """
    # Build adjacency list
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # States: 0 = unvisited, 1 = visiting, 2 = visited
    state = [0] * num_courses
    result = []
    
    def has_cycle(course: int) -> bool:
        """Check if there's a cycle starting from the given course."""
        if state[course] == 1:  # Cycle detected
            return True
        if state[course] == 2:  # Already processed
            return False
        
        state[course] = 1  # Mark as visiting
        
        # Check all dependent courses
        for dependent in graph[course]:
            if has_cycle(dependent):
                return True
        
        state[course] = 2  # Mark as visited
        result.append(course)
        return False
    
    # Check each course
    for course in range(num_courses):
        if state[course] == 0 and has_cycle(course):
            return []
    
    return result[::-1]  # Reverse to get correct order

def find_course_schedule_with_deadlines(num_courses: int, prerequisites: List[List[int]], 
                                      deadlines: List[int]) -> List[int]:
    """
    Find a valid course order that respects deadlines.
    Returns empty list if no valid order exists.
    
    Example:
    Input: num_courses = 4, prerequisites = [[1,0], [2,0], [3,1], [3,2]],
           deadlines = [0, 2, 2, 3]
    Output: [0, 1, 2, 3] (Must take course 0 first, then 1 and 2 by time 2, then 3 by time 3)
    
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    """
    if len(deadlines) != num_courses:
        return []
    
    # Build adjacency list and track deadlines
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    course_deadlines = [(i, deadlines[i]) for i in range(num_courses)]
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Sort courses by deadline
    course_deadlines.sort(key=lambda x: x[1])
    
    # Process courses in deadline order
    result = []
    time = 0
    
    for course, deadline in course_deadlines:
        # Check if all prerequisites are taken
        if in_degree[course] > 0:
            return []
        
        # Take the course
        result.append(course)
        time += 1
        
        # Update dependent courses
        for dependent in graph[course]:
            in_degree[dependent] -= 1
        
        # Check if we're within deadline
        if time > deadline:
            return []
    
    return result

# Example usage
if __name__ == "__main__":
    def test_course_schedule():
        """Test the course scheduling implementations."""
        # Test cases
        test_cases = [
            (4, [[1,0], [2,0], [3,1], [3,2]], "Valid schedule"),
            (2, [[1,0], [0,1]], "Cycle detected"),
            (3, [[1,0], [2,1], [0,2]], "Complex cycle"),
            (3, [], "No prerequisites"),
            (4, [[1,0], [2,0], [3,1]], "Multiple paths")
        ]
        
        for num_courses, prerequisites, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"Number of courses: {num_courses}")
            print(f"Prerequisites: {prerequisites}")
            
            # Test if courses can be finished
            can_finish = can_finish_courses(num_courses, prerequisites)
            print(f"\nCan finish courses: {can_finish}")
            
            # Test finding course order (BFS)
            order_bfs = find_course_order(num_courses, prerequisites)
            print(f"Course order (BFS): {order_bfs}")
            
            # Test finding course order (DFS)
            order_dfs = find_course_order_dfs(num_courses, prerequisites)
            print(f"Course order (DFS): {order_dfs}")
            
            # Test with deadlines
            deadlines = [0] + [i for i in range(1, num_courses)]
            order_deadlines = find_course_schedule_with_deadlines(
                num_courses, prerequisites, deadlines)
            print(f"Course order with deadlines: {order_deadlines}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. Course Scheduling:")
            print("   - Uses topological sort to find valid order")
            print("   - Detects cycles in prerequisites")
            print("2. Approaches:")
            print("   - BFS (Kahn's algorithm):")
            print("     * Process nodes with no incoming edges")
            print("     * O(V + E) time complexity")
            print("   - DFS:")
            print("     * Use visited states to detect cycles")
            print("     * Same time complexity")
            print("3. With Deadlines:")
            print("   - Sort courses by deadline")
            print("   - Ensure prerequisites are taken in time")
            print("   - O(V + E) time complexity")
            print()
    
    # Run tests
    test_course_schedule() 