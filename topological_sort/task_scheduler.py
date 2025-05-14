from typing import List, Dict, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from heapq import heappush, heappop

@dataclass
class Task:
    """Represents a task with its properties."""
    id: int
    duration: int
    resources: List[str]  # Required resources
    priority: int = 0     # Higher number means higher priority

def schedule_tasks(tasks: List[Task], dependencies: List[List[int]], 
                  available_resources: Dict[str, int]) -> List[int]:
    """
    Schedule tasks respecting dependencies and resource constraints.
    Returns list of task IDs in execution order.
    
    Example:
    Input: tasks = [
        Task(0, 2, ["CPU"]),
        Task(1, 3, ["CPU", "GPU"]),
        Task(2, 1, ["GPU"])
    ], dependencies = [[1,0], [2,1]], available_resources = {"CPU": 2, "GPU": 1}
    Output: [0, 1, 2] (Valid schedule: 0 -> 1 -> 2)
    
    Time Complexity: O(V + E + V * log V) where V is number of tasks and E is dependencies
    Space Complexity: O(V + E)
    """
    # Build adjacency list and track in-degrees
    graph = defaultdict(list)
    in_degree = [0] * len(tasks)
    task_map = {task.id: task for task in tasks}
    
    for dependent, prerequisite in dependencies:
        graph[prerequisite].append(dependent)
        in_degree[dependent] += 1
    
    # Initialize queue with tasks having no dependencies
    queue = []
    for i, task in enumerate(tasks):
        if in_degree[task.id] == 0:
            # Use negative priority for max heap behavior
            heappush(queue, (-task.priority, task.id))
    
    # Track available resources
    resources = available_resources.copy()
    result = []
    
    # Process tasks
    while queue:
        # Get highest priority task
        _, task_id = heappop(queue)
        task = task_map[task_id]
        
        # Check if resources are available
        can_execute = True
        for resource in task.resources:
            if resources[resource] <= 0:
                can_execute = False
                break
        
        if can_execute:
            # Execute task
            result.append(task_id)
            
            # Update resources
            for resource in task.resources:
                resources[resource] -= 1
            
            # Add dependent tasks
            for dependent in graph[task_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    heappush(queue, (-task_map[dependent].priority, dependent))
            
            # Restore resources
            for resource in task.resources:
                resources[resource] += 1
        else:
            # Put task back in queue with lower priority
            heappush(queue, (-task.priority - 1, task_id))
    
    return result if len(result) == len(tasks) else []

def schedule_tasks_with_deadlines(tasks: List[Task], dependencies: List[List[int]],
                                deadlines: Dict[int, int]) -> List[int]:
    """
    Schedule tasks respecting dependencies and deadlines.
    Returns list of task IDs in execution order.
    
    Time Complexity: O(V + E + V * log V)
    Space Complexity: O(V + E)
    """
    # Build adjacency list and track in-degrees
    graph = defaultdict(list)
    in_degree = [0] * len(tasks)
    task_map = {task.id: task for task in tasks}
    
    for dependent, prerequisite in dependencies:
        graph[prerequisite].append(dependent)
        in_degree[dependent] += 1
    
    # Initialize queue with tasks having no dependencies
    queue = []
    for task in tasks:
        if in_degree[task.id] == 0:
            # Use deadline as priority (earlier deadline = higher priority)
            heappush(queue, (deadlines[task.id], task.id))
    
    result = []
    current_time = 0
    
    # Process tasks
    while queue:
        deadline, task_id = heappop(queue)
        task = task_map[task_id]
        
        # Check if we can meet the deadline
        if current_time + task.duration > deadline:
            return []  # Cannot meet deadline
        
        # Execute task
        result.append(task_id)
        current_time += task.duration
        
        # Add dependent tasks
        for dependent in graph[task_id]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                heappush(queue, (deadlines[dependent], dependent))
    
    return result if len(result) == len(tasks) else []

def schedule_tasks_with_parallel_execution(tasks: List[Task], dependencies: List[List[int]],
                                         available_resources: Dict[str, int]) -> List[List[int]]:
    """
    Schedule tasks for parallel execution respecting dependencies and resources.
    Returns list of task lists, where each inner list represents tasks that can run in parallel.
    
    Time Complexity: O(V + E + V * log V)
    Space Complexity: O(V + E)
    """
    # Build adjacency list and track in-degrees
    graph = defaultdict(list)
    in_degree = [0] * len(tasks)
    task_map = {task.id: task for task in tasks}
    
    for dependent, prerequisite in dependencies:
        graph[prerequisite].append(dependent)
        in_degree[dependent] += 1
    
    # Initialize queue with tasks having no dependencies
    queue = []
    for task in tasks:
        if in_degree[task.id] == 0:
            heappush(queue, (-task.priority, task.id))
    
    # Track available resources
    resources = available_resources.copy()
    result = []
    
    # Process tasks in parallel
    while queue:
        parallel_tasks = []
        next_queue = []
        
        # Try to execute as many tasks as possible in parallel
        while queue:
            _, task_id = heappop(queue)
            task = task_map[task_id]
            
            # Check if resources are available
            can_execute = True
            for resource in task.resources:
                if resources[resource] <= 0:
                    can_execute = False
                    break
            
            if can_execute:
                # Execute task
                parallel_tasks.append(task_id)
                
                # Update resources
                for resource in task.resources:
                    resources[resource] -= 1
                
                # Add dependent tasks
                for dependent in graph[task_id]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        heappush(next_queue, (-task_map[dependent].priority, dependent))
            else:
                # Put task back in queue
                heappush(next_queue, (-task.priority, task_id))
        
        if parallel_tasks:
            result.append(parallel_tasks)
        
        # Restore resources
        for task_id in parallel_tasks:
            task = task_map[task_id]
            for resource in task.resources:
                resources[resource] += 1
        
        queue = next_queue
    
    return result if sum(len(parallel) for parallel in result) == len(tasks) else []

# Example usage
if __name__ == "__main__":
    def test_task_scheduler():
        """Test the task scheduling implementations."""
        # Test cases
        tasks = [
            Task(0, 2, ["CPU"], 1),
            Task(1, 3, ["CPU", "GPU"], 2),
            Task(2, 1, ["GPU"], 1),
            Task(3, 4, ["CPU"], 3)
        ]
        dependencies = [[1,0], [2,1], [3,1]]
        resources = {"CPU": 2, "GPU": 1}
        deadlines = {0: 2, 1: 5, 2: 6, 3: 10}
        
        print("\nTesting Task Scheduling:")
        print("Tasks:")
        for task in tasks:
            print(f"Task {task.id}: duration={task.duration}, "
                  f"resources={task.resources}, priority={task.priority}")
        print(f"Dependencies: {dependencies}")
        print(f"Available resources: {resources}")
        print(f"Deadlines: {deadlines}")
        
        # Test basic scheduling
        schedule = schedule_tasks(tasks, dependencies, resources)
        print("\nBasic Schedule:")
        print(f"Task order: {schedule}")
        
        # Test scheduling with deadlines
        schedule_deadlines = schedule_tasks_with_deadlines(tasks, dependencies, deadlines)
        print("\nSchedule with Deadlines:")
        print(f"Task order: {schedule_deadlines}")
        
        # Test parallel scheduling
        parallel_schedule = schedule_tasks_with_parallel_execution(tasks, dependencies, resources)
        print("\nParallel Schedule:")
        for i, parallel_tasks in enumerate(parallel_schedule):
            print(f"Time step {i}: {parallel_tasks}")
        
        # Print explanation
        print("\nExplanation:")
        print("1. Basic Task Scheduling:")
        print("   - Uses topological sort with resource constraints")
        print("   - Prioritizes tasks based on priority")
        print("   - O(V + E + V * log V) time complexity")
        print("2. Scheduling with Deadlines:")
        print("   - Considers task durations and deadlines")
        print("   - Uses deadline as priority")
        print("   - Same time complexity")
        print("3. Parallel Scheduling:")
        print("   - Groups tasks that can run in parallel")
        print("   - Respects dependencies and resources")
        print("   - Returns list of parallel task groups")
        print()
    
    # Run tests
    test_task_scheduler() 