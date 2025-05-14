import threading
import time
import queue
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict

class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class Task:
    """Represents a task with dependencies and resource requirements."""
    id: str
    func: Callable[..., Any]
    args: tuple = ()
    kwargs: dict = None
    dependencies: Set[str] = None
    required_resources: Set[str] = None
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 0
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.dependencies is None:
            self.dependencies = set()
        if self.required_resources is None:
            self.required_resources = set()

@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    start_time: float = 0.0
    end_time: float = 0.0
    retry_count: int = 0

class ResourceManager:
    """Manages resource allocation and deallocation."""
    
    def __init__(self):
        self.available_resources: Set[str] = set()
        self.allocated_resources: Dict[str, str] = {}  # resource -> task_id
        self.lock = threading.Lock()
    
    def add_resource(self, resource: str) -> None:
        """Add a resource to the pool."""
        with self.lock:
            self.available_resources.add(resource)
    
    def remove_resource(self, resource: str) -> None:
        """Remove a resource from the pool."""
        with self.lock:
            self.available_resources.discard(resource)
            if resource in self.allocated_resources:
                del self.allocated_resources[resource]
    
    def allocate_resources(
        self,
        task_id: str,
        resources: Set[str]
    ) -> bool:
        """
        Try to allocate resources for a task.
        Returns True if all resources were allocated.
        """
        with self.lock:
            # Check if all resources are available
            if not resources.issubset(self.available_resources):
                return False
            
            # Allocate resources
            for resource in resources:
                self.available_resources.remove(resource)
                self.allocated_resources[resource] = task_id
            
            return True
    
    def release_resources(self, task_id: str) -> None:
        """Release all resources allocated to a task."""
        with self.lock:
            resources_to_release = [
                resource for resource, tid in self.allocated_resources.items()
                if tid == task_id
            ]
            for resource in resources_to_release:
                del self.allocated_resources[resource]
                self.available_resources.add(resource)

class TaskExecutor:
    """
    Executes tasks in parallel while respecting dependencies and resource constraints.
    
    Features:
    - Parallel task execution
    - Dependency management
    - Resource constraints
    - Task prioritization
    - Retry mechanism
    - Progress monitoring
    - Graceful shutdown
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        resource_manager: Optional[ResourceManager] = None
    ):
        """
        Initialize the task executor.
        
        Args:
            max_workers: Maximum number of worker threads
            resource_manager: Optional resource manager
        """
        self.max_workers = max_workers
        self.resource_manager = resource_manager or ResourceManager()
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.results: Dict[str, TaskResult] = {}
        self.task_lock = threading.Lock()
        
        # Task queue (priority queue)
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )
        self.worker_thread.start()
    
    def submit_task(self, task: Task) -> None:
        """
        Submit a task for execution.
        
        Args:
            task: Task to execute
        """
        with self.task_lock:
            if task.id in self.tasks:
                raise ValueError(f"Task {task.id} already exists")
            
            self.tasks[task.id] = task
            self.results[task.id] = TaskResult(
                task_id=task.id,
                status=TaskStatus.PENDING
            )
            
            # Check if task can be queued
            if self._can_queue_task(task):
                self._queue_task(task)
    
    def _can_queue_task(self, task: Task) -> bool:
        """Check if a task can be queued (dependencies satisfied)."""
        for dep_id in task.dependencies:
            if dep_id not in self.results:
                return False
            if self.results[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True
    
    def _queue_task(self, task: Task) -> None:
        """Queue a task for execution."""
        # Priority is negative so higher priority tasks come first
        self.task_queue.put((-task.priority, task.id, task))
    
    def _worker_loop(self) -> None:
        """Main worker loop that processes tasks."""
        while self.running:
            try:
                # Get next task
                _, task_id, task = self.task_queue.get(timeout=0.1)
                
                # Check if task can run
                if not self._can_queue_task(task):
                    self.task_queue.put((-task.priority, task_id, task))
                    continue
                
                # Try to allocate resources
                if task.required_resources:
                    if not self.resource_manager.allocate_resources(
                        task_id,
                        task.required_resources
                    ):
                        self.task_queue.put((-task.priority, task_id, task))
                        continue
                
                # Update task status
                with self.task_lock:
                    self.results[task_id].status = TaskStatus.RUNNING
                    self.results[task_id].start_time = time.time()
                
                # Execute task
                future = self.executor.submit(
                    self._execute_task,
                    task
                )
                future.add_done_callback(
                    lambda f: self._handle_task_completion(task, f)
                )
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in worker loop: {e}")
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a task with timeout and retry logic."""
        try:
            if task.timeout:
                # Execute with timeout
                future = self.executor.submit(task.func, *task.args, **task.kwargs)
                return future.result(timeout=task.timeout)
            else:
                # Execute without timeout
                return task.func(*task.args, **task.kwargs)
        
        except Exception as e:
            if task.retry_count < task.max_retries:
                # Retry task
                task.retry_count += 1
                time.sleep(1)  # Wait before retry
                return self._execute_task(task)
            raise
    
    def _handle_task_completion(
        self,
        task: Task,
        future: Future
    ) -> None:
        """Handle task completion and update status."""
        try:
            result = future.result()
            with self.task_lock:
                self.results[task.id].status = TaskStatus.COMPLETED
                self.results[task.id].result = result
                self.results[task.id].end_time = time.time()
                self.results[task.id].retry_count = task.retry_count
            
            # Release resources
            if task.required_resources:
                self.resource_manager.release_resources(task.id)
            
            # Check dependent tasks
            self._check_dependent_tasks(task.id)
        
        except Exception as e:
            with self.task_lock:
                self.results[task.id].status = TaskStatus.FAILED
                self.results[task.id].error = e
                self.results[task.id].end_time = time.time()
                self.results[task.id].retry_count = task.retry_count
            
            # Release resources
            if task.required_resources:
                self.resource_manager.release_resources(task.id)
    
    def _check_dependent_tasks(self, completed_task_id: str) -> None:
        """Check and queue tasks that depend on the completed task."""
        with self.task_lock:
            for task_id, task in self.tasks.items():
                if (completed_task_id in task.dependencies and
                    self.results[task_id].status == TaskStatus.PENDING and
                    self._can_queue_task(task)):
                    self._queue_task(task)
    
    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get the status of a task."""
        with self.task_lock:
            return self.results.get(task_id)
    
    def get_all_statuses(self) -> Dict[str, TaskResult]:
        """Get status of all tasks."""
        with self.task_lock:
            return dict(self.results)
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the task executor.
        
        Args:
            wait: Whether to wait for tasks to complete
        """
        self.running = False
        self.executor.shutdown(wait=wait)

# Example usage
if __name__ == "__main__":
    def test_task_executor():
        """Test the task executor implementation."""
        # Create resource manager
        resource_manager = ResourceManager()
        for i in range(3):
            resource_manager.add_resource(f"resource_{i}")
        
        # Create task executor
        executor = TaskExecutor(
            max_workers=2,
            resource_manager=resource_manager
        )
        
        # Task functions
        def task_a():
            print("Task A running")
            time.sleep(1)
            return "A"
        
        def task_b():
            print("Task B running")
            time.sleep(2)
            return "B"
        
        def task_c():
            print("Task C running")
            time.sleep(1)
            return "C"
        
        def task_d():
            print("Task D running")
            time.sleep(3)
            return "D"
        
        # Create tasks
        tasks = [
            Task(
                id="A",
                func=task_a,
                required_resources={"resource_0"},
                priority=1
            ),
            Task(
                id="B",
                func=task_b,
                dependencies={"A"},
                required_resources={"resource_1"},
                priority=2
            ),
            Task(
                id="C",
                func=task_c,
                dependencies={"A"},
                required_resources={"resource_2"},
                priority=1
            ),
            Task(
                id="D",
                func=task_d,
                dependencies={"B", "C"},
                required_resources={"resource_0", "resource_1"},
                priority=3,
                timeout=5.0,
                max_retries=1
            )
        ]
        
        # Submit tasks
        for task in tasks:
            executor.submit_task(task)
        
        # Monitor progress
        try:
            while True:
                statuses = executor.get_all_statuses()
                print("\nTask Status:")
                for task_id, result in statuses.items():
                    print(f"{task_id}: {result.status.value}")
                    if result.status == TaskStatus.COMPLETED:
                        print(f"  Result: {result.result}")
                        print(f"  Time: {result.end_time - result.start_time:.1f}s")
                    elif result.status == TaskStatus.FAILED:
                        print(f"  Error: {result.error}")
                        print(f"  Retries: {result.retry_count}")
                
                # Check if all tasks are done
                if all(
                    result.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}
                    for result in statuses.values()
                ):
                    break
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            executor.shutdown()
            
            # Print final statistics
            statuses = executor.get_all_statuses()
            print("\nFinal Statistics:")
            completed = sum(
                1 for result in statuses.values()
                if result.status == TaskStatus.COMPLETED
            )
            failed = sum(
                1 for result in statuses.values()
                if result.status == TaskStatus.FAILED
            )
            print(f"Total Tasks: {len(statuses)}")
            print(f"Completed: {completed}")
            print(f"Failed: {failed}")
    
    # Run test
    test_task_executor() 