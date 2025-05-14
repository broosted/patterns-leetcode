from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
from heapq import heappush, heappop

@dataclass
class Package:
    """Represents a package with its properties."""
    name: str
    version: str
    dependencies: List[Tuple[str, str]]  # List of (package_name, version_constraint)
    size: int = 0  # Size in MB
    priority: int = 0  # Higher number means higher priority

def resolve_dependencies(packages: List[Package]) -> List[Package]:
    """
    Resolve package dependencies using topological sort.
    Returns list of packages in installation order.
    
    Example:
    Input: packages = [
        Package("A", "1.0", [("B", ">=2.0"), ("C", "1.0")]),
        Package("B", "2.0", [("D", "1.0")]),
        Package("C", "1.0", []),
        Package("D", "1.0", [])
    ]
    Output: [D, B, C, A] (Valid installation order)
    
    Time Complexity: O(V + E) where V is number of packages and E is dependencies
    Space Complexity: O(V + E)
    """
    # Build adjacency list and track in-degrees
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    package_map = {pkg.name: pkg for pkg in packages}
    
    # Build dependency graph
    for package in packages:
        for dep_name, _ in package.dependencies:
            if dep_name in package_map:
                graph[dep_name].append(package.name)
                in_degree[package.name] += 1
    
    # Initialize queue with packages having no dependencies
    queue = deque()
    for package in packages:
        if in_degree[package.name] == 0:
            queue.append(package.name)
    
    result = []
    
    # Process packages
    while queue:
        package_name = queue.popleft()
        package = package_map[package_name]
        result.append(package)
        
        # Add dependent packages
        for dependent in graph[package_name]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    
    return result if len(result) == len(packages) else []

def resolve_dependencies_with_conflicts(packages: List[Package]) -> List[Package]:
    """
    Resolve package dependencies while handling version conflicts.
    Returns list of packages in installation order, or empty list if conflicts exist.
    
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    """
    # Build version graph and track in-degrees
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    package_map = {(pkg.name, pkg.version): pkg for pkg in packages}
    
    def satisfies_constraint(version: str, constraint: str) -> bool:
        """Check if version satisfies the constraint."""
        if constraint.startswith(">="):
            return version >= constraint[2:]
        elif constraint.startswith("<="):
            return version <= constraint[2:]
        elif constraint.startswith(">"):
            return version > constraint[1:]
        elif constraint.startswith("<"):
            return version < constraint[1:]
        else:
            return version == constraint
    
    # Build dependency graph with version constraints
    for package in packages:
        for dep_name, dep_constraint in package.dependencies:
            for (pkg_name, pkg_version), pkg in package_map.items():
                if pkg_name == dep_name and satisfies_constraint(pkg_version, dep_constraint):
                    graph[(pkg_name, pkg_version)].append((package.name, package.version))
                    in_degree[(package.name, package.version)] += 1
    
    # Initialize queue with packages having no dependencies
    queue = deque()
    for (pkg_name, pkg_version), pkg in package_map.items():
        if in_degree[(pkg_name, pkg_version)] == 0:
            queue.append((pkg_name, pkg_version))
    
    result = []
    installed = set()  # Track installed package names
    
    # Process packages
    while queue:
        pkg_key = queue.popleft()
        pkg_name, pkg_version = pkg_key
        package = package_map[pkg_key]
        
        # Check for conflicts
        if pkg_name in installed:
            return []  # Conflict detected
        
        result.append(package)
        installed.add(pkg_name)
        
        # Add dependent packages
        for dependent in graph[pkg_key]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    
    return result if len(result) == len(packages) else []

def resolve_dependencies_with_optimization(packages: List[Package], 
                                         available_space: int) -> List[Package]:
    """
    Resolve package dependencies while optimizing for space and priority.
    Returns list of packages in installation order.
    
    Time Complexity: O(V + E + V * log V)
    Space Complexity: O(V + E)
    """
    # Build adjacency list and track in-degrees
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    package_map = {pkg.name: pkg for pkg in packages}
    
    # Build dependency graph
    for package in packages:
        for dep_name, _ in package.dependencies:
            if dep_name in package_map:
                graph[dep_name].append(package.name)
                in_degree[package.name] += 1
    
    # Initialize queue with packages having no dependencies
    queue = []
    for package in packages:
        if in_degree[package.name] == 0:
            # Use negative priority for max heap behavior
            heappush(queue, (-package.priority, package.size, package.name))
    
    result = []
    remaining_space = available_space
    
    # Process packages
    while queue:
        priority, size, package_name = heappop(queue)
        package = package_map[package_name]
        
        # Check if we have enough space
        if size > remaining_space:
            # Put package back with lower priority
            heappush(queue, (priority + 1, size, package_name))
            continue
        
        result.append(package)
        remaining_space -= size
        
        # Add dependent packages
        for dependent in graph[package_name]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                dep_package = package_map[dependent]
                heappush(queue, (-dep_package.priority, dep_package.size, dependent))
    
    return result if len(result) == len(packages) else []

# Example usage
if __name__ == "__main__":
    def test_package_dependencies():
        """Test the package dependency resolution implementations."""
        # Test cases
        packages = [
            Package("A", "1.0", [("B", ">=2.0"), ("C", "1.0")], 100, 1),
            Package("B", "2.0", [("D", "1.0")], 200, 2),
            Package("C", "1.0", [], 150, 1),
            Package("D", "1.0", [], 50, 3),
            Package("B", "1.0", [("D", "1.0")], 180, 1)  # Different version
        ]
        available_space = 500
        
        print("\nTesting Package Dependency Resolution:")
        print("Packages:")
        for package in packages:
            print(f"Package {package.name} v{package.version}: "
                  f"deps={package.dependencies}, size={package.size}MB, "
                  f"priority={package.priority}")
        
        # Test basic dependency resolution
        resolved = resolve_dependencies(packages)
        print("\nBasic Resolution:")
        if resolved:
            print("Installation order:")
            for package in resolved:
                print(f"- {package.name} v{package.version}")
        else:
            print("No valid resolution found")
        
        # Test resolution with conflicts
        resolved_conflicts = resolve_dependencies_with_conflicts(packages)
        print("\nResolution with Conflicts:")
        if resolved_conflicts:
            print("Installation order:")
            for package in resolved_conflicts:
                print(f"- {package.name} v{package.version}")
        else:
            print("Conflicts detected")
        
        # Test optimized resolution
        resolved_optimized = resolve_dependencies_with_optimization(packages, available_space)
        print("\nOptimized Resolution:")
        if resolved_optimized:
            print("Installation order:")
            total_size = 0
            for package in resolved_optimized:
                print(f"- {package.name} v{package.version} "
                      f"(size: {package.size}MB, priority: {package.priority})")
                total_size += package.size
            print(f"Total size: {total_size}MB")
        else:
            print("No valid resolution found")
        
        # Print explanation
        print("\nExplanation:")
        print("1. Basic Dependency Resolution:")
        print("   - Uses topological sort to find valid installation order")
        print("   - O(V + E) time complexity")
        print("2. Resolution with Conflicts:")
        print("   - Handles version constraints and conflicts")
        print("   - Detects incompatible versions")
        print("3. Optimized Resolution:")
        print("   - Considers package size and priority")
        print("   - Respects available space")
        print("   - O(V + E + V * log V) time complexity")
        print()
    
    # Run tests
    test_package_dependencies() 