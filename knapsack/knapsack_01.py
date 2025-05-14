from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class Item:
    """Represents an item with weight and value."""
    weight: int
    value: int
    name: str = ""

def knapsack_01_dp(items: List[Item], capacity: int) -> Tuple[int, List[Item]]:
    """
    Solve the 0/1 knapsack problem using dynamic programming.
    
    Example:
    Input: items = [
        Item(2, 3, "A"),
        Item(3, 4, "B"),
        Item(4, 5, "C"),
        Item(5, 6, "D")
    ], capacity = 10
    Output: (13, [A, B, D])
    
    Time Complexity: O(n * capacity) where n is the number of items
    Space Complexity: O(n * capacity)
    """
    if not items or capacity <= 0:
        return 0, []
    
    n = len(items)
    # Initialize DP table
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    # Store selected items
    selected = [[False] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if items[i-1].weight <= w:
                # Try including current item
                include = items[i-1].value + dp[i-1][w - items[i-1].weight]
                # Try excluding current item
                exclude = dp[i-1][w]
                
                if include > exclude:
                    dp[i][w] = include
                    selected[i][w] = True
                else:
                    dp[i][w] = exclude
            else:
                dp[i][w] = dp[i-1][w]
    
    # Reconstruct the solution
    result = []
    w = capacity
    for i in range(n, 0, -1):
        if selected[i][w]:
            result.append(items[i-1])
            w -= items[i-1].weight
    
    return dp[n][capacity], result

def knapsack_01_optimized(items: List[Item], capacity: int) -> Tuple[int, List[Item]]:
    """
    Solve the 0/1 knapsack problem using optimized dynamic programming.
    Uses 1D array to reduce space complexity.
    
    Time Complexity: O(n * capacity)
    Space Complexity: O(capacity)
    """
    if not items or capacity <= 0:
        return 0, []
    
    n = len(items)
    # Initialize DP array
    dp = [0] * (capacity + 1)
    # Store selected items
    selected = [False] * (capacity + 1)
    # Store item indices for reconstruction
    item_indices = [-1] * (capacity + 1)
    
    # Fill DP array
    for i in range(n):
        for w in range(capacity, items[i].weight - 1, -1):
            if items[i].value + dp[w - items[i].weight] > dp[w]:
                dp[w] = items[i].value + dp[w - items[i].weight]
                selected[w] = True
                item_indices[w] = i
    
    # Reconstruct the solution
    result = []
    w = capacity
    while w > 0 and item_indices[w] != -1:
        i = item_indices[w]
        result.append(items[i])
        w -= items[i].weight
    
    return dp[capacity], result

def knapsack_01_with_constraints(items: List[Item], capacity: int, 
                                constraints: Dict[str, int]) -> Tuple[int, List[Item]]:
    """
    Solve the 0/1 knapsack problem with additional constraints on item selection.
    
    Example:
    Input: items = [
        Item(2, 3, "A"),
        Item(3, 4, "B"),
        Item(4, 5, "C"),
        Item(5, 6, "D")
    ], capacity = 10, constraints = {"A": 1, "B": 2, "C": 1, "D": 1}
    Output: (13, [A, B, D])
    
    Time Complexity: O(n * capacity * max_constraint)
    Space Complexity: O(n * capacity * max_constraint)
    """
    if not items or capacity <= 0:
        return 0, []
    
    n = len(items)
    max_constraint = max(constraints.values())
    
    # Initialize DP table
    dp = [[[0] * (capacity + 1) for _ in range(max_constraint + 1)] for _ in range(n + 1)]
    # Store selected items
    selected = [[[False] * (capacity + 1) for _ in range(max_constraint + 1)] for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        item = items[i-1]
        max_count = constraints.get(item.name, 1)
        
        for count in range(max_constraint + 1):
            for w in range(capacity + 1):
                # Try including current item
                if (count > 0 and item.weight <= w and 
                    dp[i-1][count-1][w - item.weight] + item.value > dp[i-1][count][w]):
                    dp[i][count][w] = dp[i-1][count-1][w - item.weight] + item.value
                    selected[i][count][w] = True
                else:
                    dp[i][count][w] = dp[i-1][count][w]
    
    # Find maximum value
    max_value = 0
    max_count = 0
    for count in range(max_constraint + 1):
        if dp[n][count][capacity] > max_value:
            max_value = dp[n][count][capacity]
            max_count = count
    
    # Reconstruct the solution
    result = []
    w = capacity
    count = max_count
    for i in range(n, 0, -1):
        if selected[i][count][w]:
            result.append(items[i-1])
            w -= items[i-1].weight
            count -= 1
    
    return max_value, result

# Example usage
if __name__ == "__main__":
    def test_knapsack_01():
        """Test the 0/1 knapsack implementations."""
        # Test cases
        items = [
            Item(2, 3, "A"),
            Item(3, 4, "B"),
            Item(4, 5, "C"),
            Item(5, 6, "D")
        ]
        capacity = 10
        constraints = {"A": 1, "B": 2, "C": 1, "D": 1}
        
        print("\nTesting 0/1 Knapsack:")
        print("Items:")
        for item in items:
            print(f"{item.name}: weight={item.weight}, value={item.value}")
        print(f"Capacity: {capacity}")
        
        # Test basic DP approach
        max_value, result1 = knapsack_01_dp(items, capacity)
        print("\nBasic DP Approach:")
        print(f"Selected items: {[item.name for item in result1]}")
        print(f"Total value: {max_value}")
        print(f"Total weight: {sum(item.weight for item in result1)}")
        
        # Test optimized DP approach
        max_value, result2 = knapsack_01_optimized(items, capacity)
        print("\nOptimized DP Approach:")
        print(f"Selected items: {[item.name for item in result2]}")
        print(f"Total value: {max_value}")
        print(f"Total weight: {sum(item.weight for item in result2)}")
        
        # Test with constraints
        max_value, result3 = knapsack_01_with_constraints(items, capacity, constraints)
        print("\nWith Constraints:")
        print(f"Constraints: {constraints}")
        print(f"Selected items: {[item.name for item in result3]}")
        print(f"Total value: {max_value}")
        print(f"Total weight: {sum(item.weight for item in result3)}")
        
        # Print explanation
        print("\nExplanation:")
        print("1. Basic DP Approach:")
        print("   - Use 2D DP table")
        print("   - O(n * capacity) time complexity")
        print("   - O(n * capacity) space complexity")
        print("2. Optimized DP Approach:")
        print("   - Use 1D DP array")
        print("   - Same time complexity")
        print("   - O(capacity) space complexity")
        print("3. With Constraints:")
        print("   - Handle limits on item selection")
        print("   - O(n * capacity * max_constraint) time complexity")
        print("   - O(n * capacity * max_constraint) space complexity")
        print()
    
    # Run tests
    test_knapsack_01() 