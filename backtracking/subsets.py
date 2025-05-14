from typing import List

def subsets(nums: List[int]) -> List[List[int]]:
    """
    Generate all possible subsets of a set of distinct integers.
    
    Example:
    Input: nums = [1,2,3]
    Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
    
    Time Complexity: O(n * 2^n)
    Space Complexity: O(n * 2^n)
    """
    def backtrack(start: int, current: List[int]):
        # Add current subset to result
        result.append(current[:])
        
        # Try adding each number after start
        for i in range(start, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()  # Backtrack
    
    result = []
    backtrack(0, [])
    return result

def subsets_iterative(nums: List[int]) -> List[List[int]]:
    """
    Alternative solution using iterative approach.
    This approach builds subsets by adding each number to all existing subsets.
    
    Time Complexity: O(n * 2^n)
    Space Complexity: O(n * 2^n)
    """
    result = [[]]
    
    for num in nums:
        # Add current number to all existing subsets
        result.extend([subset + [num] for subset in result])
    
    return result

def subsets_with_duplicates(nums: List[int]) -> List[List[int]]:
    """
    Generate all possible subsets of a set that may contain duplicates.
    The solution should not contain duplicate subsets.
    
    Example:
    Input: nums = [1,2,2]
    Output: [[],[1],[2],[1,2],[2,2],[1,2,2]]
    
    Time Complexity: O(n * 2^n)
    Space Complexity: O(n * 2^n)
    """
    def backtrack(start: int, current: List[int]):
        result.append(current[:])
        
        for i in range(start, len(nums)):
            # Skip duplicates
            if i > start and nums[i] == nums[i-1]:
                continue
            
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    
    # Sort to handle duplicates
    nums.sort()
    result = []
    backtrack(0, [])
    return result

# Example usage
if __name__ == "__main__":
    # Test cases for distinct numbers
    test_cases = [
        [1, 2, 3],      # Example from problem
        [1, 2],         # Two numbers
        [1],            # Single number
        [],             # Empty set
    ]
    
    print("Testing subsets with distinct numbers:")
    for nums in test_cases:
        print(f"\nInput: nums = {nums}")
        
        # Test recursive solution
        result1 = subsets(nums)
        print(f"Recursive solution ({len(result1)} subsets):")
        for i, subset in enumerate(result1, 1):
            print(f"{i}. {subset}")
        
        # Test iterative solution
        result2 = subsets_iterative(nums)
        print(f"\nIterative solution ({len(result2)} subsets):")
        for i, subset in enumerate(result2, 1):
            print(f"{i}. {subset}")
        
        # Verify both solutions give same results
        print(f"\nSolutions match: {sorted(result1) == sorted(result2)}")
        
        # Print explanation
        if nums:
            print("\nExplanation:")
            print(f"For set {nums}, we generate all possible subsets.")
            print("Each number can be either included or excluded from a subset.")
            print(f"Total subsets: {len(result1)} (2^{len(nums)} = {2**len(nums)})")
        print()
    
    # Test cases for numbers with duplicates
    test_cases_dup = [
        [1, 2, 2],      # Example from problem
        [1, 1, 2, 2],   # Multiple duplicates
        [1, 1, 1],      # All same numbers
    ]
    
    print("\nTesting subsets with duplicates:")
    for nums in test_cases_dup:
        print(f"\nInput: nums = {nums}")
        
        result = subsets_with_duplicates(nums)
        print(f"Solution ({len(result)} subsets):")
        for i, subset in enumerate(result, 1):
            print(f"{i}. {subset}")
        
        # Print explanation
        print("\nExplanation:")
        print(f"For set {nums} with duplicates, we generate all unique subsets.")
        print("Duplicate numbers are handled by sorting and skipping repeated numbers.")
        print(f"Total unique subsets: {len(result)}")
        print() 