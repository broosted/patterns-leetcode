from typing import List, Set, Tuple
from collections import Counter

def subsets_with_duplicates_backtrack(nums: List[int]) -> List[List[int]]:
    """
    Generate all subsets of an array that may contain duplicates using backtracking.
    
    Example:
    Input: nums = [1, 2, 2]
    Output: [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]
    
    Time Complexity: O(n * 2^n)
    Space Complexity: O(n) for recursion stack
    """
    def backtrack(start: int, path: List[int]) -> None:
        """Backtracking helper function."""
        result.append(path[:])
        
        for i in range(start, len(nums)):
            # Skip duplicates
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    
    # Sort the array to handle duplicates
    nums.sort()
    result = []
    backtrack(0, [])
    return result

def subsets_with_duplicates_iterative(nums: List[int]) -> List[List[int]]:
    """
    Generate all subsets of an array that may contain duplicates using an iterative approach.
    
    Time Complexity: O(n * 2^n)
    Space Complexity: O(1) excluding the output
    """
    # Sort the array to handle duplicates
    nums.sort()
    result = [[]]
    start, end = 0, 0
    
    for i, num in enumerate(nums):
        start = 0
        # If current number is same as previous, start from where we left
        if i > 0 and nums[i] == nums[i - 1]:
            start = end
        end = len(result)
        
        # Add current number to all existing subsets
        for j in range(start, end):
            result.append(result[j] + [num])
    
    return result

def subsets_with_duplicates_counter(nums: List[int]) -> List[List[int]]:
    """
    Generate all subsets of an array that may contain duplicates using a counter.
    
    Time Complexity: O(n * 2^n)
    Space Complexity: O(n) for the counter
    """
    def generate_subsets(counts: List[Tuple[int, int]], index: int, path: List[int]) -> None:
        """Generate subsets using frequency counts."""
        if index == len(counts):
            result.append(path[:])
            return
        
        num, freq = counts[index]
        # Try including 0 to freq occurrences of current number
        for i in range(freq + 1):
            path.extend([num] * i)
            generate_subsets(counts, index + 1, path)
            path[-i:] = [] if i > 0 else []
    
    # Count frequencies
    counts = list(Counter(nums).items())
    result = []
    generate_subsets(counts, 0, [])
    return result

def subsets_with_duplicates_bitmask(nums: List[int]) -> List[List[int]]:
    """
    Generate all subsets of an array that may contain duplicates using bit manipulation.
    
    Time Complexity: O(n * 2^n)
    Space Complexity: O(1) excluding the output
    """
    # Sort the array to handle duplicates
    nums.sort()
    n = len(nums)
    result = []
    
    # Generate all possible bitmasks
    for mask in range(1 << n):
        subset = []
        # Check each bit
        for i in range(n):
            if mask & (1 << i):
                # Skip if this number is a duplicate and we haven't included its previous occurrence
                if i > 0 and nums[i] == nums[i - 1] and not (mask & (1 << (i - 1))):
                    break
                subset.append(nums[i])
        else:  # Only add if we didn't break
            result.append(subset)
    
    return result

# Example usage
if __name__ == "__main__":
    def test_subsets_with_duplicates():
        """Test the subsets with duplicates implementations."""
        # Test cases
        test_cases = [
            ([1, 2, 2], "Standard case with duplicates"),
            ([1, 1, 2, 2], "Multiple duplicates"),
            ([1, 1, 1], "All same numbers"),
            ([1, 2, 3], "No duplicates"),
            ([], "Empty array"),
            ([1], "Single element"),
        ]
        
        for nums, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"nums = {nums}")
            
            # Test backtracking approach
            result1 = subsets_with_duplicates_backtrack(nums)
            print("\nBacktracking Approach:")
            print(f"Number of subsets: {len(result1)}")
            print(f"Subsets: {result1}")
            
            # Test iterative approach
            result2 = subsets_with_duplicates_iterative(nums)
            print("\nIterative Approach:")
            print(f"Number of subsets: {len(result2)}")
            print(f"Subsets: {result2}")
            
            # Test counter approach
            result3 = subsets_with_duplicates_counter(nums)
            print("\nCounter Approach:")
            print(f"Number of subsets: {len(result3)}")
            print(f"Subsets: {result3}")
            
            # Test bitmask approach
            result4 = subsets_with_duplicates_bitmask(nums)
            print("\nBitmask Approach:")
            print(f"Number of subsets: {len(result4)}")
            print(f"Subsets: {result4}")
            
            # Verify all results match
            all_results = [result1, result2, result3, result4]
            all_match = all(sorted(r) == sorted(result1) for r in all_results[1:])
            print(f"\nAll results match: {all_match}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. Backtracking Approach:")
            print("   - Uses recursive backtracking")
            print("   - Skips duplicates by checking previous number")
            print("   - O(n * 2^n) time complexity")
            print("2. Iterative Approach:")
            print("   - Builds subsets incrementally")
            print("   - Handles duplicates by tracking start position")
            print("   - Same time complexity but no recursion")
            print("3. Counter Approach:")
            print("   - Uses frequency counts to handle duplicates")
            print("   - More intuitive for handling multiple occurrences")
            print("   - Same time complexity")
            print("4. Bitmask Approach:")
            print("   - Uses bit manipulation to generate subsets")
            print("   - Handles duplicates by checking previous bits")
            print("   - Same time complexity but different implementation")
            print()
    
    # Run tests
    test_subsets_with_duplicates() 