from typing import List, Set, Tuple
from collections import Counter

def subsets_with_sum_backtrack(nums: List[int], target: int) -> List[List[int]]:
    """
    Find all subsets that sum to the target value using backtracking.
    
    Example:
    Input: nums = [1, 2, 3, 4], target = 5
    Output: [[1, 4], [2, 3]]
    
    Time Complexity: O(2^n)
    Space Complexity: O(n) for recursion stack
    """
    def backtrack(start: int, path: List[int], curr_sum: int) -> None:
        """Backtracking helper function."""
        if curr_sum == target:
            result.append(path[:])
            return
        if curr_sum > target:
            return
        
        for i in range(start, len(nums)):
            # Skip duplicates
            if i > start and nums[i] == nums[i - 1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path, curr_sum + nums[i])
            path.pop()
    
    # Sort the array to handle duplicates
    nums.sort()
    result = []
    backtrack(0, [], 0)
    return result

def subsets_with_sum_dp(nums: List[int], target: int) -> List[List[int]]:
    """
    Find all subsets that sum to the target value using dynamic programming.
    
    Time Complexity: O(n * target)
    Space Complexity: O(n * target)
    """
    # Initialize DP table
    dp = [[[] for _ in range(target + 1)] for _ in range(len(nums) + 1)]
    dp[0][0] = [[]]  # Empty subset sums to 0
    
    # Fill DP table
    for i in range(1, len(nums) + 1):
        for j in range(target + 1):
            # Copy previous solutions
            dp[i][j] = dp[i - 1][j][:]
            
            # Add current number if possible
            if j >= nums[i - 1]:
                for subset in dp[i - 1][j - nums[i - 1]]:
                    dp[i][j].append(subset + [nums[i - 1]])
    
    return dp[len(nums)][target]

def subsets_with_sum_counter(nums: List[int], target: int) -> List[List[int]]:
    """
    Find all subsets that sum to the target value using a counter.
    
    Time Complexity: O(2^n)
    Space Complexity: O(n) for the counter
    """
    def generate_subsets(counts: List[Tuple[int, int]], index: int, 
                        path: List[int], curr_sum: int) -> None:
        """Generate subsets using frequency counts."""
        if curr_sum == target:
            result.append(path[:])
            return
        if curr_sum > target or index == len(counts):
            return
        
        num, freq = counts[index]
        # Try including 0 to freq occurrences of current number
        for i in range(freq + 1):
            if curr_sum + num * i > target:
                break
            path.extend([num] * i)
            generate_subsets(counts, index + 1, path, curr_sum + num * i)
            path[-i:] = [] if i > 0 else []
    
    # Count frequencies
    counts = list(Counter(nums).items())
    result = []
    generate_subsets(counts, 0, [], 0)
    return result

def subsets_with_sum_bitmask(nums: List[int], target: int) -> List[List[int]]:
    """
    Find all subsets that sum to the target value using bit manipulation.
    
    Time Complexity: O(2^n)
    Space Complexity: O(1) excluding the output
    """
    n = len(nums)
    result = []
    
    # Generate all possible bitmasks
    for mask in range(1 << n):
        subset = []
        curr_sum = 0
        
        # Check each bit
        for i in range(n):
            if mask & (1 << i):
                # Skip if this number is a duplicate and we haven't included its previous occurrence
                if i > 0 and nums[i] == nums[i - 1] and not (mask & (1 << (i - 1))):
                    break
                subset.append(nums[i])
                curr_sum += nums[i]
                if curr_sum > target:
                    break
        else:  # Only add if we didn't break and sum matches target
            if curr_sum == target:
                result.append(subset)
    
    return result

def count_subsets_with_sum(nums: List[int], target: int) -> int:
    """
    Count the number of subsets that sum to the target value using dynamic programming.
    
    Time Complexity: O(n * target)
    Space Complexity: O(target)
    """
    # Initialize DP array
    dp = [0] * (target + 1)
    dp[0] = 1  # Empty subset sums to 0
    
    # Fill DP array
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] += dp[j - num]
    
    return dp[target]

# Example usage
if __name__ == "__main__":
    def test_subsets_with_sum():
        """Test the subsets with sum implementations."""
        # Test cases
        test_cases = [
            ([1, 2, 3, 4], 5, "Standard case"),
            ([1, 1, 2, 2], 3, "With duplicates"),
            ([1, 1, 1], 2, "All same numbers"),
            ([1, 2, 3], 0, "Target is zero"),
            ([], 5, "Empty array"),
            ([1], 1, "Single element"),
        ]
        
        for nums, target, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"nums = {nums}")
            print(f"target = {target}")
            
            # Test backtracking approach
            result1 = subsets_with_sum_backtrack(nums, target)
            print("\nBacktracking Approach:")
            print(f"Number of subsets: {len(result1)}")
            print(f"Subsets: {result1}")
            
            # Test DP approach
            result2 = subsets_with_sum_dp(nums, target)
            print("\nDP Approach:")
            print(f"Number of subsets: {len(result2)}")
            print(f"Subsets: {result2}")
            
            # Test counter approach
            result3 = subsets_with_sum_counter(nums, target)
            print("\nCounter Approach:")
            print(f"Number of subsets: {len(result3)}")
            print(f"Subsets: {result3}")
            
            # Test bitmask approach
            result4 = subsets_with_sum_bitmask(nums, target)
            print("\nBitmask Approach:")
            print(f"Number of subsets: {len(result4)}")
            print(f"Subsets: {result4}")
            
            # Test count only
            count = count_subsets_with_sum(nums, target)
            print("\nCount Only:")
            print(f"Number of subsets: {count}")
            
            # Verify all results match
            all_results = [result1, result2, result3, result4]
            all_match = all(sorted(r) == sorted(result1) for r in all_results[1:])
            print(f"\nAll results match: {all_match}")
            print(f"Count matches: {count == len(result1)}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. Backtracking Approach:")
            print("   - Uses recursive backtracking")
            print("   - Prunes branches when sum exceeds target")
            print("   - O(2^n) time complexity")
            print("2. DP Approach:")
            print("   - Uses dynamic programming")
            print("   - More efficient for small target values")
            print("   - O(n * target) time complexity")
            print("3. Counter Approach:")
            print("   - Uses frequency counts to handle duplicates")
            print("   - More intuitive for handling multiple occurrences")
            print("   - O(2^n) time complexity")
            print("4. Bitmask Approach:")
            print("   - Uses bit manipulation to generate subsets")
            print("   - Handles duplicates by checking previous bits")
            print("   - O(2^n) time complexity")
            print("5. Count Only:")
            print("   - Uses optimized DP to count subsets")
            print("   - O(n * target) time complexity")
            print("   - O(target) space complexity")
            print()
    
    # Run tests
    test_subsets_with_sum() 