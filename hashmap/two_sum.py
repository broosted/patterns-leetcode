from typing import List, Optional, Dict, Tuple

def two_sum_hashmap(nums: List[int], target: int) -> Optional[Tuple[int, int]]:
    """
    Find two numbers in the array that add up to target using a hashmap.
    
    Example:
    Input: nums = [2,7,11,15], target = 9
    Output: (0, 1)
    Explanation: Because nums[0] + nums[1] == 9, we return (0, 1).
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    seen: Dict[int, int] = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    
    return None

def two_sum_two_pointers(nums: List[int], target: int) -> Optional[Tuple[int, int]]:
    """
    Find two numbers in the sorted array that add up to target using two pointers.
    This approach requires the input array to be sorted.
    
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(n) to store original indices
    """
    # Create list of (number, original index) pairs
    nums_with_indices = [(num, i) for i, num in enumerate(nums)]
    nums_with_indices.sort()  # Sort by number
    
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums_with_indices[left][0] + nums_with_indices[right][0]
        
        if current_sum == target:
            return (nums_with_indices[left][1], nums_with_indices[right][1])
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return None

def two_sum_sorted(nums: List[int], target: int) -> Optional[Tuple[int, int]]:
    """
    Find two numbers in a sorted array that add up to target.
    This is a more efficient version of two_sum_two_pointers when the input
    is already sorted.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        
        if current_sum == target:
            return (left, right)
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return None

def two_sum_all_pairs(nums: List[int], target: int) -> List[Tuple[int, int]]:
    """
    Find all pairs of numbers in the array that add up to target.
    
    Example:
    Input: nums = [1,1,1,2,2,3,3,4,4,4], target = 5
    Output: [(0,5), (0,6), (1,5), (1,6), (2,5), (2,6), (3,7), (3,8), (4,7), (4,8)]
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Count occurrences of each number
    count: Dict[int, List[int]] = {}
    for i, num in enumerate(nums):
        if num not in count:
            count[num] = []
        count[num].append(i)
    
    result = []
    seen = set()  # To avoid duplicate pairs
    
    for i, num in enumerate(nums):
        complement = target - num
        
        # Skip if we've already processed this pair
        if (num, complement) in seen or (complement, num) in seen:
            continue
        
        if complement in count:
            # Add all valid pairs
            for j in count[complement]:
                if i != j:  # Avoid using same index twice
                    result.append((i, j))
            
            # Mark this pair as processed
            seen.add((num, complement))
    
    return result

# Example usage
if __name__ == "__main__":
    # Test cases for two_sum
    test_cases = [
        ([2,7,11,15], 9),      # Example from problem
        ([3,2,4], 6),          # Another example
        ([3,3], 6),            # Same numbers
        ([1,2,3,4,5], 9),      # Numbers at ends
        ([1,2,3,4,5], 10),     # No solution
        ([], 0),               # Empty array
    ]
    
    print("Testing two_sum:")
    for nums, target in test_cases:
        print(f"\nInput: nums = {nums}, target = {target}")
        
        # Test hashmap solution
        result1 = two_sum_hashmap(nums, target)
        print(f"Hashmap solution: {result1}")
        
        # Test two pointers solution (if array is not empty)
        if nums:
            sorted_nums = sorted(nums)
            result2 = two_sum_two_pointers(nums, target)
            result3 = two_sum_sorted(sorted_nums, target)
            print(f"Two pointers solution: {result2}")
            print(f"Sorted array solution: {result3}")
            
            # Verify solutions
            if result1:
                print(f"Hashmap solution valid: {nums[result1[0]] + nums[result1[1]] == target}")
            if result2:
                print(f"Two pointers solution valid: {nums[result2[0]] + nums[result2[1]] == target}")
            if result3:
                print(f"Sorted array solution valid: {sorted_nums[result3[0]] + sorted_nums[result3[1]] == target}")
        
        # Print explanation
        if nums:
            print("\nExplanation:")
            if result1:
                print(f"Found pair at indices {result1}: {nums[result1[0]]} + {nums[result1[1]]} = {target}")
            else:
                print("No solution found")
            print("Approaches:")
            print("1. Hashmap: O(n) time, O(n) space")
            print("2. Two pointers: O(n log n) time, O(n) space")
            print("3. Sorted array: O(n) time, O(1) space")
        print()
    
    # Test cases for two_sum_all_pairs
    test_cases_all = [
        ([1,1,1,2,2,3,3,4,4,4], 5),  # Example with multiple pairs
        ([1,2,3,4,5], 6),            # Multiple pairs
        ([1,1,1,1], 2),              # All same numbers
        ([1,2,3,4,5], 10),           # No solution
        ([], 0),                     # Empty array
    ]
    
    print("\nTesting two_sum_all_pairs:")
    for nums, target in test_cases_all:
        print(f"\nInput: nums = {nums}, target = {target}")
        result = two_sum_all_pairs(nums, target)
        print(f"All pairs: {result}")
        
        # Print explanation
        if nums:
            print("\nExplanation:")
            if result:
                print(f"Found {len(result)} pairs that sum to {target}:")
                for i, j in result:
                    print(f"  {nums[i]} + {nums[j]} = {target}")
            else:
                print("No pairs found")
            print("Using hashmap approach:")
            print("1. Count occurrences of each number")
            print("2. For each number, find its complement")
            print("3. Generate all valid pairs")
            print("4. Avoid duplicate pairs")
        print() 