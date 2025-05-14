from typing import List, Optional

def find_duplicate_cyclic_sort(nums: List[int]) -> int:
    """
    Find the duplicate number in an array containing n + 1 integers in range [1, n].
    Using cyclic sort approach.
    
    Example:
    Input: nums = [1,3,4,2,2]
    Output: 2
    Explanation: 2 is the duplicate number since it appears twice.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    i = 0
    n = len(nums)
    
    # Cyclic sort
    while i < n:
        # If current number is in correct position, move to next
        if nums[i] == i + 1:
            i += 1
        else:
            # Get correct position for current number
            correct_pos = nums[i] - 1
            
            # If number at correct position is same as current number,
            # we found our duplicate
            if nums[i] == nums[correct_pos]:
                return nums[i]
            
            # Swap current number to its correct position
            nums[i], nums[correct_pos] = nums[correct_pos], nums[i]

def find_duplicate_floyd(nums: List[int]) -> int:
    """
    Find the duplicate number using Floyd's Cycle Finding algorithm.
    This approach treats the array as a linked list where each number
    points to the index equal to its value.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # Find the intersection point of the two runners
    tortoise = hare = nums[0]
    while True:
        tortoise = nums[tortoise]
        hare = nums[nums[hare]]
        if tortoise == hare:
            break
    
    # Find the entrance to the cycle
    tortoise = nums[0]
    while tortoise != hare:
        tortoise = nums[tortoise]
        hare = nums[hare]
    
    return hare

def find_duplicate_binary_search(nums: List[int]) -> int:
    """
    Find the duplicate number using binary search on the range [1, n].
    This approach counts numbers less than or equal to mid.
    
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    """
    n = len(nums) - 1
    left, right = 1, n
    
    while left < right:
        mid = (left + right) // 2
        count = 0
        
        # Count numbers less than or equal to mid
        for num in nums:
            if num <= mid:
                count += 1
        
        # If count > mid, duplicate is in [left, mid]
        if count > mid:
            right = mid
        else:
            left = mid + 1
    
    return left

def find_all_duplicates(nums: List[int]) -> List[int]:
    """
    Find all duplicate numbers in an array containing n integers in range [1, n].
    
    Example:
    Input: nums = [4,3,2,7,8,2,3,1]
    Output: [2,3]
    Explanation: 2 and 3 appear twice in the array.
    
    Time Complexity: O(n)
    Space Complexity: O(1) excluding output array
    """
    n = len(nums)
    i = 0
    
    # Cyclic sort
    while i < n:
        # If current number is in correct position, move to next
        if nums[i] == i + 1 or nums[i] > n or nums[i] < 1:
            i += 1
        else:
            # Get correct position for current number
            correct_pos = nums[i] - 1
            
            # If number at correct position is same as current number,
            # we found a duplicate
            if nums[i] == nums[correct_pos]:
                i += 1
            else:
                # Swap current number to its correct position
                nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
    
    # Find all duplicates
    duplicates = []
    for i in range(n):
        if nums[i] != i + 1:
            duplicates.append(nums[i])
    
    return duplicates

# Example usage
if __name__ == "__main__":
    # Test cases for find_duplicate
    test_cases = [
        [1,3,4,2,2],        # Example from problem
        [3,1,3,4,2],        # Another example
        [1,1],              # Simple case
        [1,1,2],            # Duplicate at start
        [1,2,2],            # Duplicate at end
        [2,2,2,2,2],        # All same numbers
    ]
    
    print("Testing find_duplicate:")
    for nums in test_cases:
        print(f"\nInput: nums = {nums}")
        
        # Test all solutions
        result1 = find_duplicate_cyclic_sort(nums.copy())
        result2 = find_duplicate_floyd(nums.copy())
        result3 = find_duplicate_binary_search(nums.copy())
        
        print(f"Cyclic sort: {result1}")
        print(f"Floyd's algorithm: {result2}")
        print(f"Binary search: {result3}")
        
        # Verify all solutions give same result
        print(f"All solutions match: {result1 == result2 == result3}")
        
        # Print explanation
        if nums:
            print("\nExplanation:")
            print(f"The duplicate number is: {result1}")
            print("Approaches:")
            print("1. Cyclic sort: O(n) time, O(1) space")
            print("2. Floyd's algorithm: O(n) time, O(1) space")
            print("3. Binary search: O(n log n) time, O(1) space")
            print(f"Array length: {len(nums)}")
            print(f"Expected range: [1, {len(nums)-1}]")
        print()
    
    # Test cases for find_all_duplicates
    test_cases_all = [
        [4,3,2,7,8,2,3,1],   # Example from problem
        [1,1,2,2],           # Two duplicates
        [1,2,3,4],           # No duplicates
        [1,1,1],             # All same numbers
        [],                  # Empty array
    ]
    
    print("\nTesting find_all_duplicates:")
    for nums in test_cases_all:
        print(f"\nInput: nums = {nums}")
        result = find_all_duplicates(nums.copy())
        print(f"Duplicate numbers: {result}")
        
        # Print explanation
        if nums:
            print("\nExplanation:")
            print(f"Found {len(result)} duplicate numbers: {result}")
            print("Using cyclic sort approach:")
            print("1. Sort numbers to their correct positions")
            print("2. Numbers not in their correct positions are duplicates")
            print(f"Array length: {len(nums)}")
            print(f"Expected range: [1, {len(nums)}]")
        print() 