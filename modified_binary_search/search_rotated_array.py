from typing import List

def search_rotated_array(nums: List[int], target: int) -> int:
    """
    Given the array nums after the possible rotation and an integer target,
    return the index of target if it is in nums, or -1 if it is not in nums.
    
    Example:
    Input: nums = [4,5,6,7,0,1,2], target = 0
    Output: 4
    Explanation: The array was rotated at index 3, and target 0 is at index 4.
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    if not nums:
        return -1
    
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return mid
        
        # Check if left half is sorted
        if nums[left] <= nums[mid]:
            # Check if target is in left half
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half must be sorted
        else:
            # Check if target is in right half
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

def search_rotated_array_with_duplicates(nums: List[int], target: int) -> bool:
    """
    Given the array nums after the possible rotation and an integer target,
    return true if target is in nums, or false if it is not in nums.
    This version handles arrays with duplicate elements.
    
    Example:
    Input: nums = [2,5,6,0,0,1,2], target = 0
    Output: true
    Explanation: The array was rotated at index 3, and target 0 is at indices 3 and 4.
    
    Time Complexity: O(n) in worst case (when all elements are same)
    Space Complexity: O(1)
    """
    if not nums:
        return False
    
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if nums[mid] == target:
            return True
        
        # Handle duplicates
        if nums[left] == nums[mid] == nums[right]:
            left += 1
            right -= 1
            continue
        
        # Check if left half is sorted
        if nums[left] <= nums[mid]:
            # Check if target is in left half
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half must be sorted
        else:
            # Check if target is in right half
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return False

def find_rotation_point(nums: List[int]) -> int:
    """
    Find the index where the array was rotated.
    Returns 0 if the array is not rotated.
    
    Example:
    Input: nums = [4,5,6,7,0,1,2]
    Output: 4
    Explanation: The array was rotated at index 4 (value 0).
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    if not nums or nums[0] <= nums[-1]:
        return 0
    
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if nums[mid] > nums[mid + 1]:
            return mid + 1
        
        if nums[left] <= nums[mid]:
            left = mid + 1
        else:
            right = mid
    
    return left

# Example usage
if __name__ == "__main__":
    # Test cases for search_rotated_array
    test_cases = [
        ([4, 5, 6, 7, 0, 1, 2], 0),    # Example from problem
        ([4, 5, 6, 7, 0, 1, 2], 3),    # Target not in array
        ([1], 0),                      # Single element
        ([1, 3], 3),                   # Two elements
        ([1, 3], 1),                   # Two elements
        ([3, 1], 1),                   # Rotated two elements
        ([3, 1], 3),                   # Rotated two elements
    ]
    
    print("Testing search_rotated_array:")
    for nums, target in test_cases:
        result = search_rotated_array(nums, target)
        print(f"\nArray: {nums}")
        print(f"Target: {target}")
        print(f"Found at index: {result}")
        if result != -1:
            print(f"Verification: nums[{result}] = {nums[result]}")
    
    # Test cases for search_rotated_array_with_duplicates
    test_cases_dup = [
        ([2, 5, 6, 0, 0, 1, 2], 0),    # Example from problem
        ([2, 5, 6, 0, 0, 1, 2], 3),    # Target not in array
        ([1, 1, 1, 1, 1], 1),          # All same elements
        ([1, 1, 1, 1, 1], 2),          # Target not in array
    ]
    
    print("\nTesting search_rotated_array_with_duplicates:")
    for nums, target in test_cases_dup:
        result = search_rotated_array_with_duplicates(nums, target)
        print(f"\nArray: {nums}")
        print(f"Target: {target}")
        print(f"Found: {result}")
    
    # Test cases for find_rotation_point
    test_cases_rot = [
        [4, 5, 6, 7, 0, 1, 2],         # Example from problem
        [0, 1, 2, 4, 5, 6, 7],         # Not rotated
        [1],                            # Single element
        [1, 3],                         # Not rotated
        [3, 1],                         # Rotated
        [2, 2, 2, 0, 2, 2],            # With duplicates
    ]
    
    print("\nTesting find_rotation_point:")
    for nums in test_cases_rot:
        result = find_rotation_point(nums)
        print(f"\nArray: {nums}")
        print(f"Rotation point index: {result}")
        if result < len(nums):
            print(f"Value at rotation point: {nums[result]}")
        print(f"Array is {'not ' if result == 0 else ''}rotated") 