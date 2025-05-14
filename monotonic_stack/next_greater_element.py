from typing import List

def next_greater_element(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Given two arrays nums1 and nums2, where nums1 is a subset of nums2,
    find the next greater element for each element in nums1 in nums2.
    
    Example:
    Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
    Output: [-1,3,-1]
    Explanation: The next greater element for each value of nums1 is as follows:
    - 4 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
    - 1 is underlined in nums2 = [1,3,4,2]. The next greater element is 3.
    - 2 is underlined in nums2 = [1,3,4,2]. There is no next greater element, so the answer is -1.
    
    Time Complexity: O(n + m) where n and m are lengths of nums1 and nums2
    Space Complexity: O(n + m)
    """
    # Create a mapping of each number to its next greater element
    next_greater = {}
    stack = []
    
    # Process nums2 from right to left
    for num in reversed(nums2):
        # Pop all elements smaller than current number
        while stack and stack[-1] <= num:
            stack.pop()
        
        # If stack is empty, no greater element exists
        next_greater[num] = stack[-1] if stack else -1
        
        # Push current number to stack
        stack.append(num)
    
    # Get next greater element for each number in nums1
    return [next_greater[num] for num in nums1]

def next_greater_element_circular(nums: List[int]) -> List[int]:
    """
    Given a circular array nums, find the next greater number for every element in nums.
    The next greater number of a number x is the first greater number to its traversing-order next in the array.
    
    Example:
    Input: nums = [1,2,1]
    Output: [2,-1,2]
    Explanation: The first 1's next greater number is 2;
    The number 2 can't find next greater number;
    The second 1's next greater number needs to search circularly, which is also 2.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    n = len(nums)
    result = [-1] * n
    stack = []
    
    # Process array twice to handle circular nature
    for i in range(2 * n):
        # Get the actual index in the original array
        idx = i % n
        
        # Pop all elements smaller than current number
        while stack and nums[stack[-1]] < nums[idx]:
            result[stack.pop()] = nums[idx]
        
        # Only push indices from first iteration
        if i < n:
            stack.append(idx)
    
    return result

# Example usage
if __name__ == "__main__":
    # Test next_greater_element
    print("Testing next_greater_element:")
    test_cases = [
        ([4, 1, 2], [1, 3, 4, 2]),     # Expected: [-1, 3, -1]
        ([2, 4], [1, 2, 3, 4]),        # Expected: [3, -1]
        ([1, 3, 5, 2, 4], [6, 5, 4, 3, 2, 1, 7])  # Expected: [7, 7, 7, 7, 7]
    ]
    
    for nums1, nums2 in test_cases:
        result = next_greater_element(nums1, nums2)
        print(f"nums1: {nums1}")
        print(f"nums2: {nums2}")
        print(f"Result: {result}")
        print()
    
    # Test next_greater_element_circular
    print("\nTesting next_greater_element_circular:")
    test_cases = [
        [1, 2, 1],           # Expected: [2, -1, 2]
        [1, 2, 3, 4, 3],     # Expected: [2, 3, 4, -1, 4]
        [5, 4, 3, 2, 1],     # Expected: [-1, 5, 5, 5, 5]
    ]
    
    for nums in test_cases:
        result = next_greater_element_circular(nums)
        print(f"Input: {nums}")
        print(f"Result: {result}")
        print() 