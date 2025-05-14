from typing import List

class NumArray:
    """
    Given an integer array nums, handle multiple queries of the following type:
    Calculate the sum of the elements of nums between indices left and right inclusive.
    
    Example:
    Input: nums = [-2, 0, 3, -5, 2, -1]
    NumArray numArray = new NumArray(nums);
    numArray.sumRange(0, 2) -> 1
    numArray.sumRange(2, 5) -> -1
    numArray.sumRange(0, 5) -> -3
    
    Time Complexity:
    - Initialization: O(n)
    - Query: O(1)
    Space Complexity: O(n)
    """
    def __init__(self, nums: List[int]):
        # Create prefix sum array
        self.prefix_sum = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            self.prefix_sum[i + 1] = self.prefix_sum[i] + nums[i]
    
    def sumRange(self, left: int, right: int) -> int:
        """
        Calculate the sum of elements between indices left and right inclusive.
        """
        return self.prefix_sum[right + 1] - self.prefix_sum[left]

# Example usage
if __name__ == "__main__":
    # Test cases
    nums = [-2, 0, 3, -5, 2, -1]
    numArray = NumArray(nums)
    
    test_ranges = [
        (0, 2),  # Expected: 1
        (2, 5),  # Expected: -1
        (0, 5),  # Expected: -3
        (1, 3),  # Expected: -2
    ]
    
    for left, right in test_ranges:
        result = numArray.sumRange(left, right)
        print(f"Range [{left}, {right}]: {result}")
        print(f"Explanation: Sum of {nums[left:right+1]} = {result}\n") 