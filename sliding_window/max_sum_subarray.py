from typing import List

def max_sum_subarray(nums: List[int], k: int) -> int:
    """
    Given an array of integers nums and an integer k, find the maximum sum of any contiguous subarray of size k.
    
    Example:
    Input: nums = [2, 1, 5, 1, 3, 2], k = 3
    Output: 9
    Explanation: Subarray with maximum sum is [5, 1, 3] with sum 9.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not nums or k <= 0 or k > len(nums):
        return 0
    
    # Calculate sum of first window
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide the window and update max_sum
    for i in range(len(nums) - k):
        # Subtract element going out of window and add element coming into window
        window_sum = window_sum - nums[i] + nums[i + k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ([2, 1, 5, 1, 3, 2], 3),      # Expected: 9
        ([1, 4, 2, 10, 2, 3, 1, 0, 20], 4),  # Expected: 24
        ([1, 2, 3, 4, 5], 2),         # Expected: 9
        ([1, 1, 1, 1, 1], 3),         # Expected: 3
    ]
    
    for nums, k in test_cases:
        result = max_sum_subarray(nums, k)
        print(f"Array: {nums}, k: {k}")
        print(f"Maximum sum of subarray of size {k}: {result}")
        
        # Find and show the subarray
        max_sum = 0
        max_subarray = []
        for i in range(len(nums) - k + 1):
            current_sum = sum(nums[i:i+k])
            if current_sum > max_sum:
                max_sum = current_sum
                max_subarray = nums[i:i+k]
        
        print(f"Subarray with maximum sum: {max_subarray}\n") 