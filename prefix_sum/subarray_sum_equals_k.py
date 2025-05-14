from typing import List

def subarray_sum_equals_k(nums: List[int], k: int) -> int:
    """
    Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.
    
    Example:
    Input: nums = [1,1,1], k = 2
    Output: 2
    Explanation: The subarrays [1,1] and [1,1] sum to 2.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    count = 0
    prefix_sum = 0
    # Dictionary to store prefix sums and their frequencies
    prefix_sum_count = {0: 1}  # Initialize with 0:1 because empty subarray has sum 0
    
    for num in nums:
        prefix_sum += num
        
        # If (prefix_sum - k) exists in our map, we found a subarray
        if prefix_sum - k in prefix_sum_count:
            count += prefix_sum_count[prefix_sum - k]
        
        # Update the frequency of current prefix sum
        prefix_sum_count[prefix_sum] = prefix_sum_count.get(prefix_sum, 0) + 1
    
    return count

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ([1, 1, 1], 2),           # Output: 2
        ([1, 2, 3], 3),           # Output: 2
        ([1, -1, 0], 0),          # Output: 3
        ([3, 4, 7, 2, -3, 1, 4, 2], 7)  # Output: 4
    ]
    
    for nums, k in test_cases:
        result = subarray_sum_equals_k(nums, k)
        print(f"Array: {nums}, k: {k}")
        print(f"Number of subarrays with sum {k}: {result}\n") 