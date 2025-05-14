from typing import List
from collections import Counter
import heapq

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """
    Given an integer array nums and an integer k, return the k most frequent elements.
    You may return the answer in any order.
    
    Example:
    Input: nums = [1,1,1,2,2,3], k = 2
    Output: [1,2]
    Explanation: 1 appears 3 times, 2 appears 2 times, 3 appears 1 time.
    The two most frequent elements are 1 and 2.
    
    Time Complexity: O(n log k) where n is the length of nums
    Space Complexity: O(n)
    """
    # Count frequencies
    count = Counter(nums)
    
    # Use a min heap to keep track of k most frequent elements
    # Store (-frequency, number) to simulate a max heap
    heap = []
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    # Extract numbers from heap
    return [num for _, num in heap]

def top_k_frequent_bucket_sort(nums: List[int], k: int) -> List[int]:
    """
    Alternative solution using bucket sort.
    This approach is more efficient when k is close to n.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Count frequencies
    count = Counter(nums)
    
    # Create buckets where index represents frequency
    # The maximum frequency possible is len(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)
    
    # Collect k most frequent elements
    result = []
    for freq in range(len(nums), 0, -1):
        result.extend(buckets[freq])
        if len(result) >= k:
            return result[:k]
    
    return result

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ([1, 1, 1, 2, 2, 3], 2),      # Expected: [1, 2]
        ([1], 1),                     # Expected: [1]
        ([1, 2, 3, 4, 5], 3),         # Expected: [1, 2, 3] (or any 3 numbers)
        ([1, 1, 2, 2, 3, 3], 2),      # Expected: [1, 2] or [1, 3] or [2, 3]
    ]
    
    for nums, k in test_cases:
        print(f"Input: nums = {nums}, k = {k}")
        
        # Test heap solution
        result1 = top_k_frequent(nums, k)
        print(f"Heap solution: {result1}")
        
        # Test bucket sort solution
        result2 = top_k_frequent_bucket_sort(nums, k)
        print(f"Bucket sort solution: {result2}")
        
        # Print frequency counts for verification
        count = Counter(nums)
        print("Frequency counts:", dict(count))
        print() 