import heapq
from typing import List

class KthLargest:
    """
    Design a class to find the kth largest element in a stream.
    Note that it is the kth largest element in the sorted order, not the kth distinct element.
    
    Example:
    Input: ["KthLargest", "add", "add", "add", "add", "add"]
           [[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
    Output: [null, 4, 5, 5, 8, 8]
    Explanation:
    KthLargest kthLargest = new KthLargest(3, [4, 5, 8, 2]);
    kthLargest.add(3);   // return 4
    kthLargest.add(5);   // return 5
    kthLargest.add(10);  // return 5
    kthLargest.add(9);   // return 8
    kthLargest.add(4);   // return 8
    
    Time Complexity:
    - __init__: O(n log k) where n is the length of nums
    - add: O(log k)
    Space Complexity: O(k)
    """
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = []
        
        # Initialize the min heap with first k elements
        for num in nums:
            self.add(num)
    
    def add(self, val: int) -> int:
        """
        Add a new element to the stream and return the kth largest element.
        """
        # Add new element to heap
        heapq.heappush(self.heap, val)
        
        # Keep only k largest elements
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        
        # Return the kth largest element (smallest in the min heap)
        return self.heap[0]

class KthLargestAlternative:
    """
    Alternative implementation using a sorted list.
    This approach might be more efficient for small k and frequent additions.
    
    Time Complexity:
    - __init__: O(n log n) where n is the length of nums
    - add: O(log n)
    Space Complexity: O(n)
    """
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.nums = sorted(nums)
    
    def add(self, val: int) -> int:
        """
        Add a new element to the stream and return the kth largest element.
        """
        # Find insertion point using binary search
        left, right = 0, len(self.nums)
        while left < right:
            mid = (left + right) // 2
            if self.nums[mid] < val:
                left = mid + 1
            else:
                right = mid
        
        # Insert the new value
        self.nums.insert(left, val)
        
        # Return the kth largest element
        return self.nums[-self.k]

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        (3, [4, 5, 8, 2], [3, 5, 10, 9, 4]),  # Example from problem description
        (1, [1, 2, 3, 4, 5], [6, 7, 8]),       # k=1 (always return largest)
        (2, [1], [2, 3, 4, 5]),                # Growing stream
    ]
    
    for k, initial_nums, additions in test_cases:
        print(f"\nTesting with k={k}, initial_nums={initial_nums}")
        
        # Test heap-based solution
        kth_largest = KthLargest(k, initial_nums)
        print("Heap-based solution:")
        print("Initial kth largest:", kth_largest.heap[0] if kth_largest.heap else None)
        for val in additions:
            result = kth_largest.add(val)
            print(f"After adding {val}: {result}")
        
        # Test alternative solution
        kth_largest_alt = KthLargestAlternative(k, initial_nums)
        print("\nAlternative solution:")
        print("Initial kth largest:", kth_largest_alt.nums[-k] if len(kth_largest_alt.nums) >= k else None)
        for val in additions:
            result = kth_largest_alt.add(val)
            print(f"After adding {val}: {result}")
        print() 