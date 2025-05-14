from typing import List, Optional
import heapq

class MedianFinder:
    """
    Find the median from a data stream using two heaps.
    
    The median is the middle value in an ordered integer list. If the size of the list is even,
    there is no middle value and the median is the mean of the two middle values.
    
    Example:
    Input: ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
           [[], [1], [2], [], [3], []]
    Output: [null, null, null, 1.5, null, 2.0]
    
    Time Complexity:
        - addNum: O(log n)
        - findMedian: O(1)
    Space Complexity: O(n)
    """
    
    def __init__(self):
        """Initialize the data structure."""
        # Max heap for the lower half (using negative values)
        self.lower_half: List[int] = []
        # Min heap for the upper half
        self.upper_half: List[int] = []
    
    def add_num(self, num: int) -> None:
        """
        Add a number to the data structure.
        
        Args:
            num: The number to add
        """
        # Add to max heap (lower half)
        heapq.heappush(self.lower_half, -num)
        
        # Balance the heaps
        if (self.lower_half and self.upper_half and 
            -self.lower_half[0] > self.upper_half[0]):
            # Move the largest from lower half to upper half
            val = -heapq.heappop(self.lower_half)
            heapq.heappush(self.upper_half, val)
        
        # Ensure the heaps are balanced in size
        if len(self.lower_half) > len(self.upper_half) + 1:
            # Move one element from lower to upper
            val = -heapq.heappop(self.lower_half)
            heapq.heappush(self.upper_half, val)
        elif len(self.upper_half) > len(self.lower_half):
            # Move one element from upper to lower
            val = heapq.heappop(self.upper_half)
            heapq.heappush(self.lower_half, -val)
    
    def find_median(self) -> float:
        """
        Find the median of all numbers so far.
        
        Returns:
            The median value
        """
        if not self.lower_half and not self.upper_half:
            raise ValueError("No numbers have been added")
        
        if len(self.lower_half) > len(self.upper_half):
            return -self.lower_half[0]
        elif len(self.upper_half) > len(self.lower_half):
            return self.upper_half[0]
        else:
            return (-self.lower_half[0] + self.upper_half[0]) / 2

class MedianFinderOptimized:
    """
    An optimized version of MedianFinder that maintains the heaps in a more balanced way.
    
    This version ensures that:
    1. The lower half (max heap) is always equal to or one more than the upper half (min heap)
    2. All numbers in the lower half are less than or equal to all numbers in the upper half
    """
    
    def __init__(self):
        """Initialize the data structure."""
        self.lower_half: List[int] = []  # Max heap
        self.upper_half: List[int] = []  # Min heap
    
    def add_num(self, num: int) -> None:
        """
        Add a number to the data structure.
        
        Args:
            num: The number to add
        """
        # Always add to the lower half first
        heapq.heappush(self.lower_half, -num)
        
        # Move the largest from lower half to upper half
        val = -heapq.heappop(self.lower_half)
        heapq.heappush(self.upper_half, val)
        
        # If upper half is larger, move smallest back to lower half
        if len(self.upper_half) > len(self.lower_half):
            val = heapq.heappop(self.upper_half)
            heapq.heappush(self.lower_half, -val)
    
    def find_median(self) -> float:
        """
        Find the median of all numbers so far.
        
        Returns:
            The median value
        """
        if not self.lower_half and not self.upper_half:
            raise ValueError("No numbers have been added")
        
        if len(self.lower_half) > len(self.upper_half):
            return -self.lower_half[0]
        else:
            return (-self.lower_half[0] + self.upper_half[0]) / 2

def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    """
    Find the median of two sorted arrays.
    
    Example:
    Input: nums1 = [1, 3], nums2 = [2]
    Output: 2.0
    
    Time Complexity: O(log(min(m, n)))
    Space Complexity: O(1)
    """
    # Ensure nums1 is the shorter array
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        # Partition nums1
        partition_x = (left + right) // 2
        partition_y = (m + n + 1) // 2 - partition_x
        
        # Find the elements around the partition
        max_left_x = float('-inf') if partition_x == 0 else nums1[partition_x - 1]
        min_right_x = float('inf') if partition_x == m else nums1[partition_x]
        
        max_left_y = float('-inf') if partition_y == 0 else nums2[partition_y - 1]
        min_right_y = float('inf') if partition_y == n else nums2[partition_y]
        
        # Check if we found the correct partition
        if max_left_x <= min_right_y and max_left_y <= min_right_x:
            # Found the correct partition
            if (m + n) % 2 == 0:
                return (max(max_left_x, max_left_y) + min(min_right_x, min_right_y)) / 2
            else:
                return max(max_left_x, max_left_y)
        elif max_left_x > min_right_y:
            # Move partition_x to the left
            right = partition_x - 1
        else:
            # Move partition_x to the right
            left = partition_x + 1
    
    raise ValueError("Input arrays are not sorted")

# Example usage
if __name__ == "__main__":
    def test_median_finder():
        """Test the MedianFinder class."""
        print("Testing MedianFinder:")
        finder = MedianFinder()
        
        # Test case 1: [1, 2, 3]
        finder.add_num(1)
        finder.add_num(2)
        print(f"After adding 1, 2: median = {finder.find_median()}")
        finder.add_num(3)
        print(f"After adding 3: median = {finder.find_median()}")
        
        # Test case 2: [1, 2, 3, 4]
        finder2 = MedianFinder()
        for num in [1, 2, 3, 4]:
            finder2.add_num(num)
        print(f"\nAfter adding 1, 2, 3, 4: median = {finder2.find_median()}")
        
        # Test case 3: Empty
        finder3 = MedianFinder()
        try:
            finder3.find_median()
        except ValueError as e:
            print(f"\nEmpty finder: {e}")
    
    def test_median_finder_optimized():
        """Test the MedianFinderOptimized class."""
        print("\nTesting MedianFinderOptimized:")
        finder = MedianFinderOptimized()
        
        # Test case 1: [1, 2, 3]
        finder.add_num(1)
        finder.add_num(2)
        print(f"After adding 1, 2: median = {finder.find_median()}")
        finder.add_num(3)
        print(f"After adding 3: median = {finder.find_median()}")
        
        # Test case 2: [1, 2, 3, 4]
        finder2 = MedianFinderOptimized()
        for num in [1, 2, 3, 4]:
            finder2.add_num(num)
        print(f"\nAfter adding 1, 2, 3, 4: median = {finder2.find_median()}")
    
    def test_find_median_sorted_arrays():
        """Test the find_median_sorted_arrays function."""
        print("\nTesting find_median_sorted_arrays:")
        
        # Test cases
        test_cases = [
            ([1, 3], [2], "Odd total length"),
            ([1, 2], [3, 4], "Even total length"),
            ([], [1], "One empty array"),
            ([1, 2], [], "One empty array"),
            ([1, 2, 3], [4, 5, 6], "No overlap"),
            ([1, 3, 5], [2, 4, 6], "Interleaved"),
        ]
        
        for nums1, nums2, name in test_cases:
            result = find_median_sorted_arrays(nums1, nums2)
            print(f"\n{name}:")
            print(f"nums1 = {nums1}")
            print(f"nums2 = {nums2}")
            print(f"median = {result}")
    
    # Run all tests
    test_median_finder()
    test_median_finder_optimized()
    test_find_median_sorted_arrays()
    
    # Print explanation
    print("\nExplanation:")
    print("1. MedianFinder:")
    print("   - Uses two heaps: max heap for lower half, min heap for upper half")
    print("   - Maintains balance between heaps")
    print("   - O(log n) for addNum, O(1) for findMedian")
    print("2. MedianFinderOptimized:")
    print("   - More efficient implementation")
    print("   - Always adds to lower half first, then rebalances")
    print("   - Same time complexity but fewer operations")
    print("3. find_median_sorted_arrays:")
    print("   - Uses binary search to find the median")
    print("   - O(log(min(m, n))) time complexity")
    print("   - No extra space needed")
    print() 