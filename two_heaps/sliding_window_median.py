from typing import List, Deque, Dict, Set
from collections import deque, defaultdict
import heapq

class SlidingWindowMedian:
    """
    Find the median in a sliding window using two heaps.
    
    Example:
    Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
    Output: [1.0,-1.0,-1.0,3.0,5.0,6.0]
    
    Time Complexity: O(n * log k) where n is the length of nums and k is the window size
    Space Complexity: O(k)
    """
    
    def __init__(self, k: int):
        """
        Initialize the data structure.
        
        Args:
            k: The size of the sliding window
        """
        self.k = k
        # Max heap for the lower half (using negative values)
        self.lower_half: List[int] = []
        # Min heap for the upper half
        self.upper_half: List[int] = []
        # Count of elements in each heap
        self.lower_count = 0
        self.upper_count = 0
        # Hash maps to track elements to be removed
        self.lower_to_remove: Dict[int, int] = defaultdict(int)
        self.upper_to_remove: Dict[int, int] = defaultdict(int)
    
    def _clean_heaps(self) -> None:
        """Remove elements that are marked for removal from both heaps."""
        # Clean lower half
        while (self.lower_half and 
               self.lower_to_remove[-self.lower_half[0]] > 0):
            val = -heapq.heappop(self.lower_half)
            self.lower_to_remove[val] -= 1
            if self.lower_to_remove[val] == 0:
                del self.lower_to_remove[val]
        
        # Clean upper half
        while (self.upper_half and 
               self.upper_to_remove[self.upper_half[0]] > 0):
            val = heapq.heappop(self.upper_half)
            self.upper_to_remove[val] -= 1
            if self.upper_to_remove[val] == 0:
                del self.upper_to_remove[val]
    
    def _balance_heaps(self) -> None:
        """Balance the heaps to maintain the median property."""
        # Ensure lower half is at most one element larger than upper half
        while self.lower_count > self.upper_count + 1:
            val = -heapq.heappop(self.lower_half)
            heapq.heappush(self.upper_half, val)
            self.lower_count -= 1
            self.upper_count += 1
        
        # Ensure upper half is not larger than lower half
        while self.upper_count > self.lower_count:
            val = heapq.heappop(self.upper_half)
            heapq.heappush(self.lower_half, -val)
            self.upper_count -= 1
            self.lower_count += 1
        
        self._clean_heaps()
    
    def add_num(self, num: int) -> None:
        """
        Add a number to the window.
        
        Args:
            num: The number to add
        """
        # Add to appropriate heap
        if not self.lower_half or num <= -self.lower_half[0]:
            heapq.heappush(self.lower_half, -num)
            self.lower_count += 1
        else:
            heapq.heappush(self.upper_half, num)
            self.upper_count += 1
        
        self._balance_heaps()
    
    def remove_num(self, num: int) -> None:
        """
        Remove a number from the window.
        
        Args:
            num: The number to remove
        """
        # Mark the number for removal
        if num <= -self.lower_half[0]:
            self.lower_to_remove[num] += 1
            self.lower_count -= 1
        else:
            self.upper_to_remove[num] += 1
            self.upper_count -= 1
        
        self._balance_heaps()
    
    def get_median(self) -> float:
        """
        Get the median of the current window.
        
        Returns:
            The median value
        """
        self._clean_heaps()
        
        if self.lower_count > self.upper_count:
            return -self.lower_half[0]
        else:
            return (-self.lower_half[0] + self.upper_half[0]) / 2

def median_sliding_window(nums: List[int], k: int) -> List[float]:
    """
    Find the median for each sliding window of size k.
    
    Args:
        nums: The input array
        k: The size of the sliding window
    
    Returns:
        A list of medians for each window
    """
    if not nums or k <= 0 or k > len(nums):
        return []
    
    window = SlidingWindowMedian(k)
    result = []
    
    # Initialize the first window
    for i in range(k):
        window.add_num(nums[i])
    result.append(window.get_median())
    
    # Slide the window
    for i in range(k, len(nums)):
        window.remove_num(nums[i - k])
        window.add_num(nums[i])
        result.append(window.get_median())
    
    return result

def median_sliding_window_naive(nums: List[int], k: int) -> List[float]:
    """
    A naive implementation using sorting for each window.
    
    Time Complexity: O(n * k * log k)
    Space Complexity: O(k)
    """
    if not nums or k <= 0 or k > len(nums):
        return []
    
    result = []
    for i in range(len(nums) - k + 1):
        window = sorted(nums[i:i + k])
        if k % 2 == 0:
            median = (window[k // 2 - 1] + window[k // 2]) / 2
        else:
            median = window[k // 2]
        result.append(median)
    
    return result

# Example usage
if __name__ == "__main__":
    def test_sliding_window_median():
        """Test the sliding window median implementations."""
        # Test cases
        test_cases = [
            ([1, 3, -1, -3, 5, 3, 6, 7], 3, "Standard case"),
            ([1, 2, 3, 4, 5], 2, "Even window size"),
            ([1, 2, 3, 4, 5], 1, "Window size 1"),
            ([1, 2, 3, 4, 5], 5, "Window size equals array length"),
            ([1, 1, 1, 1, 1], 3, "All same numbers"),
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4, "Large array"),
        ]
        
        for nums, k, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"nums = {nums}")
            print(f"k = {k}")
            
            # Test efficient implementation
            result1 = median_sliding_window(nums, k)
            print("\nEfficient Implementation:")
            print(f"medians = {result1}")
            
            # Test naive implementation
            result2 = median_sliding_window_naive(nums, k)
            print("\nNaive Implementation:")
            print(f"medians = {result2}")
            
            # Verify results match
            print(f"\nResults match: {result1 == result2}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. Efficient Implementation:")
            print("   - Uses two heaps to maintain the median")
            print("   - O(n * log k) time complexity")
            print("   - O(k) space complexity")
            print("2. Naive Implementation:")
            print("   - Sorts each window")
            print("   - O(n * k * log k) time complexity")
            print("   - O(k) space complexity")
            print("3. Key Differences:")
            print("   - Efficient version maintains sorted order using heaps")
            print("   - Handles element removal efficiently")
            print("   - Uses lazy removal to avoid expensive heap operations")
            print()
    
    # Run tests
    test_sliding_window_median() 