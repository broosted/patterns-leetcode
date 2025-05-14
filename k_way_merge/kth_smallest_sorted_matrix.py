from typing import List, Tuple
import heapq

def kth_smallest_sorted_matrix_heap(matrix: List[List[int]], k: int) -> int:
    """
    Find the kth smallest element in a sorted matrix using a min heap.
    
    Example:
    Input: matrix = [
        [ 1,  5,  9],
        [10, 11, 13],
        [12, 13, 15]
    ], k = 8
    Output: 13
    
    Time Complexity: O(k * log n) where n is the number of rows
    Space Complexity: O(n) for the heap
    """
    if not matrix or not matrix[0]:
        raise ValueError("Empty matrix")
    
    n = len(matrix)
    # Initialize min heap with first element of each row
    min_heap = []
    for i in range(n):
        heapq.heappush(min_heap, (matrix[i][0], i, 0))
    
    # Extract k-1 smallest elements
    for _ in range(k - 1):
        val, row, col = heapq.heappop(min_heap)
        if col + 1 < len(matrix[row]):
            heapq.heappush(min_heap, (matrix[row][col + 1], row, col + 1))
    
    # Return kth smallest
    return min_heap[0][0]

def kth_smallest_sorted_matrix_binary_search(matrix: List[List[int]], k: int) -> int:
    """
    Find the kth smallest element in a sorted matrix using binary search.
    
    Time Complexity: O(n * log(max - min)) where n is the number of rows
    Space Complexity: O(1)
    """
    if not matrix or not matrix[0]:
        raise ValueError("Empty matrix")
    
    n = len(matrix)
    left, right = matrix[0][0], matrix[n-1][n-1]
    
    def count_less_equal(mid: int) -> int:
        """Count number of elements less than or equal to mid."""
        count = 0
        row, col = n - 1, 0
        
        while row >= 0 and col < n:
            if matrix[row][col] <= mid:
                count += row + 1
                col += 1
            else:
                row -= 1
        
        return count
    
    while left < right:
        mid = (left + right) // 2
        count = count_less_equal(mid)
        
        if count < k:
            left = mid + 1
        else:
            right = mid
    
    return left

def kth_smallest_sorted_matrix_merge(matrix: List[List[int]], k: int) -> int:
    """
    Find the kth smallest element in a sorted matrix using k-way merge.
    
    Time Complexity: O(k * log n)
    Space Complexity: O(n)
    """
    if not matrix or not matrix[0]:
        raise ValueError("Empty matrix")
    
    n = len(matrix)
    # Initialize min heap with first element of each row
    min_heap = []
    for i in range(n):
        heapq.heappush(min_heap, (matrix[i][0], i, 0))
    
    # Extract k-1 smallest elements
    for _ in range(k - 1):
        val, row, col = heapq.heappop(min_heap)
        if col + 1 < len(matrix[row]):
            heapq.heappush(min_heap, (matrix[row][col + 1], row, col + 1))
    
    # Return kth smallest
    return min_heap[0][0]

def find_k_pairs_with_smallest_sums(nums1: List[int], nums2: List[int], k: int) -> List[Tuple[int, int]]:
    """
    Find k pairs with smallest sums from two sorted arrays.
    
    Example:
    Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
    Output: [(1,2), (1,4), (1,6)]
    
    Time Complexity: O(k * log k)
    Space Complexity: O(k)
    """
    if not nums1 or not nums2:
        return []
    
    # Initialize min heap with first element of nums1 paired with all elements of nums2
    min_heap = []
    for j in range(len(nums2)):
        heapq.heappush(min_heap, (nums1[0] + nums2[j], 0, j))
    
    result = []
    # Extract k smallest pairs
    for _ in range(k):
        if not min_heap:
            break
        sum_val, i, j = heapq.heappop(min_heap)
        result.append((nums1[i], nums2[j]))
        
        # Add next pair from nums1
        if i + 1 < len(nums1):
            heapq.heappush(min_heap, (nums1[i + 1] + nums2[j], i + 1, j))
    
    return result

# Example usage
if __name__ == "__main__":
    def test_kth_smallest_sorted_matrix():
        """Test the kth smallest element in sorted matrix implementations."""
        # Test cases
        test_cases = [
            ([[1,5,9], [10,11,13], [12,13,15]], 8, "Standard case"),
            ([[1,2], [3,4]], 3, "2x2 matrix"),
            ([[1,2,3], [4,5,6], [7,8,9]], 5, "Sequential numbers"),
            ([[1,1,1], [1,1,1], [1,1,1]], 5, "All same numbers"),
        ]
        
        for matrix, k, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"Matrix: {matrix}")
            print(f"k: {k}")
            
            # Test heap approach
            result1 = kth_smallest_sorted_matrix_heap(matrix, k)
            print("\nHeap Approach:")
            print(f"kth smallest: {result1}")
            
            # Test binary search approach
            result2 = kth_smallest_sorted_matrix_binary_search(matrix, k)
            print("\nBinary Search Approach:")
            print(f"kth smallest: {result2}")
            
            # Test merge approach
            result3 = kth_smallest_sorted_matrix_merge(matrix, k)
            print("\nMerge Approach:")
            print(f"kth smallest: {result3}")
            
            # Verify results match
            print(f"\nAll results match: {result1 == result2 == result3}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. Heap Approach:")
            print("   - Use min heap to track smallest elements")
            print("   - O(k * log n) time complexity")
            print("   - O(n) space complexity")
            print("2. Binary Search Approach:")
            print("   - Binary search on value range")
            print("   - Count elements less than or equal to mid")
            print("   - O(n * log(max - min)) time complexity")
            print("   - O(1) space complexity")
            print("3. Merge Approach:")
            print("   - Similar to heap approach")
            print("   - More efficient for small k")
            print("   - Same time and space complexity")
            print()
    
    def test_find_k_pairs_with_smallest_sums():
        """Test the find k pairs with smallest sums implementation."""
        # Test cases
        test_cases = [
            ([1,7,11], [2,4,6], 3, "Standard case"),
            ([1,1,2], [1,2,3], 2, "Duplicate numbers"),
            ([1,2], [3], 3, "One array shorter"),
            ([1,2,3], [4,5,6], 9, "All possible pairs"),
        ]
        
        for nums1, nums2, k, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"nums1: {nums1}")
            print(f"nums2: {nums2}")
            print(f"k: {k}")
            
            # Test find k pairs
            result = find_k_pairs_with_smallest_sums(nums1, nums2, k)
            print(f"k pairs with smallest sums: {result}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. Algorithm:")
            print("   - Use min heap to track smallest sums")
            print("   - Start with first element of nums1 paired with all elements of nums2")
            print("   - For each pair, add next pair from nums1")
            print("   - O(k * log k) time complexity")
            print("   - O(k) space complexity")
            print()
    
    # Run all tests
    test_kth_smallest_sorted_matrix()
    test_find_k_pairs_with_smallest_sums() 