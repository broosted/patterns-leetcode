from typing import List, Optional
import heapq

class ListNode:
    """Node for singly linked list."""
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next
    
    def __lt__(self, other: 'ListNode') -> bool:
        """Compare nodes for min heap."""
        return self.val < other.val

def merge_k_sorted_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Merge k sorted linked lists into one sorted list.
    Uses a min heap to efficiently merge the lists.
    
    Example:
    Input: lists = [[1,4,5], [1,3,4], [2,6]]
    Output: [1,1,2,3,4,4,5,6]
    
    Time Complexity: O(n * log k) where n is total number of nodes and k is number of lists
    Space Complexity: O(k) for the heap
    """
    # Initialize min heap with first node of each list
    min_heap = []
    for head in lists:
        if head:
            heapq.heappush(min_heap, head)
    
    # Create dummy head for result
    dummy = ListNode(0)
    current = dummy
    
    # Merge lists
    while min_heap:
        # Get smallest node
        node = heapq.heappop(min_heap)
        
        # Add to result
        current.next = node
        current = current.next
        
        # Add next node from the same list
        if node.next:
            heapq.heappush(min_heap, node.next)
    
    return dummy.next

def merge_k_sorted_lists_divide_conquer(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Merge k sorted linked lists using divide and conquer approach.
    
    Time Complexity: O(n * log k)
    Space Complexity: O(log k) for recursion stack
    """
    def merge_two_lists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """Merge two sorted linked lists."""
        dummy = ListNode(0)
        current = dummy
        
        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next
        
        current.next = l1 if l1 else l2
        return dummy.next
    
    def merge_lists(lists: List[Optional[ListNode]], start: int, end: int) -> Optional[ListNode]:
        """Recursively merge lists using divide and conquer."""
        if start == end:
            return lists[start]
        if start > end:
            return None
        
        mid = (start + end) // 2
        left = merge_lists(lists, start, mid)
        right = merge_lists(lists, mid + 1, end)
        return merge_two_lists(left, right)
    
    if not lists:
        return None
    return merge_lists(lists, 0, len(lists) - 1)

def merge_k_sorted_arrays(arrays: List[List[int]]) -> List[int]:
    """
    Merge k sorted arrays into one sorted array.
    Uses a min heap to efficiently merge the arrays.
    
    Example:
    Input: arrays = [[1,4,5], [1,3,4], [2,6]]
    Output: [1,1,2,3,4,4,5,6]
    
    Time Complexity: O(n * log k) where n is total number of elements
    Space Complexity: O(k) for the heap
    """
    # Initialize min heap with first element of each array
    min_heap = []
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(min_heap, (arr[0], i, 0))
    
    result = []
    # Merge arrays
    while min_heap:
        val, arr_idx, elem_idx = heapq.heappop(min_heap)
        result.append(val)
        
        # Add next element from the same array
        if elem_idx + 1 < len(arrays[arr_idx]):
            heapq.heappush(min_heap, (arrays[arr_idx][elem_idx + 1], arr_idx, elem_idx + 1))
    
    return result

# Example usage
if __name__ == "__main__":
    def create_linked_list(values: List[int]) -> Optional[ListNode]:
        """Create a linked list from a list of values."""
        if not values:
            return None
        head = ListNode(values[0])
        current = head
        for val in values[1:]:
            current.next = ListNode(val)
            current = current.next
        return head
    
    def linked_list_to_list(head: Optional[ListNode]) -> List[int]:
        """Convert a linked list to a list of values."""
        result = []
        current = head
        while current:
            result.append(current.val)
            current = current.next
        return result
    
    def test_merge_k_sorted_lists():
        """Test the merge k sorted lists implementations."""
        # Test cases
        test_cases = [
            ([[1,4,5], [1,3,4], [2,6]], "Standard case"),
            ([[1,2,3], [4,5,6], [7,8,9]], "No overlap"),
            ([[1,1,1], [2,2,2], [3,3,3]], "Same values"),
            ([], "Empty list"),
            ([[]], "Empty sublist"),
        ]
        
        for lists, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"Input lists: {lists}")
            
            # Convert to linked lists
            linked_lists = [create_linked_list(lst) for lst in lists]
            
            # Test heap approach
            result1 = merge_k_sorted_lists(linked_lists)
            print("\nHeap Approach:")
            print(f"Merged list: {linked_list_to_list(result1)}")
            
            # Test divide and conquer approach
            linked_lists = [create_linked_list(lst) for lst in lists]  # Create fresh lists
            result2 = merge_k_sorted_lists_divide_conquer(linked_lists)
            print("\nDivide and Conquer Approach:")
            print(f"Merged list: {linked_list_to_list(result2)}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. Heap Approach:")
            print("   - Use min heap to track smallest elements")
            print("   - O(n * log k) time complexity")
            print("   - O(k) space complexity")
            print("2. Divide and Conquer Approach:")
            print("   - Recursively merge pairs of lists")
            print("   - Same time complexity")
            print("   - O(log k) space for recursion")
            print()
    
    def test_merge_k_sorted_arrays():
        """Test the merge k sorted arrays implementation."""
        # Test cases
        test_cases = [
            ([[1,4,5], [1,3,4], [2,6]], "Standard case"),
            ([[1,2,3], [4,5,6], [7,8,9]], "No overlap"),
            ([[1,1,1], [2,2,2], [3,3,3]], "Same values"),
            ([], "Empty list"),
            ([[]], "Empty sublist"),
        ]
        
        for arrays, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"Input arrays: {arrays}")
            
            # Test merge k sorted arrays
            result = merge_k_sorted_arrays(arrays)
            print(f"Merged array: {result}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. Algorithm:")
            print("   - Use min heap to track smallest elements")
            print("   - Store array index and element index in heap")
            print("   - O(n * log k) time complexity")
            print("   - O(k) space complexity")
            print()
    
    # Run all tests
    test_merge_k_sorted_lists()
    test_merge_k_sorted_arrays() 