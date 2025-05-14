from .list_node import ListNode

def reverse_list_iterative(head: ListNode) -> ListNode:
    """
    Reverse a singly linked list iteratively.
    
    Example:
    Input: head = [1,2,3,4,5]
    Output: [5,4,3,2,1]
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    prev = None
    current = head
    
    while current:
        # Store next node
        next_node = current.next
        # Reverse current node's pointer
        current.next = prev
        # Move prev and current one step forward
        prev = current
        current = next_node
    
    return prev

def reverse_list_recursive(head: ListNode) -> ListNode:
    """
    Reverse a singly linked list recursively.
    
    Example:
    Input: head = [1,2,3,4,5]
    Output: [5,4,3,2,1]
    
    Time Complexity: O(n)
    Space Complexity: O(n) due to recursion stack
    """
    # Base case: empty list or single node
    if not head or not head.next:
        return head
    
    # Recursive case: reverse the rest of the list
    new_head = reverse_list_recursive(head.next)
    
    # Reverse the link between current node and next node
    head.next.next = head
    head.next = None
    
    return new_head

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        [1, 2, 3, 4, 5],  # Multiple nodes
        [1, 2],           # Two nodes
        [1],              # Single node
        []                # Empty list
    ]
    
    for values in test_cases:
        # Test iterative reversal
        head1 = ListNode.from_list(values)
        print(f"Original list: {head1}")
        reversed_iter = reverse_list_iterative(head1)
        print(f"Reversed (iterative): {reversed_iter}")
        
        # Test recursive reversal
        head2 = ListNode.from_list(values)
        reversed_rec = reverse_list_recursive(head2)
        print(f"Reversed (recursive): {reversed_rec}")
        print() 