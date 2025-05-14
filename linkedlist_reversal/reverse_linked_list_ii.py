from .list_node import ListNode

def reverse_between(head: ListNode, left: int, right: int) -> ListNode:
    """
    Reverse the nodes of the list from position left to position right, and return the reversed list.
    
    Example:
    Input: head = [1,2,3,4,5], left = 2, right = 4
    Output: [1,4,3,2,5]
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or left == right:
        return head
    
    # Create a dummy node to handle the case where left = 1
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    
    # Move prev to the node before the left position
    for _ in range(left - 1):
        prev = prev.next
    
    # Start reversing from the left position
    current = prev.next
    next_node = current.next
    
    # Reverse the nodes between left and right
    for _ in range(right - left):
        current.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node
        next_node = current.next
    
    return dummy.next

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ([1, 2, 3, 4, 5], 2, 4),    # Reverse middle portion
        ([1, 2, 3, 4, 5], 1, 5),    # Reverse entire list
        ([1, 2, 3, 4, 5], 1, 3),    # Reverse from start
        ([1, 2, 3, 4, 5], 3, 5),    # Reverse to end
        ([5], 1, 1),                # Single node
    ]
    
    for values, left, right in test_cases:
        head = ListNode.from_list(values)
        print(f"Original list: {head}")
        print(f"Reverse from position {left} to {right}")
        reversed_list = reverse_between(head, left, right)
        print(f"Result: {reversed_list}")
        print() 