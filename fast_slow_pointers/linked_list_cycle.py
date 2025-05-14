from ..linkedlist_reversal.list_node import ListNode

def has_cycle(head: ListNode) -> bool:
    """
    Given head, the head of a linked list, determine if the linked list has a cycle in it.
    
    Example:
    Input: head = [3,2,0,-4], pos = 1 (where pos is the index where the tail connects to)
    Output: true
    Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False

def detect_cycle_start(head: ListNode) -> ListNode:
    """
    Given the head of a linked list, return the node where the cycle begins.
    If there is no cycle, return null.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or not head.next:
        return None
    
    # First phase: detect if there's a cycle
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            break
    else:
        return None  # No cycle found
    
    # Second phase: find the start of the cycle
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow

# Example usage
if __name__ == "__main__":
    # Test case 1: No cycle
    head1 = ListNode.from_list([1, 2, 3, 4])
    print("Test 1 - No cycle:")
    print(f"List: {head1}")
    print(f"Has cycle: {has_cycle(head1)}")
    print(f"Cycle start: {detect_cycle_start(head1)}\n")
    
    # Test case 2: With cycle
    head2 = ListNode.from_list([3, 2, 0, -4])
    # Create cycle: -4 -> 2
    head2.next.next.next.next = head2.next
    print("Test 2 - With cycle:")
    print("List: 3 -> 2 -> 0 -> -4 -> 2 (cycle)")
    print(f"Has cycle: {has_cycle(head2)}")
    cycle_start = detect_cycle_start(head2)
    print(f"Cycle starts at node with value: {cycle_start.val if cycle_start else None}")
    
    # Test case 3: Single node cycle
    head3 = ListNode(1)
    head3.next = head3
    print("\nTest 3 - Single node cycle:")
    print("List: 1 -> 1 (cycle)")
    print(f"Has cycle: {has_cycle(head3)}")
    cycle_start = detect_cycle_start(head3)
    print(f"Cycle starts at node with value: {cycle_start.val if cycle_start else None}") 