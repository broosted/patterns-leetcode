from ..linkedlist_reversal.list_node import ListNode

def find_middle(head: ListNode) -> ListNode:
    """
    Given the head of a singly linked list, return the middle node of the linked list.
    If there are two middle nodes, return the second middle node.
    
    Example:
    Input: head = [1,2,3,4,5]
    Output: [3,4,5]
    Explanation: The middle node of the list is node 3.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head:
        return None
    
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow

def find_middle_left(head: ListNode) -> ListNode:
    """
    Given the head of a singly linked list, return the first middle node of the linked list.
    If there are two middle nodes, return the first middle node.
    
    Example:
    Input: head = [1,2,3,4,5]
    Output: [3,4,5]
    Input: head = [1,2,3,4]
    Output: [2,3,4]
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head:
        return None
    
    slow = fast = head
    
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        [1, 2, 3, 4, 5],      # Odd length
        [1, 2, 3, 4],         # Even length
        [1, 2],               # Two nodes
        [1],                  # Single node
        []                    # Empty list
    ]
    
    for values in test_cases:
        head = ListNode.from_list(values)
        print(f"List: {head}")
        
        middle = find_middle(head)
        print(f"Middle node (second if even): {middle.val if middle else None}")
        
        middle_left = find_middle_left(head)
        print(f"Middle node (first if even): {middle_left.val if middle_left else None}")
        print() 