class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __str__(self):
        """Helper method to print the linked list"""
        result = []
        current = self
        while current:
            result.append(str(current.val))
            current = current.next
        return " -> ".join(result)
    
    @classmethod
    def from_list(cls, values):
        """Helper method to create a linked list from a list of values"""
        if not values:
            return None
        head = cls(values[0])
        current = head
        for val in values[1:]:
            current.next = cls(val)
            current = current.next
        return head 