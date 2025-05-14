from typing import Optional, Dict, Any
from collections import OrderedDict

class Node:
    """Node for doubly linked list used in LRU Cache."""
    def __init__(self, key: int, value: Any):
        self.key = key
        self.value = value
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class LRUCache:
    """
    LRU Cache implementation using a hashmap and doubly linked list.
    
    Example:
    cache = LRUCache(2)  # capacity = 2
    cache.put(1, 1)
    cache.put(2, 2)
    cache.get(1)         # returns 1
    cache.put(3, 3)      # evicts key 2
    cache.get(2)         # returns -1 (not found)
    cache.put(4, 4)      # evicts key 1
    cache.get(1)         # returns -1 (not found)
    cache.get(3)         # returns 3
    cache.get(4)         # returns 4
    
    Time Complexity:
    - get: O(1)
    - put: O(1)
    Space Complexity: O(capacity)
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[int, Node] = {}
        self.head = Node(0, 0)  # dummy head
        self.tail = Node(0, 0)  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
    
    def _add_to_front(self, node: Node) -> None:
        """Add node to the front of the list (most recently used)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node: Node) -> None:
        """Remove node from the list."""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_front(self, node: Node) -> None:
        """Move node to the front of the list (most recently used)."""
        self._remove_node(node)
        self._add_to_front(node)
    
    def get(self, key: int) -> int:
        """Get the value of the key if it exists, otherwise return -1."""
        if key not in self.cache:
            return -1
        
        node = self.cache[key]
        self._move_to_front(node)
        return node.value
    
    def put(self, key: int, value: Any) -> None:
        """Put the key-value pair in the cache."""
        if key in self.cache:
            # Update existing node
            node = self.cache[key]
            node.value = value
            self._move_to_front(node)
        else:
            # Create new node
            node = Node(key, value)
            self.cache[key] = node
            self._add_to_front(node)
            self.size += 1
            
            # Remove least recently used if capacity exceeded
            if self.size > self.capacity:
                lru_node = self.tail.prev
                self._remove_node(lru_node)
                del self.cache[lru_node.key]
                self.size -= 1

class LRUCacheOrderedDict:
    """
    Alternative LRU Cache implementation using OrderedDict.
    This is a simpler implementation that uses Python's built-in OrderedDict.
    
    Time Complexity:
    - get: O(1)
    - put: O(1)
    Space Complexity: O(capacity)
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: int) -> int:
        """Get the value of the key if it exists, otherwise return -1."""
        if key not in self.cache:
            return -1
        
        # Move key to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: Any) -> None:
        """Put the key-value pair in the cache."""
        if key in self.cache:
            # Remove existing key to update its position
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used item
            self.cache.popitem(last=False)
        
        # Add new key-value pair
        self.cache[key] = value

# Example usage
if __name__ == "__main__":
    def test_lru_cache(cache_class, name: str) -> None:
        print(f"\nTesting {name}:")
        
        # Test case 1: Basic operations
        cache = cache_class(2)
        print("\nTest case 1: Basic operations")
        print("cache.put(1, 1)")
        cache.put(1, 1)
        print("cache.put(2, 2)")
        cache.put(2, 2)
        print(f"cache.get(1) = {cache.get(1)}")  # returns 1
        print("cache.put(3, 3)")  # evicts key 2
        cache.put(3, 3)
        print(f"cache.get(2) = {cache.get(2)}")  # returns -1 (not found)
        print("cache.put(4, 4)")  # evicts key 1
        cache.put(4, 4)
        print(f"cache.get(1) = {cache.get(1)}")  # returns -1 (not found)
        print(f"cache.get(3) = {cache.get(3)}")  # returns 3
        print(f"cache.get(4) = {cache.get(4)}")  # returns 4
        
        # Test case 2: Update existing key
        cache = cache_class(2)
        print("\nTest case 2: Update existing key")
        print("cache.put(1, 1)")
        cache.put(1, 1)
        print("cache.put(2, 2)")
        cache.put(2, 2)
        print("cache.put(1, 10)")  # update existing key
        cache.put(1, 10)
        print(f"cache.get(1) = {cache.get(1)}")  # returns 10
        print(f"cache.get(2) = {cache.get(2)}")  # returns 2
        
        # Test case 3: Capacity 1
        cache = cache_class(1)
        print("\nTest case 3: Capacity 1")
        print("cache.put(1, 1)")
        cache.put(1, 1)
        print("cache.put(2, 2)")  # evicts key 1
        cache.put(2, 2)
        print(f"cache.get(1) = {cache.get(1)}")  # returns -1
        print(f"cache.get(2) = {cache.get(2)}")  # returns 2
        
        # Test case 4: Empty cache
        cache = cache_class(2)
        print("\nTest case 4: Empty cache")
        print(f"cache.get(1) = {cache.get(1)}")  # returns -1
    
    # Test both implementations
    test_lru_cache(LRUCache, "LRUCache (Doubly Linked List)")
    test_lru_cache(LRUCacheOrderedDict, "LRUCacheOrderedDict")
    
    print("\nImplementation Details:")
    print("1. LRUCache (Doubly Linked List):")
    print("   - Uses a hashmap for O(1) lookups")
    print("   - Uses a doubly linked list to maintain order")
    print("   - Most recently used items at front")
    print("   - Least recently used items at back")
    print("   - Time complexity: O(1) for both get and put")
    print("   - Space complexity: O(capacity)")
    print("\n2. LRUCacheOrderedDict:")
    print("   - Uses Python's built-in OrderedDict")
    print("   - Simpler implementation")
    print("   - Same time and space complexity")
    print("   - Less control over internal implementation") 