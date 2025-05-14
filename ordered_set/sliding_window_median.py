from typing import List, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum

class Color(Enum):
    """Node color for Red-Black Tree."""
    RED = 1
    BLACK = 2

@dataclass
class Node:
    """Node for Red-Black Tree."""
    value: int
    color: Color = Color.RED
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    parent: Optional['Node'] = None
    size: int = 1  # Size of subtree including this node
    count: int = 1  # Count of duplicate values

class OrderedSet:
    """Ordered set implementation using Red-Black Tree."""
    
    def __init__(self):
        self.root: Optional[Node] = None
        self.size: int = 0
    
    def insert(self, value: int) -> None:
        """Insert a value into the set."""
        if not self.root:
            self.root = Node(value, Color.BLACK)
            self.size = 1
            return
        
        current = self.root
        parent = None
        while current:
            parent = current
            if value == current.value:
                current.count += 1
                self.size += 1
                return
            elif value < current.value:
                current = current.left
            else:
                current = current.right
        
        node = Node(value)
        node.parent = parent
        if value < parent.value:
            parent.left = node
        else:
            parent.right = node
        
        self._update_sizes(node)
        self._fix_insert(node)
        self.size += 1
    
    def remove(self, value: int) -> bool:
        """Remove a value from the set."""
        node = self._find(value)
        if not node:
            return False
        
        if node.count > 1:
            node.count -= 1
            self.size -= 1
            return True
        
        self._delete_node(node)
        self.size -= 1
        return True
    
    def get_kth_smallest(self, k: int) -> Optional[int]:
        """Get kth smallest value (1-based)."""
        if k < 1 or k > self.size:
            return None
        
        current = self.root
        while current:
            left_size = self._get_size(current.left)
            if k <= left_size:
                current = current.left
            elif k <= left_size + current.count:
                return current.value
            else:
                k -= left_size + current.count
                current = current.right
        
        return None
    
    def _find(self, value: int) -> Optional[Node]:
        """Find node with given value."""
        current = self.root
        while current:
            if value == current.value:
                return current
            elif value < current.value:
                current = current.left
            else:
                current = current.right
        return None
    
    def _get_size(self, node: Optional[Node]) -> int:
        """Get size of node's subtree."""
        return node.size if node else 0
    
    def _update_sizes(self, node: Node) -> None:
        """Update sizes of nodes from node to root."""
        while node:
            node.size = (self._get_size(node.left) + 
                        self._get_size(node.right) + node.count)
            node = node.parent
    
    def _fix_insert(self, node: Node) -> None:
        """Fix Red-Black properties after insertion."""
        while node != self.root and node.parent.color == Color.RED:
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle and uncle.color == Color.RED:
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        node = node.parent
                        self._rotate_left(node)
                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._rotate_right(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle and uncle.color == Color.RED:
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._rotate_right(node)
                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._rotate_left(node.parent.parent)
        
        self.root.color = Color.BLACK
    
    def _rotate_left(self, node: Node) -> None:
        """Left rotation."""
        right = node.right
        node.right = right.left
        if right.left:
            right.left.parent = node
        right.parent = node.parent
        if not node.parent:
            self.root = right
        elif node == node.parent.left:
            node.parent.left = right
        else:
            node.parent.right = right
        right.left = node
        node.parent = right
        self._update_sizes(node)
    
    def _rotate_right(self, node: Node) -> None:
        """Right rotation."""
        left = node.left
        node.left = left.right
        if left.right:
            left.right.parent = node
        left.parent = node.parent
        if not node.parent:
            self.root = left
        elif node == node.parent.right:
            node.parent.right = left
        else:
            node.parent.left = left
        left.right = node
        node.parent = left
        self._update_sizes(node)
    
    def _delete_node(self, node: Node) -> None:
        """Delete node from tree."""
        if not node.left and not node.right:
            if node == self.root:
                self.root = None
            elif node == node.parent.left:
                node.parent.left = None
            else:
                node.parent.right = None
            self._update_sizes(node.parent)
        elif not node.left:
            self._replace_node(node, node.right)
        elif not node.right:
            self._replace_node(node, node.left)
        else:
            successor = self._find_min(node.right)
            node.value = successor.value
            node.count = successor.count
            self._delete_node(successor)
    
    def _replace_node(self, old: Node, new: Optional[Node]) -> None:
        """Replace old node with new node."""
        if not old.parent:
            self.root = new
        elif old == old.parent.left:
            old.parent.left = new
        else:
            old.parent.right = new
        if new:
            new.parent = old.parent
        self._update_sizes(old.parent)
    
    def _find_min(self, node: Node) -> Node:
        """Find minimum value in subtree."""
        while node.left:
            node = node.left
        return node

class SlidingWindowMedian:
    """
    Maintains median of a sliding window using two ordered sets.
    One set for values less than or equal to median, one for greater.
    """
    
    def __init__(self, window_size: int):
        """
        Initialize sliding window median finder.
        
        Args:
            window_size: Size of the sliding window
        """
        self.window_size = window_size
        self.lower = OrderedSet()  # Values <= median
        self.upper = OrderedSet()  # Values > median
        self.window = deque()      # Current window values
    
    def add(self, value: int) -> Optional[float]:
        """
        Add a value to the window and return the median.
        Returns None if window is not full.
        
        Time Complexity: O(log k) where k is window size
        """
        # Add to window
        self.window.append(value)
        
        # Remove oldest value if window is full
        if len(self.window) > self.window_size:
            old_value = self.window.popleft()
            self._remove_value(old_value)
        
        # Add new value
        self._add_value(value)
        
        # Return median if window is full
        if len(self.window) == self.window_size:
            return self._get_median()
        return None
    
    def _add_value(self, value: int) -> None:
        """Add value to appropriate set."""
        if not self.lower.root or value <= self._get_median():
            self.lower.insert(value)
        else:
            self.upper.insert(value)
        self._balance_sets()
    
    def _remove_value(self, value: int) -> None:
        """Remove value from appropriate set."""
        if value <= self._get_median():
            self.lower.remove(value)
        else:
            self.upper.remove(value)
        self._balance_sets()
    
    def _balance_sets(self) -> None:
        """Balance the two sets to maintain median property."""
        while self.lower.size > self.upper.size + 1:
            # Move largest from lower to upper
            value = self.lower.get_kth_smallest(self.lower.size)
            if value is not None:
                self.lower.remove(value)
                self.upper.insert(value)
        
        while self.upper.size > self.lower.size:
            # Move smallest from upper to lower
            value = self.upper.get_kth_smallest(1)
            if value is not None:
                self.upper.remove(value)
                self.lower.insert(value)
    
    def _get_median(self) -> float:
        """Get current median value."""
        if self.lower.size > self.upper.size:
            return self.lower.get_kth_smallest(self.lower.size)
        else:
            lower_max = self.lower.get_kth_smallest(self.lower.size)
            upper_min = self.upper.get_kth_smallest(1)
            return (lower_max + upper_min) / 2

# Example usage
if __name__ == "__main__":
    def test_sliding_window_median():
        """Test the sliding window median implementation."""
        # Test cases
        test_cases = [
            ([1, 3, -1, -3, 5, 3, 6, 7], 3, "Standard case"),
            ([1, 2, 3, 4, 5], 2, "Even window size"),
            ([1, 2, 3, 4, 5], 5, "Full array window"),
            ([1, 1, 1, 1, 1], 3, "All same values"),
            ([5, 4, 3, 2, 1], 3, "Decreasing sequence")
        ]
        
        for nums, k, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"Array: {nums}")
            print(f"Window size: {k}")
            
            # Create median finder
            finder = SlidingWindowMedian(k)
            
            # Process array
            print("Medians:")
            medians = []
            for num in nums:
                median = finder.add(num)
                if median is not None:
                    medians.append(median)
                    print(f"Window: {list(finder.window)}, Median: {median}")
            
            print(f"Final medians: {medians}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. Sliding Window Median:")
            print("   - Uses two ordered sets to maintain median")
            print("   - Lower set: values <= median")
            print("   - Upper set: values > median")
            print("2. Operations:")
            print("   - Add/Remove: O(log k) time")
            print("   - Get Median: O(1) time")
            print("   - Space: O(k) where k is window size")
            print("3. Features:")
            print("   - Handles duplicate values")
            print("   - Maintains balanced sets")
            print("   - Efficient median updates")
            print()
    
    # Run tests
    test_sliding_window_median() 