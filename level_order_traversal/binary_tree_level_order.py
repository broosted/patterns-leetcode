from typing import List, Optional, Deque
from collections import deque

class TreeNode:
    """Node for binary tree."""
    def __init__(self, val: int = 0, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

def level_order_traversal(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Perform level order traversal of a binary tree.
    
    Example:
    Input:
        3
       / \
      9  20
         / \
        15  7
    Output: [[3], [9, 20], [15, 7]]
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return []
    
    result = []
    queue: Deque[TreeNode] = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result

def level_order_traversal_zigzag(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Perform zigzag level order traversal of a binary tree.
    First level goes left to right, second level right to left, and so on.
    
    Example:
    Input:
        3
       / \
      9  20
         / \
        15  7
    Output: [[3], [20, 9], [15, 7]]
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return []
    
    result = []
    queue: Deque[TreeNode] = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        current_level = deque()
        
        for _ in range(level_size):
            node = queue.popleft()
            
            # Add to current level based on direction
            if left_to_right:
                current_level.append(node.val)
            else:
                current_level.appendleft(node.val)
            
            # Add children to queue
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(list(current_level))
        left_to_right = not left_to_right
    
    return result

def level_order_traversal_bottom_up(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Perform level order traversal from bottom to top.
    
    Example:
    Input:
        3
       / \
      9  20
         / \
        15  7
    Output: [[15, 7], [9, 20], [3]]
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return []
    
    result = []
    queue: Deque[TreeNode] = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result[::-1]  # Reverse the result

def level_order_traversal_average(root: Optional[TreeNode]) -> List[float]:
    """
    Calculate the average value of nodes at each level.
    
    Example:
    Input:
        3
       / \
      9  20
         / \
        15  7
    Output: [3.0, 14.5, 11.0]
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return []
    
    result = []
    queue: Deque[TreeNode] = deque([root])
    
    while queue:
        level_size = len(queue)
        level_sum = 0
        
        for _ in range(level_size):
            node = queue.popleft()
            level_sum += node.val
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level_sum / level_size)
    
    return result

# Example usage
if __name__ == "__main__":
    def create_test_tree() -> TreeNode:
        """Create a test binary tree."""
        root = TreeNode(3)
        root.left = TreeNode(9)
        root.right = TreeNode(20)
        root.right.left = TreeNode(15)
        root.right.right = TreeNode(7)
        return root
    
    def create_test_tree_2() -> TreeNode:
        """Create another test binary tree."""
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.left = TreeNode(4)
        root.left.right = TreeNode(5)
        root.right.left = TreeNode(6)
        root.right.right = TreeNode(7)
        return root
    
    # Test cases
    test_trees = [
        (create_test_tree(), "Test Tree 1"),
        (create_test_tree_2(), "Test Tree 2"),
        (None, "Empty Tree"),
    ]
    
    for root, name in test_trees:
        print(f"\nTesting {name}:")
        
        # Test regular level order traversal
        result1 = level_order_traversal(root)
        print("\nLevel Order Traversal:")
        print(result1)
        
        # Test zigzag level order traversal
        result2 = level_order_traversal_zigzag(root)
        print("\nZigzag Level Order Traversal:")
        print(result2)
        
        # Test bottom-up level order traversal
        result3 = level_order_traversal_bottom_up(root)
        print("\nBottom-up Level Order Traversal:")
        print(result3)
        
        # Test level order average
        result4 = level_order_traversal_average(root)
        print("\nLevel Order Average:")
        print(result4)
        
        # Print explanation
        if root:
            print("\nExplanation:")
            print("1. Regular Level Order:")
            print("   - Process nodes level by level from top to bottom")
            print("   - Use a queue to maintain the order of nodes")
            print("2. Zigzag Level Order:")
            print("   - Alternate between left-to-right and right-to-left")
            print("   - Use a deque to efficiently add elements at both ends")
            print("3. Bottom-up Level Order:")
            print("   - Same as regular traversal but reverse the result")
            print("4. Level Order Average:")
            print("   - Calculate average value at each level")
            print("   - Useful for finding average value of nodes at each depth")
        print() 