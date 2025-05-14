from .tree_node import TreeNode
from collections import deque

def level_order_traversal(root: TreeNode) -> list:
    """
    Level Order (BFS) traversal using a queue
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            current = queue.popleft()
            current_level.append(current.val)
            
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
        
        result.append(current_level)
    
    return result

def level_order_traversal_simple(root: TreeNode) -> list:
    """
    Simple Level Order traversal that returns a flat list
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        current = queue.popleft()
        result.append(current.val)
        
        if current.left:
            queue.append(current.left)
        if current.right:
            queue.append(current.right)
    
    return result

# Example usage
if __name__ == "__main__":
    # Create a sample tree:
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    
    print("Level Order traversal (by levels):", level_order_traversal(root))
    # Output: [[1], [2, 3], [4, 5]]
    
    print("Level Order traversal (flat):", level_order_traversal_simple(root))
    # Output: [1, 2, 3, 4, 5] 