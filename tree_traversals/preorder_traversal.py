from .tree_node import TreeNode

def preorder_traversal_recursive(root: TreeNode) -> list:
    """
    Preorder traversal (Root -> Left -> Right) using recursion
    Time Complexity: O(n)
    Space Complexity: O(h) where h is the height of the tree
    """
    result = []
    
    def preorder(node):
        if not node:
            return
        result.append(node.val) # First visit root
        preorder(node.left)     # Then visit left
        preorder(node.right)    # Finally visit right
    
    preorder(root)
    return result

def preorder_traversal_iterative(root: TreeNode) -> list:
    """
    Preorder traversal using iteration and stack
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        current = stack.pop()
        result.append(current.val)
        
        # Push right first, then left
        # This ensures left is processed before right
        if current.right:
            stack.append(current.right)
        if current.left:
            stack.append(current.left)
    
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
    
    print("Preorder traversal (recursive):", preorder_traversal_recursive(root))
    print("Preorder traversal (iterative):", preorder_traversal_iterative(root))
    # Output: [1, 2, 4, 5, 3] 