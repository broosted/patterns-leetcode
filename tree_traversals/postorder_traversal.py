from .tree_node import TreeNode

def postorder_traversal_recursive(root: TreeNode) -> list:
    """
    Postorder traversal (Left -> Right -> Root) using recursion
    Time Complexity: O(n)
    Space Complexity: O(h) where h is the height of the tree
    """
    result = []
    
    def postorder(node):
        if not node:
            return
        postorder(node.left)     # First visit left
        postorder(node.right)    # Then visit right
        result.append(node.val)  # Finally visit root
    
    postorder(root)
    return result

def postorder_traversal_iterative(root: TreeNode) -> list:
    """
    Postorder traversal using iteration and stack
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        current = stack.pop()
        result.insert(0, current.val)  # Insert at beginning
        
        # Push left first, then right
        # This ensures right is processed before left
        if current.left:
            stack.append(current.left)
        if current.right:
            stack.append(current.right)
    
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
    
    print("Postorder traversal (recursive):", postorder_traversal_recursive(root))
    print("Postorder traversal (iterative):", postorder_traversal_iterative(root))
    # Output: [4, 5, 2, 3, 1] 