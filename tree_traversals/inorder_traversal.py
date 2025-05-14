from .tree_node import TreeNode

def inorder_traversal_recursive(root: TreeNode) -> list:
    """
    Inorder traversal (Left -> Root -> Right) using recursion
    Time Complexity: O(n)
    Space Complexity: O(h) where h is the height of the tree
    """
    result = []
    
    def inorder(node):
        if not node:
            return
        inorder(node.left)      # First visit left
        result.append(node.val) # Then visit root
        inorder(node.right)     # Finally visit right
    
    inorder(root)
    return result

def inorder_traversal_iterative(root: TreeNode) -> list:
    """
    Inorder traversal using iteration and stack
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    result = []
    stack = []
    current = root
    
    while current or stack:
        # Reach the leftmost node of current node
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node and move to right subtree
        current = stack.pop()
        result.append(current.val)
        current = current.right
    
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
    
    print("Inorder traversal (recursive):", inorder_traversal_recursive(root))
    print("Inorder traversal (iterative):", inorder_traversal_iterative(root))
    # Output: [4, 2, 5, 1, 3] 