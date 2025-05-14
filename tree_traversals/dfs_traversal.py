from .tree_node import TreeNode

def dfs_traversal_recursive(root: TreeNode) -> list:
    """
    DFS traversal using recursion
    This is essentially a preorder traversal (Root -> Left -> Right)
    Time Complexity: O(n)
    Space Complexity: O(h) where h is the height of the tree
    """
    result = []
    
    def dfs(node):
        if not node:
            return
        result.append(node.val)  # Process current node
        dfs(node.left)          # Explore left subtree
        dfs(node.right)         # Explore right subtree
    
    dfs(root)
    return result

def dfs_traversal_iterative(root: TreeNode) -> list:
    """
    DFS traversal using iteration and stack
    This is essentially a preorder traversal (Root -> Left -> Right)
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
    
    print("DFS traversal (recursive):", dfs_traversal_recursive(root))
    print("DFS traversal (iterative):", dfs_traversal_iterative(root))
    # Output: [1, 2, 4, 5, 3] 