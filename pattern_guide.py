"""
Pattern Hack Sheet - Core Techniques for Common Algorithmic Problems

Each pattern includes:
1. Core implementation technique
2. Key points to remember
3. Time/Space complexity
"""

from typing import List, Optional, Dict, Any, Tuple, Set
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass
from enum import Enum
import heapq
import time
import random
import threading

# ============================================================================
# Tree Patterns
# ============================================================================

class TreeNode:
    """Node for binary tree."""
    def __init__(self, val: int = 0, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

# Core Tree Traversal Patterns
def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """Left -> Root -> Right traversal"""
    result = []
    stack = []
    curr = root
    while curr or stack:  # Continue while we have nodes to process
        while curr:  # Go as far left as possible
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()  # Process current node
        result.append(curr.val)
        curr = curr.right  # Move to right subtree
    return result

def preorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """Root -> Left -> Right traversal"""
    if not root:
        return []
    result = []
    stack = [root]  # Start with root
    while stack:
        node = stack.pop()
        result.append(node.val)  # Process current node
        if node.right:  # Push right first (LIFO)
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result

def postorder_traversal(root: Optional[TreeNode]) -> List[int]:
    """Left -> Right -> Root traversal"""
    if not root:
        return []
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.insert(0, node.val)  # Insert at beginning (reverse order)
        if node.left:  # Push left first (LIFO)
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    return result

def dfs_traversal(root: Optional[TreeNode]) -> List[int]:
    """Depth-first traversal (preorder)"""
    result = []
    def dfs(node: Optional[TreeNode]) -> None:
        if not node:
            return
        result.append(node.val)  # Process current node
        dfs(node.left)  # Recurse left
        dfs(node.right)  # Recurse right
    dfs(root)
    return result

# Core Tree Traversal Pattern
def level_order_traversal(root: Optional[TreeNode]) -> List[List[int]]:
    """BFS pattern for tree traversal - use for level-wise operations"""
    if not root:
        return []
    result = []
    queue = deque([root])  # Start with root node
    while queue:
        level_size = len(queue)  # Process one level at a time
        current_level = []
        for _ in range(level_size):
            node = queue.popleft()  # Get next node in level
            current_level.append(node.val)
            # Add children for next level
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(current_level)
    return result

# ============================================================================
# Graph Patterns
# ============================================================================

# Core BFS Pattern
def bfs_graph(graph: Dict[int, List[int]], start: int) -> List[int]:
    """BFS for graphs - use for shortest path, level-wise traversal, or finding minimum steps"""
    if not graph:
        return []
    
    visited = set([start])  # Track visited nodes
    queue = deque([start])  # Queue for BFS
    result = []
    
    while queue:
        node = queue.popleft()  # Get next node
        result.append(node)     # Process current node
        
        # Add unvisited neighbors to queue
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

def bfs_shortest_path(graph: Dict[int, List[int]], start: int, end: int) -> List[int]:
    """BFS for finding shortest path in unweighted graph"""
    if start not in graph or end not in graph:
        return []
    
    # Track visited nodes and their parents for path reconstruction
    visited = {start: None}  # node -> parent
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        if node == end:  # Found target
            # Reconstruct path
            path = []
            while node is not None:
                path.append(node)
                node = visited[node]
            return path[::-1]  # Reverse to get start->end
        
        # Add unvisited neighbors
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited[neighbor] = node
                queue.append(neighbor)
    
    return []  # No path found

# Core Topological Sort Pattern
def topological_sort(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """Kahn's algorithm - use for dependency resolution"""
    # Build adjacency list and count incoming edges
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    for u, v in prerequisites:  # v must be completed before u
        graph[v].append(u)
        in_degree[u] += 1
    
    # Start with courses having no prerequisites
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    result = []
    
    # Process courses in topological order
    while queue:
        current = queue.popleft()
        result.append(current)
        # Reduce in-degree for dependent courses
        for next_node in graph[current]:
            in_degree[next_node] -= 1
            if in_degree[next_node] == 0:  # No more prerequisites
                queue.append(next_node)
    
    return result if len(result) == num_courses else []  # Check if all courses can be taken

# Core Union-Find Pattern
class UnionFind:
    """Use for connected components and cycle detection"""
    def __init__(self, n: int):
        self.parent = list(range(n))  # Each node is its own parent initially
        self.rank = [0] * n  # Track tree height for balancing
    
    def find(self, x: int) -> int:
        # Path compression: make every node point directly to root
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:  # Already in same set
            return False
        # Union by rank: attach smaller tree to root of larger tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1
        return True

# ============================================================================
# Array/String Patterns
# ============================================================================

# Core Sliding Window Pattern
def sliding_window_max(nums: List[int], k: int) -> List[int]:
    """Use for subarray/substring problems with fixed window"""
    if not nums or k <= 0:
        return []
    result = []
    window = deque()  # Store indices of potential max values
    for i, num in enumerate(nums):
        # Remove elements outside current window
        while window and window[0] <= i - k:
            window.popleft()
        # Remove smaller elements as they can't be max
        while window and nums[window[-1]] < num:
            window.pop()
        window.append(i)
        # Add max to result when window is full
        if i >= k - 1:
            result.append(nums[window[0]])
    return result

# Core Two Pointers Pattern
def two_sum_sorted(nums: List[int], target: int) -> List[int]:
    """Use for array problems with sorted input"""
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:  # Need larger sum
            left += 1
        else:  # Need smaller sum
            right -= 1
    return []

# ============================================================================
# Search Patterns
# ============================================================================

# Core Binary Search Pattern
def binary_search(nums: List[int], target: int) -> int:
    """Use for sorted array search problems"""
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:  # Search right half
            left = mid + 1
        else:  # Search left half
            right = mid - 1
    return -1

# ============================================================================
# Dynamic Programming Patterns
# ============================================================================

# Core 1D DP Pattern
def fibonacci_dp(n: int) -> int:
    """Use for problems with overlapping subproblems"""
    if n <= 1:
        return n
    dp = [0] * (n + 1)  # dp[i] stores fibonacci(i)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]  # Current = sum of previous two
    return dp[n]

# Core 2D DP Pattern
def longest_common_subsequence(text1: str, text2: str) -> int:
    """Use for problems with two sequences"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]  # dp[i][j] = LCS of text1[:i] and text2[:j]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:  # Characters match
                dp[i][j] = dp[i-1][j-1] + 1
            else:  # Take max of previous states
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# ============================================================================
# Additional Core Patterns
# ============================================================================

# Core Backtracking Pattern
def subsets(nums: List[int]) -> List[List[int]]:
    """Use for problems requiring all possible combinations"""
    result = []
    def backtrack(start: int, curr: List[int]) -> None:
        result.append(curr[:])  # Add current subset
        for i in range(start, len(nums)):
            curr.append(nums[i])  # Include current number
            backtrack(i + 1, curr)  # Recurse with next number
            curr.pop()  # Backtrack: remove current number
    backtrack(0, [])
    return result

# Core Cyclic Sort Pattern
def cyclic_sort(nums: List[int]) -> List[int]:
    """Use for array containing numbers from 1 to n"""
    i = 0
    while i < len(nums):
        correct_pos = nums[i] - 1  # Number should be at index (number-1)
        if nums[i] != nums[correct_pos]:  # If not in correct position
            nums[i], nums[correct_pos] = nums[correct_pos], nums[i]  # Swap
        else:
            i += 1  # Move to next number
    return nums

# Core K-way Merge Pattern
def merge_k_sorted_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """Use for merging k sorted sequences"""
    if not lists:
        return None
    min_heap = []  # Store (value, list_index, node) tuples
    # Add first node of each list to heap
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(min_heap, (node.val, i, node))
    
    dummy = ListNode(0)  # Dummy head for result
    current = dummy
    while min_heap:
        _, _, node = heapq.heappop(min_heap)  # Get smallest value
        current.next = node
        current = current.next
        if node.next:  # Add next node from same list
            heapq.heappush(min_heap, (node.next.val, i, node.next))
    return dummy.next

# Core Overlapping Intervals Pattern
def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """Use for interval problems"""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])  # Sort by start time
    merged = [intervals[0]]
    for interval in intervals[1:]:
        if interval[0] <= merged[-1][1]:  # If intervals overlap
            merged[-1][1] = max(merged[-1][1], interval[1])  # Merge
        else:
            merged.append(interval)  # Add as new interval
    return merged

# Core Top K Elements Pattern
def find_k_largest(nums: List[int], k: int) -> List[int]:
    """Use for problems requiring k largest/smallest elements"""
    min_heap = []  # Keep k largest elements
    for num in nums:
        if len(min_heap) < k:  # Heap not full
            heapq.heappush(min_heap, num)
        elif num > min_heap[0]:  # Found larger number
            heapq.heappop(min_heap)  # Remove smallest
            heapq.heappush(min_heap, num)  # Add new number
    return sorted(min_heap, reverse=True)  # Return in descending order

# ============================================================================
# Heap Patterns
# ============================================================================

class MedianFinder:
    """Two heaps pattern for finding running median"""
    def __init__(self):
        self.max_heap = []  # Store smaller half (max at top)
        self.min_heap = []  # Store larger half (min at top)
    
    def add_num(self, num: int) -> None:
        # Always add to max_heap first
        heapq.heappush(self.max_heap, -num)  # Negate for max heap
        
        # Balance heaps: max_heap can be at most 1 element larger
        if len(self.max_heap) > len(self.min_heap) + 1:
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
        
        # Ensure max_heap's max <= min_heap's min
        if self.min_heap and -self.max_heap[0] > self.min_heap[0]:
            max_val = -heapq.heappop(self.max_heap)
            min_val = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -min_val)
            heapq.heappush(self.min_heap, max_val)
    
    def find_median(self) -> float:
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        return (-self.max_heap[0] + self.min_heap[0]) / 2

# ============================================================================
# Trie Pattern
# ============================================================================

class TrieNode:
    """Node for trie data structure"""
    def __init__(self):
        self.children = {}  # Map of char to TrieNode
        self.is_word = False  # Marks end of word
        self.word_count = 0  # Count of words with this prefix

class Trie:
    """Trie for efficient string operations"""
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """Insert word into trie"""
        curr = self.root
        for char in word:
            if char not in curr.children:
                curr.children[char] = TrieNode()
            curr = curr.children[char]
            curr.word_count += 1  # Increment prefix count
        curr.is_word = True
    
    def search(self, word: str) -> bool:
        """Search for exact word"""
        curr = self.root
        for char in word:
            if char not in curr.children:
                return False
            curr = curr.children[char]
        return curr.is_word
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with prefix"""
        curr = self.root
        for char in prefix:
            if char not in curr.children:
                return False
            curr = curr.children[char]
        return True
    
    def count_words_with_prefix(self, prefix: str) -> int:
        """Count words with given prefix"""
        curr = self.root
        for char in prefix:
            if char not in curr.children:
                return 0
            curr = curr.children[char]
        return curr.word_count

# ============================================================================
# Top K Elements Pattern (Enhanced)
# ============================================================================

def find_k_frequent(nums: List[int], k: int) -> List[int]:
    """Find k most frequent elements using bucket sort"""
    # Count frequencies
    freq = defaultdict(int)
    for num in nums:
        freq[num] += 1
    
    # Create buckets: index = frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, count in freq.items():
        buckets[count].append(num)
    
    # Get k most frequent
    result = []
    for i in range(len(buckets) - 1, 0, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            return result[:k]
    return result

def find_k_closest(points: List[List[int]], k: int) -> List[List[int]]:
    """Find k closest points to origin using max heap"""
    # Store (-distance, x, y) to create max heap
    heap = []
    for x, y in points:
        dist = -(x*x + y*y)  # Negative for max heap
        if len(heap) < k:
            heapq.heappush(heap, (dist, x, y))
        elif dist > heap[0][0]:
            heapq.heappop(heap)
            heapq.heappush(heap, (dist, x, y))
    return [[x, y] for _, x, y in heap]

# ============================================================================
# Prefix Sum Pattern
# ============================================================================

class PrefixSum:
    """Prefix sum pattern for efficient range sum queries"""
    def __init__(self, nums: List[int]):
        self.prefix = [0] * (len(nums) + 1)  # prefix[i] = sum of nums[0...i-1]
        for i in range(len(nums)):
            self.prefix[i + 1] = self.prefix[i] + nums[i]
    
    def sum_range(self, left: int, right: int) -> int:
        """Get sum of range [left, right] in O(1) time"""
        return self.prefix[right + 1] - self.prefix[left]

def subarray_sum_equals_k(nums: List[int], k: int) -> int:
    """Find number of subarrays with sum k using prefix sum"""
    prefix_sum = 0
    count = 0
    sum_count = {0: 1}  # Track frequency of prefix sums
    
    for num in nums:
        prefix_sum += num
        # If (prefix_sum - k) exists, we found a subarray with sum k
        if prefix_sum - k in sum_count:
            count += sum_count[prefix_sum - k]
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
    
    return count

# ============================================================================
# Bitwise Patterns
# ============================================================================

def single_number(nums: List[int]) -> int:
    """Find single number in array where all others appear twice using XOR"""
    result = 0
    for num in nums:
        result ^= num  # XOR cancels out pairs
    return result

# ============================================================================
# Linked List Patterns
# ============================================================================

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

class DoublyListNode:
    def __init__(self, val: int = 0, prev: Optional['DoublyListNode'] = None, next: Optional['DoublyListNode'] = None):
        self.val = val
        self.prev = prev
        self.next = next

def reverse_linked_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """Reverse linked list iteratively"""
    prev = None
    curr = head
    while curr:
        next_node = curr.next  # Save next node
        curr.next = prev       # Reverse link
        prev = curr           # Move prev forward
        curr = next_node      # Move curr forward
    return prev

def reverse_doubly_linked_list(head: Optional[DoublyListNode]) -> Optional[DoublyListNode]:
    """Reverse doubly linked list"""
    curr = head
    while curr:
        # Swap prev and next pointers
        curr.prev, curr.next = curr.next, curr.prev
        # Move to next node (which is now prev due to swap)
        curr = curr.next
    return head.prev if head else None

# ============================================================================
# Stack Patterns
# ============================================================================

def next_greater_element(nums: List[int]) -> List[int]:
    """Find next greater element using monotonic stack"""
    n = len(nums)
    result = [-1] * n
    stack = []  # Monotonic stack (decreasing)
    
    for i in range(n):
        # Pop elements smaller than current
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    
    return result

def valid_parentheses(s: str) -> bool:
    """Check if parentheses are valid using stack"""
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in pairs.values():  # Opening bracket
            stack.append(char)
        elif not stack or stack.pop() != pairs[char]:  # Closing bracket
            return False
    
    return not stack  # Stack should be empty

# ============================================================================
# Knapsack Pattern
# ============================================================================

def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """0/1 Knapsack using dynamic programming"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:  # Can include current item
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]],  # Include
                              dp[i-1][w])  # Exclude
            else:
                dp[i][w] = dp[i-1][w]  # Can't include
    
    return dp[n][capacity]

# ============================================================================
# GCD and Math Patterns
# ============================================================================

def gcd(a: int, b: int) -> int:
    """Find Greatest Common Divisor using Euclidean algorithm"""
    while b:
        a, b = b, a % b  # a becomes b, b becomes remainder
    return a

def gcd_array(nums: List[int]) -> int:
    """Find GCD of array of numbers"""
    result = nums[0]
    for num in nums[1:]:
        result = gcd(result, num)
    return result

def is_prime(n: int) -> bool:
    """Check if number is prime using optimized trial division"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    # Check odd numbers up to sqrt(n)
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# ============================================================================
# Sorting Patterns
# ============================================================================

def merge_sort(nums: List[int]) -> List[int]:
    """Merge sort implementation"""
    if len(nums) <= 1:
        return nums
    
    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    
    # Merge sorted halves
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(nums: List[int]) -> List[int]:
    """Quick sort implementation using Lomuto partition"""
    def partition(low: int, high: int) -> int:
        pivot = nums[high]
        i = low - 1
        for j in range(low, high):
            if nums[j] <= pivot:
                i += 1
                nums[i], nums[j] = nums[j], nums[i]
        nums[i + 1], nums[high] = nums[high], nums[i + 1]
        return i + 1
    
    def sort(low: int, high: int) -> None:
        if low < high:
            pivot_idx = partition(low, high)
            sort(low, pivot_idx - 1)
            sort(pivot_idx + 1, high)
    
    nums = nums.copy()  # Don't modify input
    sort(0, len(nums) - 1)
    return nums

# ============================================================================
# String Patterns
# ============================================================================

def is_palindrome(s: str) -> bool:
    """Check if string is palindrome using two pointers"""
    # Convert to lowercase and remove non-alphanumeric
    s = ''.join(c.lower() for c in s if c.isalnum())
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

def longest_palindrome(s: str) -> str:
    """Find longest palindromic substring using expand around center"""
    def expand(left: int, right: int) -> str:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    longest = ""
    for i in range(len(s)):
        # Check odd length palindromes
        odd = expand(i, i)
        if len(odd) > len(longest):
            longest = odd
        # Check even length palindromes
        even = expand(i, i + 1)
        if len(even) > len(longest):
            longest = even
    return longest

# ============================================================================
# Example Usage
# ============================================================================

def test_tree_traversals():
    """Test tree traversal implementations."""
    # Create sample tree:
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
    
    print("\nTesting Tree Traversals:")
    print("Level Order:", level_order_traversal(root))

def test_graph_patterns():
    """Test graph pattern implementations."""
    # Test topological sort
    num_courses = 4
    prerequisites = [[1,0], [2,0], [3,1], [3,2]]
    print("\nTesting Topological Sort:")
    print("Courses:", num_courses)
    print("Prerequisites:", prerequisites)
    print("Order:", topological_sort(num_courses, prerequisites))
    
    # Test Union-Find
    uf = UnionFind(5)
    print("\nTesting Union-Find:")
    print("Union(0,1):", uf.union(0, 1))
    print("Union(1,2):", uf.union(1, 2))
    print("Union(3,4):", uf.union(3, 4))
    print("Find(0):", uf.find(0))
    print("Find(3):", uf.find(3))

def test_array_patterns():
    """Test array pattern implementations."""
    # Test sliding window
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    print("\nTesting Sliding Window:")
    print("Array:", nums)
    print("Window size:", k)
    print("Max in windows:", sliding_window_max(nums, k))
    
    # Test two pointers
    nums = [1, 2, 3, 4, 5]
    target = 7
    print("\nTesting Two Pointers:")
    print("Array:", nums)
    print("Target:", target)
    print("Two sum indices:", two_sum_sorted(nums, target))

def test_search_patterns():
    """Test search pattern implementations."""
    # Test binary search
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    target = 5
    print("\nTesting Binary Search:")
    print("Array:", nums)
    print("Target:", target)
    print("Index:", binary_search(nums, target))

def test_dp_patterns():
    """Test dynamic programming pattern implementations."""
    # Test 1D DP
    n = 10
    print("\nTesting 1D DP (Fibonacci):")
    print("n:", n)
    print("Fibonacci(n):", fibonacci_dp(n))
    
    # Test 2D DP
    text1 = "abcde"
    text2 = "ace"
    print("\nTesting 2D DP (LCS):")
    print("Text1:", text1)
    print("Text2:", text2)
    print("LCS length:", longest_common_subsequence(text1, text2))

def test_additional_patterns():
    """Test additional pattern implementations."""
    # Test Backtracking
    nums = [1, 2, 3]
    print("\nTesting Backtracking:")
    print("Array:", nums)
    print("Subsets:", subsets(nums))
    
    # Test Cyclic Sort
    nums = [3, 1, 5, 4, 2]
    print("\nTesting Cyclic Sort:")
    print("Original array:", nums)
    print("Sorted array:", cyclic_sort(nums))
    
    # Test K-way Merge
    # Create sample linked lists: 1->4->5, 1->3->4, 2->6
    list1 = ListNode(1, ListNode(4, ListNode(5)))
    list2 = ListNode(1, ListNode(3, ListNode(4)))
    list3 = ListNode(2, ListNode(6))
    lists = [list1, list2, list3]
    merged = merge_k_sorted_lists(lists)
    result = []
    while merged:
        result.append(merged.val)
        merged = merged.next
    print("\nTesting K-way Merge:")
    print("Merged k sorted lists:", result)
    
    # Test Overlapping Intervals
    intervals = [[1,3], [2,6], [8,10], [15,18]]
    print("\nTesting Overlapping Intervals:")
    print("Original intervals:", intervals)
    print("Merged intervals:", merge_intervals(intervals))
    
    # Test Top K Elements
    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    print("\nTesting Top K Elements:")
    print("Array:", nums)
    print("K largest elements:", find_k_largest(nums, k))

if __name__ == "__main__":
    print("Pattern Hack Sheet - Core Techniques")
    print("===================================")
    print("\nKey Patterns and Their Use Cases:")
    print("1. Tree Patterns:")
    print("   - Tree Traversals (Inorder, Preorder, Postorder, DFS)")
    print("   - Level Order (BFS): Use for level-wise operations")
    print("2. Graph Patterns:")
    print("   - BFS: Use for shortest path and level-wise traversal")
    print("   - Topological Sort: Use for dependency resolution")
    print("   - Union-Find: Use for connected components")
    print("3. Array/String Patterns:")
    print("   - Sliding Window: Use for subarray problems")
    print("   - Two Pointers: Use for sorted array problems")
    print("4. Search Patterns:")
    print("   - Binary Search: Use for sorted array search")
    print("5. Dynamic Programming:")
    print("   - 1D DP: Use for overlapping subproblems")
    print("   - 2D DP: Use for two sequence problems")
    print("6. Heap Patterns:")
    print("   - Two Heaps: Use for running median/frequency")
    print("   - Top K Elements: Use for k largest/frequent/closest")
    print("7. Trie Pattern:")
    print("   - Use for efficient string operations")
    print("8. Additional Patterns:")
    print("   - Backtracking: Use for combinations/permutations")
    print("   - Cyclic Sort: Use for 1 to n array problems")
    print("   - K-way Merge: Use for merging sorted sequences")
    print("   - Overlapping Intervals: Use for interval problems")
    print("9. Prefix Sum Pattern:")
    print("   - Range Sum Queries: Use for efficient sum calculations")
    print("   - Subarray Sum: Use for finding subarrays with target sum")
    print("10. Bitwise Patterns:")
    print("    - XOR: Use for finding single number")
    print("11. Linked List Patterns:")
    print("    - Reversal: Use for reversing singly/doubly linked lists")
    print("12. Stack Patterns:")
    print("    - Monotonic Stack: Use for next greater element")
    print("    - Stack: Use for parentheses matching")
    print("13. Knapsack Pattern:")
    print("    - 0/1 Knapsack: Use for subset sum problems")
    print("14. GCD and Math Patterns:")
    print("    - GCD: Use for finding greatest common divisor")
    print("    - Prime: Use for prime number operations")
    print("15. Sorting Patterns:")
    print("    - Merge Sort: Use for stable O(n log n) sorting")
    print("    - Quick Sort: Use for in-place O(n log n) sorting")
    print("16. String Patterns:")
    print("    - Palindrome: Use for palindrome checking and finding")
    
    print("\nTime Complexities:")
    print("Tree Operations: O(n)")
    print("Graph Operations: O(V + E) where V=vertices, E=edges")
    print("Array Operations: O(n)")
    print("Search Operations: O(log n)")
    print("DP Operations: O(n) or O(n²)")
    print("Heap Operations: O(log n)")
    print("Trie Operations: O(m) where m is word length")
    print("Backtracking: O(2^n) or O(n!)")
    print("Cyclic Sort: O(n)")
    print("K-way Merge: O(n log k)")
    print("Intervals: O(n log n)")
    print("Top K: O(n log k)")
    print("Prefix Sum: O(1) for queries, O(n) for initialization")
    print("Bitwise Operations: O(n)")
    print("Linked List Operations: O(n)")
    print("Stack Operations: O(n)")
    print("Knapsack: O(nW) where W is capacity")
    print("GCD: O(log min(a,b))")
    print("Prime Check: O(sqrt(n))")
    print("Merge Sort: O(n log n)")
    print("Quick Sort: O(n log n) average, O(n²) worst")
    print("Palindrome Check: O(n)")
    print("Longest Palindrome: O(n²)")
    
    print("\nSpace Complexities:")
    print("1. Tree Operations: O(h) where h is height")
    print("2. Graph Operations: O(V)")
    print("3. Array Operations: O(1) to O(n)")
    print("4. Search Operations: O(1)")
    print("5. DP Operations: O(n) or O(n²)")
    print("6. Additional Operations: O(1) to O(n)")
    print("12. Prefix Sum: O(n)")
    print("13. Bitwise: O(1)")
    print("14. Linked List: O(1)")
    print("15. Stack: O(n)")
    print("16. Knapsack: O(nW)")
    print("17. GCD: O(1)")
    print("18. Prime: O(1)")
    print("19. Merge Sort: O(n)")
    print("20. Quick Sort: O(log n) for recursion stack")
    print("21. Palindrome: O(1) for check, O(n) for longest")
    
    print("\nAdditional Space Complexities:")
    print("7. Backtracking: O(n)")
    print("8. Cyclic Sort: O(1)")
    print("9. K-way Merge: O(k)")
    print("10. Overlapping Intervals: O(n)")
    print("11. Top K: O(k)")
    print("12. Bitwise: O(1)")
    print("13. Linked List: O(1)")
    print("14. Stack: O(n)")
    print("15. Knapsack: O(nW)")
    print("16. GCD: O(1)")
    print("17. Prime: O(1)")
    print("18. Merge Sort: O(n)")
    print("19. Quick Sort: O(log n) for recursion stack")
    print("20. Palindrome: O(1) for check, O(n) for longest")
    
    # Run all tests
    test_tree_traversals()
    test_graph_patterns()
    test_array_patterns()
    test_search_patterns()
    test_dp_patterns()
    
    # Add test for additional patterns
    test_additional_patterns()
    
    print("\nPattern Guide Summary:")
    print("1. Tree Patterns:")
    print("   - Level Order Traversal (BFS)")
    print("2. Graph Patterns:")
    print("   - Topological Sort")
    print("   - Union-Find")
    print("3. Array/String Patterns:")
    print("   - Sliding Window")
    print("   - Two Pointers")
    print("4. Search Patterns:")
    print("   - Binary Search")
    print("5. Dynamic Programming Patterns:")
    print("   - 1D DP")
    print("   - 2D DP")
    print("6. Additional Patterns:")
    print("   - Backtracking")
    print("   - Cyclic Sort")
    print("   - K-way Merge")
    print("   - Overlapping Intervals")
    print("   - Top K Elements")
    print("7. Prefix Sum Pattern:")
    print("   - Range Sum Queries")
    print("   - Subarray Sum")
    print("8. Bitwise Patterns:")
    print("   - XOR")
    print("9. Linked List Patterns:")
    print("   - Reversal")
    print("10. Stack Patterns:")
    print("   - Monotonic Stack")
    print("   - Stack")
    print("11. Knapsack Pattern:")
    print("   - 0/1 Knapsack")
    print("12. GCD and Math Patterns:")
    print("   - GCD")
    print("   - Prime")
    print("13. Sorting Patterns:")
    print("   - Merge Sort")
    print("   - Quick Sort")
    print("14. String Patterns:")
    print("   - Palindrome")
    print("   - Longest Palindrome")
    
    print("\nSpace Complexities:")
    print("1. Tree Operations: O(h) where h is height")
    print("2. Graph Operations: O(V)")
    print("3. Array Operations: O(1) to O(n)")
    print("4. Search Operations: O(1)")
    print("5. DP Operations: O(n) or O(n²)")
    print("6. Additional Operations: O(1) to O(n)")
    print("12. Prefix Sum: O(n)")
    print("13. Bitwise: O(1)")
    print("14. Linked List: O(1)")
    print("15. Stack: O(n)")
    print("16. Knapsack: O(nW)")
    print("17. GCD: O(1)")
    print("18. Prime: O(1)")
    print("19. Merge Sort: O(n)")
    print("20. Quick Sort: O(log n) for recursion stack")
    print("21. Palindrome: O(1) for check, O(n) for longest")
    
    print("\nAdditional Space Complexities:")
    print("7. Backtracking: O(n)")
    print("8. Cyclic Sort: O(1)")
    print("9. K-way Merge: O(k)")
    print("10. Overlapping Intervals: O(n)")
    print("11. Top K: O(k)")
    print("12. Bitwise: O(1)")
    print("13. Linked List: O(1)")
    print("14. Stack: O(n)")
    print("15. Knapsack: O(nW)")
    print("16. GCD: O(1)")
    print("17. Prime: O(1)")
    print("18. Merge Sort: O(n)")
    print("19. Quick Sort: O(log n) for recursion stack")
    print("20. Palindrome: O(1) for check, O(n) for longest")
    print("21. Bitwise: O(1)")
    print("22. Linked List: O(1)")
    print("23. Stack: O(n)")
    print("24. Knapsack: O(nW)") 