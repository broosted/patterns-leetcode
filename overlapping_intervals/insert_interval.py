from typing import List

def insert_interval(intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
    """
    Given a list of non-overlapping intervals sorted by their start time,
    insert a new interval into the intervals (merge if necessary).
    
    Example:
    Input: intervals = [[1,3],[6,9]], new_interval = [2,5]
    Output: [[1,5],[6,9]]
    Explanation: Since the new interval [2,5] overlaps with [1,3], merge them into [1,5].
    
    Time Complexity: O(n) where n is the number of intervals
    Space Complexity: O(n)
    """
    if not intervals:
        return [new_interval]
    
    result = []
    i = 0
    n = len(intervals)
    
    # Add all intervals that end before new interval starts
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge all overlapping intervals
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    
    # Add the merged interval
    result.append(new_interval)
    
    # Add all remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result

def insert_interval_alternative(intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
    """
    Alternative solution using binary search to find insertion point.
    This approach might be more efficient for large lists of intervals.
    
    Time Complexity: O(log n + n) where n is the number of intervals
    Space Complexity: O(n)
    """
    if not intervals:
        return [new_interval]
    
    # Find insertion point using binary search
    left, right = 0, len(intervals)
    while left < right:
        mid = (left + right) // 2
        if intervals[mid][0] < new_interval[0]:
            left = mid + 1
        else:
            right = mid
    
    # Insert new interval at the found position
    intervals.insert(left, new_interval)
    
    # Merge overlapping intervals
    return merge_intervals(intervals)

def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """Helper function to merge overlapping intervals"""
    if not intervals:
        return []
    
    merged = []
    current = intervals[0]
    
    for interval in intervals[1:]:
        if current[1] >= interval[0]:
            current[1] = max(current[1], interval[1])
        else:
            merged.append(current)
            current = interval
    
    merged.append(current)
    return merged

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ([[1, 3], [6, 9]], [2, 5]),           # Example from problem
        ([[1, 3], [6, 9]], [4, 5]),           # No overlap
        ([[1, 3], [6, 9]], [3, 6]),           # Adjacent intervals
        ([[1, 3], [6, 9]], [2, 7]),           # Overlap with both
        ([[1, 3], [6, 9]], [0, 10]),          # Complete overlap
        ([], [1, 5]),                         # Empty intervals
        ([[1, 5]], [2, 3]),                   # Single interval, contained
        ([[1, 5]], [6, 8]),                   # Single interval, no overlap
    ]
    
    for intervals, new_interval in test_cases:
        print(f"\nInput intervals: {intervals}")
        print(f"New interval: {new_interval}")
        
        # Test both solutions
        result1 = insert_interval(intervals, new_interval)
        result2 = insert_interval_alternative(intervals, new_interval)
        
        print("Solution 1 (linear):", result1)
        print("Solution 2 (binary search):", result2)
        
        # Print explanation
        print("Explanation:")
        if not intervals:
            print(f"  Inserted new interval: [{new_interval[0]}, {new_interval[1]}]")
        else:
            print(f"  Original intervals: {intervals}")
            print(f"  After inserting [{new_interval[0]}, {new_interval[1]}]: {result1}")
        print() 