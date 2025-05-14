from typing import List

def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """
    Given an array of intervals where intervals[i] = [starti, endi],
    merge all overlapping intervals, and return an array of the non-overlapping intervals
    that cover all the intervals in the input.
    
    Example:
    Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
    Output: [[1,6],[8,10],[15,18]]
    Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
    
    Time Complexity: O(n log n) where n is the number of intervals
    Space Complexity: O(n)
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    current = intervals[0]
    
    for interval in intervals[1:]:
        # If current interval overlaps with next interval
        if current[1] >= interval[0]:
            # Merge intervals by taking the maximum end time
            current[1] = max(current[1], interval[1])
        else:
            # No overlap, add current interval to result
            merged.append(current)
            current = interval
    
    # Add the last interval
    merged.append(current)
    return merged

def merge_intervals_alternative(intervals: List[List[int]]) -> List[List[int]]:
    """
    Alternative solution using a stack.
    This approach might be more intuitive for some people.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    stack = [intervals[0]]
    
    for interval in intervals[1:]:
        # If current interval overlaps with top of stack
        if stack[-1][1] >= interval[0]:
            # Merge intervals by taking the maximum end time
            stack[-1][1] = max(stack[-1][1], interval[1])
        else:
            # No overlap, push current interval to stack
            stack.append(interval)
    
    return stack

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        [[1, 3], [2, 6], [8, 10], [15, 18]],  # Example from problem
        [[1, 4], [4, 5]],                      # Adjacent intervals
        [[1, 4], [0, 4]],                      # Complete overlap
        [[1, 4], [2, 3]],                      # Partial overlap
        [[1, 4], [5, 6]],                      # No overlap
        [],                                     # Empty input
        [[1, 4]],                              # Single interval
    ]
    
    for intervals in test_cases:
        print(f"\nInput intervals: {intervals}")
        
        # Test both solutions
        result1 = merge_intervals(intervals)
        result2 = merge_intervals_alternative(intervals)
        
        print("Solution 1 (in-place):", result1)
        print("Solution 2 (stack):", result2)
        
        # Print explanation
        if intervals:
            print("Explanation:")
            for i, interval in enumerate(result1):
                if i > 0:
                    print(f"  and")
                print(f"  Interval {i+1}: [{interval[0]}, {interval[1]}]")
        print() 