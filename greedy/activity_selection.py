from typing import List, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Activity:
    """Represents an activity with start and end times."""
    start: int
    end: int
    name: str = ""

def activity_selection(activities: List[Activity]) -> List[Activity]:
    """
    Find the maximum number of activities that can be performed by a single person.
    Uses a greedy approach by selecting activities with earliest end times.
    
    Example:
    Input: activities = [
        Activity(1, 4, "A"),
        Activity(3, 5, "B"),
        Activity(0, 6, "C"),
        Activity(5, 7, "D"),
        Activity(3, 8, "E"),
        Activity(5, 9, "F"),
        Activity(6, 10, "G"),
        Activity(8, 11, "H"),
        Activity(8, 12, "I"),
        Activity(2, 13, "J"),
        Activity(12, 14, "K")
    ]
    Output: [A, D, G, K]
    
    Time Complexity: O(n log n) for sorting
    Space Complexity: O(n) for the result
    """
    if not activities:
        return []
    
    # Sort activities by end time
    sorted_activities = sorted(activities, key=lambda x: x.end)
    
    result = [sorted_activities[0]]
    last_end = sorted_activities[0].end
    
    # Select activities that don't overlap
    for activity in sorted_activities[1:]:
        if activity.start >= last_end:
            result.append(activity)
            last_end = activity.end
    
    return result

def activity_selection_with_weights(activities: List[Activity], weights: List[int]) -> List[Activity]:
    """
    Find the maximum weight subset of non-overlapping activities.
    Uses dynamic programming approach.
    
    Time Complexity: O(n^2)
    Space Complexity: O(n)
    """
    if not activities or len(activities) != len(weights):
        return []
    
    # Sort activities by end time
    sorted_indices = sorted(range(len(activities)), key=lambda i: activities[i].end)
    sorted_activities = [activities[i] for i in sorted_indices]
    sorted_weights = [weights[i] for i in sorted_indices]
    
    # Find the last non-overlapping activity for each activity
    last_non_overlapping = [-1] * len(sorted_activities)
    for i in range(len(sorted_activities)):
        for j in range(i - 1, -1, -1):
            if sorted_activities[j].end <= sorted_activities[i].start:
                last_non_overlapping[i] = j
                break
    
    # Dynamic programming to find maximum weight
    dp = [0] * len(sorted_activities)
    dp[0] = sorted_weights[0]
    
    for i in range(1, len(sorted_activities)):
        # Include current activity
        include = sorted_weights[i]
        if last_non_overlapping[i] != -1:
            include += dp[last_non_overlapping[i]]
        
        # Exclude current activity
        exclude = dp[i - 1]
        
        dp[i] = max(include, exclude)
    
    # Reconstruct the solution
    result = []
    i = len(sorted_activities) - 1
    while i >= 0:
        if i == 0 or dp[i] != dp[i - 1]:
            result.append(sorted_activities[i])
            i = last_non_overlapping[i]
        else:
            i -= 1
    
    return result[::-1]

def activity_selection_with_deadlines(activities: List[Activity], deadlines: List[int]) -> List[Activity]:
    """
    Schedule activities to maximize the number of activities completed before their deadlines.
    Uses a greedy approach by selecting activities with earliest deadlines.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if not activities or len(activities) != len(deadlines):
        return []
    
    # Sort activities by deadline
    sorted_indices = sorted(range(len(activities)), key=lambda i: deadlines[i])
    sorted_activities = [activities[i] for i in sorted_indices]
    sorted_deadlines = [deadlines[i] for i in sorted_indices]
    
    result = []
    current_time = 0
    
    # Select activities that can be completed before their deadlines
    for i, activity in enumerate(sorted_activities):
        if current_time + (activity.end - activity.start) <= sorted_deadlines[i]:
            result.append(activity)
            current_time += (activity.end - activity.start)
    
    return result

# Example usage
if __name__ == "__main__":
    def test_activity_selection():
        """Test the activity selection implementations."""
        # Test cases
        activities = [
            Activity(1, 4, "A"),
            Activity(3, 5, "B"),
            Activity(0, 6, "C"),
            Activity(5, 7, "D"),
            Activity(3, 8, "E"),
            Activity(5, 9, "F"),
            Activity(6, 10, "G"),
            Activity(8, 11, "H"),
            Activity(8, 12, "I"),
            Activity(2, 13, "J"),
            Activity(12, 14, "K")
        ]
        
        weights = [3, 5, 5, 3, 0, 6, 8, 8, 2, 12, 2]
        deadlines = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        
        print("\nTesting Activity Selection:")
        print("Activities:")
        for activity in activities:
            print(f"{activity.name}: {activity.start}-{activity.end}")
        
        # Test basic activity selection
        result1 = activity_selection(activities)
        print("\nBasic Activity Selection:")
        print("Selected activities:", [a.name for a in result1])
        print("Number of activities:", len(result1))
        
        # Test weighted activity selection
        result2 = activity_selection_with_weights(activities, weights)
        print("\nWeighted Activity Selection:")
        print("Selected activities:", [a.name for a in result2])
        print("Total weight:", sum(weights[activities.index(a)] for a in result2))
        
        # Test activity selection with deadlines
        result3 = activity_selection_with_deadlines(activities, deadlines)
        print("\nActivity Selection with Deadlines:")
        print("Selected activities:", [a.name for a in result3])
        print("Number of activities:", len(result3))
        
        # Print explanation
        print("\nExplanation:")
        print("1. Basic Activity Selection:")
        print("   - Greedy approach: select activities with earliest end times")
        print("   - O(n log n) time complexity for sorting")
        print("   - O(n) space complexity")
        print("2. Weighted Activity Selection:")
        print("   - Dynamic programming approach")
        print("   - O(n^2) time complexity")
        print("   - O(n) space complexity")
        print("3. Activity Selection with Deadlines:")
        print("   - Greedy approach: select activities with earliest deadlines")
        print("   - O(n log n) time complexity")
        print("   - O(n) space complexity")
        print()
    
    # Run tests
    test_activity_selection() 