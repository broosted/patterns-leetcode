from typing import List

def max_area(height: List[int]) -> int:
    """
    Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai).
    n vertical lines are drawn such that the two endpoints of the line i is at (i, ai) and (i, 0).
    Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.
    
    Example:
    Input: height = [1,8,6,2,5,4,8,3,7]
    Output: 49
    Explanation: The maximum area is obtained by choosing height[1] = 8 and height[8] = 7
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    max_water = 0
    left, right = 0, len(height) - 1
    
    while left < right:
        # Calculate current area
        width = right - left
        current_height = min(height[left], height[right])
        current_area = width * current_height
        
        # Update max area
        max_water = max(max_water, current_area)
        
        # Move the pointer pointing to the smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        [1, 8, 6, 2, 5, 4, 8, 3, 7],  # Expected: 49
        [1, 1],                        # Expected: 1
        [4, 3, 2, 1, 4],              # Expected: 16
        [1, 2, 1],                    # Expected: 2
    ]
    
    for height in test_cases:
        result = max_area(height)
        print(f"Height array: {height}")
        print(f"Maximum water container area: {result}")
        print("Visualization:")
        max_height = max(height)
        for h in range(max_height, 0, -1):
            line = ""
            for val in height:
                if val >= h:
                    line += "█ "
                else:
                    line += "  "
            print(line)
        print() 