from typing import List

def two_sum_sorted(numbers: List[int], target: int) -> List[int]:
    """
    Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order,
    find two numbers such that they add up to a specific target number.
    
    Example:
    Input: numbers = [2,7,11,15], target = 9
    Output: [1,2]
    Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left, right = 0, len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        
        if current_sum == target:
            # Return 1-based indices
            return [left + 1, right + 1]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return []  # No solution found

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ([2, 7, 11, 15], 9),     # Output: [1, 2]
        ([2, 3, 4], 6),          # Output: [1, 3]
        ([-1, 0], -1),           # Output: [1, 2]
        ([1, 2, 3, 4, 5], 9),    # Output: [4, 5]
    ]
    
    for numbers, target in test_cases:
        result = two_sum_sorted(numbers, target)
        print(f"Array: {numbers}, target: {target}")
        if result:
            print(f"Indices: {result}")
            print(f"Numbers: {numbers[result[0]-1]} + {numbers[result[1]-1]} = {target}\n")
        else:
            print("No solution found\n") 