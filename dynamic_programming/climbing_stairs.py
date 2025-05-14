from typing import List
from functools import lru_cache

def climb_stairs_bottom_up(n: int) -> int:
    """
    Count the number of ways to climb n stairs, where you can take 1 or 2 steps at a time.
    
    Example:
    Input: n = 3
    Output: 3
    Explanation: There are three ways to climb to the top:
    1. 1 step + 1 step + 1 step
    2. 1 step + 2 steps
    3. 2 steps + 1 step
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n <= 2:
        return n
    
    # Initialize first two values
    prev2, prev1 = 1, 2
    
    # Build up to n
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1

def climb_stairs_bottom_up_array(n: int) -> int:
    """
    Alternative solution using an array to store all values.
    This might be more intuitive for understanding the pattern.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if n <= 2:
        return n
    
    # Initialize array with base cases
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    
    # Fill array
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

@lru_cache(maxsize=None)
def climb_stairs_top_down(n: int) -> int:
    """
    Top-down solution using recursion with memoization.
    This approach might be more intuitive for some problems.
    
    Time Complexity: O(n)
    Space Complexity: O(n) for recursion stack and cache
    """
    if n <= 2:
        return n
    
    return climb_stairs_top_down(n-1) + climb_stairs_top_down(n-2)

def climb_stairs_matrix(n: int) -> int:
    """
    Solution using matrix exponentiation.
    This approach is more efficient for very large n.
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    def multiply(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
        return [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0],
             a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0],
             a[1][0] * b[0][1] + a[1][1] * b[1][1]]
        ]
    
    def matrix_pow(mat: List[List[int]], n: int) -> List[List[int]]:
        if n == 0:
            return [[1, 0], [0, 1]]
        if n == 1:
            return mat
        
        half = matrix_pow(mat, n // 2)
        result = multiply(half, half)
        
        if n % 2:
            result = multiply(result, mat)
        
        return result
    
    if n <= 2:
        return n
    
    mat = [[1, 1], [1, 0]]
    result = matrix_pow(mat, n-1)
    return result[0][0] + result[0][1]

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [0, 1, 2, 3, 4, 5, 10, 20]
    
    for n in test_cases:
        print(f"\nTesting n = {n}")
        
        # Test all solutions
        result1 = climb_stairs_bottom_up(n)
        result2 = climb_stairs_bottom_up_array(n)
        result3 = climb_stairs_top_down(n)
        result4 = climb_stairs_matrix(n)
        
        print(f"Bottom-up (space optimized): {result1}")
        print(f"Bottom-up (array): {result2}")
        print(f"Top-down (memoization): {result3}")
        print(f"Matrix exponentiation: {result4}")
        
        # Verify all solutions give same result
        print(f"All solutions match: {result1 == result2 == result3 == result4}")
        
        # Print explanation
        if n > 0:
            print("\nExplanation:")
            print(f"For {n} stairs, we can reach the top in {result1} ways.")
            print("At each step, we can either take 1 or 2 steps.")
            print("The number of ways follows the Fibonacci sequence.")
            print("Ways to climb:")
            print("1. Bottom-up (space optimized): O(n) time, O(1) space")
            print("2. Bottom-up (array): O(n) time, O(n) space")
            print("3. Top-down (memoization): O(n) time, O(n) space")
            print("4. Matrix exponentiation: O(log n) time, O(1) space")
        print() 