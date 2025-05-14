from typing import Tuple, Dict

def longest_palindrome_top_down(s: str) -> str:
    """
    Find the longest palindromic substring in s using top-down dynamic programming.
    
    Example:
    Input: s = "babad"
    Output: "bab" or "aba"
    Explanation: Both "bab" and "aba" are valid answers.
    
    Time Complexity: O(n^2)
    Space Complexity: O(n^2)
    """
    def is_palindrome(start: int, end: int, memo: Dict[Tuple[int, int], bool]) -> bool:
        # Base cases
        if start >= end:
            return True
        
        # Check memo
        if (start, end) in memo:
            return memo[(start, end)]
        
        # Check if current substring is palindrome
        if s[start] == s[end]:
            memo[(start, end)] = is_palindrome(start + 1, end - 1, memo)
        else:
            memo[(start, end)] = False
        
        return memo[(start, end)]
    
    if not s:
        return ""
    
    n = len(s)
    max_len = 1
    start = 0
    memo = {}
    
    # Try all possible substrings
    for i in range(n):
        for j in range(i + 1, n):
            if is_palindrome(i, j, memo) and j - i + 1 > max_len:
                max_len = j - i + 1
                start = i
    
    return s[start:start + max_len]

def longest_palindrome_expand_around_center(s: str) -> str:
    """
    Alternative solution using expand around center approach.
    This is more space efficient than the DP solution.
    
    Time Complexity: O(n^2)
    Space Complexity: O(1)
    """
    def expand_around_center(left: int, right: int) -> Tuple[int, int]:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - 1
    
    if not s:
        return ""
    
    start = end = 0
    
    for i in range(len(s)):
        # Check odd length palindromes
        left1, right1 = expand_around_center(i, i)
        # Check even length palindromes
        left2, right2 = expand_around_center(i, i + 1)
        
        # Update longest palindrome
        if right1 - left1 > end - start:
            start, end = left1, right1
        if right2 - left2 > end - start:
            start, end = left2, right2
    
    return s[start:end + 1]

def longest_palindrome_manacher(s: str) -> str:
    """
    Solution using Manacher's algorithm.
    This is the most efficient solution for this problem.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Preprocess string to handle even length palindromes
    t = '#'.join('^{}$'.format(s))
    n = len(t)
    p = [0] * n
    center = right = 0
    
    for i in range(1, n - 1):
        # Mirror position
        mirror = 2 * center - i
        
        # Use previously computed palindrome length
        if i < right:
            p[i] = min(right - i, p[mirror])
        
        # Expand palindrome centered at i
        while t[i + p[i] + 1] == t[i - p[i] - 1]:
            p[i] += 1
        
        # Update center and right boundary
        if i + p[i] > right:
            center = i
            right = i + p[i]
    
    # Find longest palindrome
    max_len = max(p)
    center = p.index(max_len)
    start = (center - max_len) // 2
    
    return s[start:start + max_len]

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        "babad",           # Example from problem
        "cbbd",           # Even length palindrome
        "a",              # Single character
        "ac",             # No palindrome
        "racecar",        # Odd length palindrome
        "abba",           # Even length palindrome
        "aacabdkacaa",    # Complex case
        "",               # Empty string
    ]
    
    for s in test_cases:
        print(f"\nInput: s = {s}")
        
        # Test all solutions
        result1 = longest_palindrome_top_down(s)
        result2 = longest_palindrome_expand_around_center(s)
        result3 = longest_palindrome_manacher(s)
        
        print(f"Top-down DP: {result1}")
        print(f"Expand around center: {result2}")
        print(f"Manacher's algorithm: {result3}")
        
        # Verify all solutions give same result
        print(f"All solutions match: {result1 == result2 == result3}")
        
        # Print explanation
        if s:
            print("\nExplanation:")
            print(f"The longest palindromic substring is: {result1}")
            print("Approaches:")
            print("1. Top-down DP: O(n^2) time, O(n^2) space")
            print("2. Expand around center: O(n^2) time, O(1) space")
            print("3. Manacher's algorithm: O(n) time, O(n) space")
            print(f"Length of longest palindrome: {len(result1)}")
        print() 