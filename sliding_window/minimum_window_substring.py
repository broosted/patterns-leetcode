from typing import Dict
from collections import Counter

def min_window(s: str, t: str) -> str:
    """
    Given two strings s and t, return the minimum window in s which will contain all the characters in t.
    If there is no such window in s that covers all characters in t, return the empty string "".
    
    Example:
    Input: s = "ADOBECODEBANC", t = "ABC"
    Output: "BANC"
    Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
    
    Time Complexity: O(n) where n is the length of string s
    Space Complexity: O(k) where k is the number of unique characters in t
    """
    if not s or not t or len(s) < len(t):
        return ""
    
    # Create frequency counter for string t
    target_counts = Counter(t)
    required = len(target_counts)
    
    # Initialize variables for sliding window
    window_counts: Dict[str, int] = {}
    formed = 0
    min_len = float('inf')
    result = ""
    
    left = 0
    for right, char in enumerate(s):
        # Add current character to window
        window_counts[char] = window_counts.get(char, 0) + 1
        
        # If current character's frequency matches target, increment formed
        if char in target_counts and window_counts[char] == target_counts[char]:
            formed += 1
        
        # Try to minimize window while maintaining all required characters
        while left <= right and formed == required:
            # Update result if current window is smaller
            if right - left + 1 < min_len:
                min_len = right - left + 1
                result = s[left:right + 1]
            
            # Remove leftmost character from window
            window_counts[s[left]] -= 1
            if s[left] in target_counts and window_counts[s[left]] < target_counts[s[left]]:
                formed -= 1
            
            left += 1
    
    return result

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("ADOBECODEBANC", "ABC"),  # Expected: "BANC"
        ("a", "a"),                # Expected: "a"
        ("a", "aa"),               # Expected: ""
        ("aa", "aa"),              # Expected: "aa"
        ("cabwefgewcwaefgcf", "cae"),  # Expected: "cwae"
    ]
    
    for s, t in test_cases:
        result = min_window(s, t)
        print(f"String s: {s}")
        print(f"String t: {t}")
        print(f"Minimum window substring: {result}")
        
        if result:
            # Verify that result contains all characters from t
            result_counts = Counter(result)
            target_counts = Counter(t)
            valid = all(result_counts[char] >= count for char, count in target_counts.items())
            print(f"Verification: {'Valid' if valid else 'Invalid'}")
        print() 