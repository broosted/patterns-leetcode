from typing import Dict

def length_of_longest_substring(s: str) -> int:
    """
    Given a string s, find the length of the longest substring without repeating characters.
    
    Example:
    Input: s = "abcabcbb"
    Output: 3
    Explanation: The answer is "abc", with the length of 3.
    
    Time Complexity: O(n)
    Space Complexity: O(min(m, n)) where m is the size of the character set
    """
    if not s:
        return 0
    
    # Dictionary to store the last position of each character
    char_position: Dict[str, int] = {}
    max_length = 0
    start = 0
    
    for end, char in enumerate(s):
        # If we find a repeating character, update the start pointer
        if char in char_position and char_position[char] >= start:
            start = char_position[char] + 1
        else:
            max_length = max(max_length, end - start + 1)
        
        # Update the last position of current character
        char_position[char] = end
    
    return max_length

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        "abcabcbb",    # Expected: 3 ("abc")
        "bbbbb",       # Expected: 1 ("b")
        "pwwkew",      # Expected: 3 ("wke")
        " ",          # Expected: 1 (" ")
        "dvdf",       # Expected: 3 ("vdf")
    ]
    
    for s in test_cases:
        result = length_of_longest_substring(s)
        print(f"String: {s}")
        print(f"Length of longest substring without repeating characters: {result}")
        
        # Find and show the longest substring
        max_len = 0
        longest_sub = ""
        for i in range(len(s)):
            seen = set()
            current = ""
            for j in range(i, len(s)):
                if s[j] in seen:
                    break
                seen.add(s[j])
                current += s[j]
                if len(current) > max_len:
                    max_len = len(current)
                    longest_sub = current
        
        print(f"Longest substring without repeating characters: {longest_sub}\n") 