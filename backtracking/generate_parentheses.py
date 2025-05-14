from typing import List

def generate_parentheses(n: int) -> List[str]:
    """
    Generate all valid combinations of n pairs of parentheses.
    
    Example:
    Input: n = 3
    Output: ["((()))","(()())","(())()","()(())","()()()"]
    
    Time Complexity: O(4^n / sqrt(n)) - Catalan number
    Space Complexity: O(4^n / sqrt(n))
    """
    def backtrack(current: str, open_count: int, close_count: int):
        # Base case: if we've used all parentheses
        if len(current) == 2 * n:
            result.append(current)
            return
        
        # Add opening parenthesis if we haven't used all of them
        if open_count < n:
            backtrack(current + "(", open_count + 1, close_count)
        
        # Add closing parenthesis if we have more opening than closing
        if close_count < open_count:
            backtrack(current + ")", open_count, close_count + 1)
    
    result = []
    backtrack("", 0, 0)
    return result

def generate_parentheses_iterative(n: int) -> List[str]:
    """
    Alternative solution using iterative approach with a stack.
    This might be more efficient for very large n as it avoids recursion overhead.
    
    Time Complexity: O(4^n / sqrt(n))
    Space Complexity: O(4^n / sqrt(n))
    """
    result = []
    stack = [("", 0, 0)]  # (current_string, open_count, close_count)
    
    while stack:
        current, open_count, close_count = stack.pop()
        
        if len(current) == 2 * n:
            result.append(current)
            continue
        
        if open_count < n:
            stack.append((current + "(", open_count + 1, close_count))
        
        if close_count < open_count:
            stack.append((current + ")", open_count, close_count + 1))
    
    return result

# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [0, 1, 2, 3, 4]
    
    for n in test_cases:
        print(f"\nGenerating parentheses for n = {n}")
        
        # Test recursive solution
        result1 = generate_parentheses(n)
        print(f"Recursive solution ({len(result1)} combinations):")
        for i, combo in enumerate(result1, 1):
            print(f"{i}. {combo}")
        
        # Test iterative solution
        result2 = generate_parentheses_iterative(n)
        print(f"\nIterative solution ({len(result2)} combinations):")
        for i, combo in enumerate(result2, 1):
            print(f"{i}. {combo}")
        
        # Verify both solutions give same results
        print(f"\nSolutions match: {set(result1) == set(result2)}")
        
        # Print explanation
        if n > 0:
            print("\nExplanation:")
            print(f"For n = {n}, we need to generate all valid combinations of {n} pairs of parentheses.")
            print("A combination is valid if:")
            print("1. It has equal number of opening and closing parentheses")
            print("2. At any point, the number of closing parentheses cannot exceed opening parentheses")
            print(f"Total valid combinations: {len(result1)}")
        print() 