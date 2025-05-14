from typing import List

def single_number(nums: List[int]) -> int:
    """
    Find the single number in an array where all other numbers appear twice.
    Uses bitwise XOR operation.
    
    Example:
    Input: nums = [4, 1, 2, 1, 2]
    Output: 4
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result

def single_number_ii(nums: List[int]) -> int:
    """
    Find the single number in an array where all other numbers appear three times.
    Uses bitwise operations to count bits.
    
    Example:
    Input: nums = [2, 2, 3, 2]
    Output: 3
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    ones = twos = 0
    for num in nums:
        # Update ones and twos
        ones = (ones ^ num) & ~twos
        twos = (twos ^ num) & ~ones
    return ones

def single_number_iii(nums: List[int]) -> List[int]:
    """
    Find two single numbers in an array where all other numbers appear twice.
    Uses bitwise XOR and masking.
    
    Example:
    Input: nums = [1, 2, 1, 3, 2, 5]
    Output: [3, 5]
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # XOR all numbers
    xor_result = 0
    for num in nums:
        xor_result ^= num
    
    # Find the rightmost set bit
    mask = 1
    while (xor_result & mask) == 0:
        mask <<= 1
    
    # Divide numbers into two groups
    num1 = num2 = 0
    for num in nums:
        if num & mask:
            num1 ^= num
        else:
            num2 ^= num
    
    return [num1, num2]

# Example usage
if __name__ == "__main__":
    def test_single_number():
        """Test the single number implementations."""
        # Test cases
        test_cases = [
            ([4, 1, 2, 1, 2], "Standard case"),
            ([1, 1, 2, 2, 3], "Single at end"),
            ([1], "Single element"),
            ([1, 1, 2, 2, 3, 3, 4], "Single in middle"),
        ]
        
        for nums, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"nums = {nums}")
            
            # Test single number
            result = single_number(nums)
            print(f"Single number: {result}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. XOR Properties:")
            print("   - a ^ a = 0 (number XORed with itself is 0)")
            print("   - a ^ 0 = a (number XORed with 0 is the number)")
            print("   - XOR is associative and commutative")
            print("2. Algorithm:")
            print("   - XOR all numbers together")
            print("   - Pairs of same numbers cancel out")
            print("   - Result is the single number")
            print()
    
    def test_single_number_ii():
        """Test the single number II implementation."""
        # Test cases
        test_cases = [
            ([2, 2, 3, 2], "Standard case"),
            ([0, 1, 0, 1, 0, 1, 99], "Single at end"),
            ([1, 1, 1, 2], "Single at end"),
            ([1, 1, 1], "No single number"),
        ]
        
        for nums, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"nums = {nums}")
            
            # Test single number II
            result = single_number_ii(nums)
            print(f"Single number: {result}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. Bit Counting:")
            print("   - Use two variables to track bit counts")
            print("   - ones: bits that appear once")
            print("   - twos: bits that appear twice")
            print("2. Algorithm:")
            print("   - For each number, update ones and twos")
            print("   - When a bit appears three times, it's cleared")
            print("   - Final ones value is the single number")
            print()
    
    def test_single_number_iii():
        """Test the single number III implementation."""
        # Test cases
        test_cases = [
            ([1, 2, 1, 3, 2, 5], "Standard case"),
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7], "Two singles at end"),
            ([1, 1, 2, 2], "No single numbers"),
            ([1, 2], "Two singles only"),
        ]
        
        for nums, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"nums = {nums}")
            
            # Test single number III
            result = single_number_iii(nums)
            print(f"Single numbers: {result}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. XOR Properties:")
            print("   - XOR all numbers to get xor of two single numbers")
            print("   - Find a set bit in the XOR result")
            print("2. Algorithm:")
            print("   - Use the set bit to divide numbers into two groups")
            print("   - Each group contains one single number")
            print("   - XOR each group separately to get the singles")
            print()
    
    # Run all tests
    test_single_number()
    test_single_number_ii()
    test_single_number_iii() 