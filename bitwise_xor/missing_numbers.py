from typing import List, Tuple

def find_missing_number(nums: List[int]) -> int:
    """
    Find the missing number in an array containing n distinct numbers from 0 to n.
    Uses bitwise XOR operation.
    
    Example:
    Input: nums = [3, 0, 1]
    Output: 2
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    n = len(nums)
    result = n  # Start with n since it's not in the array
    
    # XOR all numbers with their indices
    for i, num in enumerate(nums):
        result ^= i ^ num
    
    return result

def find_two_missing_numbers(nums: List[int]) -> Tuple[int, int]:
    """
    Find two missing numbers in an array containing n-2 distinct numbers from 1 to n.
    Uses bitwise XOR and masking.
    
    Example:
    Input: nums = [1, 2, 4, 6]
    Output: (3, 5)
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    n = len(nums) + 2
    # XOR all numbers from 1 to n
    xor_all = 0
    for i in range(1, n + 1):
        xor_all ^= i
    
    # XOR all numbers in array
    xor_array = 0
    for num in nums:
        xor_array ^= num
    
    # XOR of two missing numbers
    xor_missing = xor_all ^ xor_array
    
    # Find the rightmost set bit
    mask = 1
    while (xor_missing & mask) == 0:
        mask <<= 1
    
    # Divide numbers into two groups
    num1 = num2 = 0
    # XOR numbers from 1 to n
    for i in range(1, n + 1):
        if i & mask:
            num1 ^= i
        else:
            num2 ^= i
    
    # XOR numbers in array
    for num in nums:
        if num & mask:
            num1 ^= num
        else:
            num2 ^= num
    
    return (num1, num2) if num1 < num2 else (num2, num1)

def find_duplicate_number(nums: List[int]) -> int:
    """
    Find the duplicate number in an array containing n+1 numbers from 1 to n.
    Uses Floyd's Cycle Finding Algorithm (Tortoise and Hare).
    
    Example:
    Input: nums = [1, 3, 4, 2, 2]
    Output: 2
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # Find the intersection point of the two runners
    tortoise = hare = nums[0]
    while True:
        tortoise = nums[tortoise]
        hare = nums[nums[hare]]
        if tortoise == hare:
            break
    
    # Find the entrance to the cycle
    tortoise = nums[0]
    while tortoise != hare:
        tortoise = nums[tortoise]
        hare = nums[hare]
    
    return hare

# Example usage
if __name__ == "__main__":
    def test_find_missing_number():
        """Test the find missing number implementation."""
        # Test cases
        test_cases = [
            ([3, 0, 1], "Standard case"),
            ([0, 1], "Missing at end"),
            ([1, 2], "Missing at start"),
            ([0], "Single element"),
        ]
        
        for nums, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"nums = {nums}")
            
            # Test find missing number
            result = find_missing_number(nums)
            print(f"Missing number: {result}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. XOR Properties:")
            print("   - a ^ a = 0 (number XORed with itself is 0)")
            print("   - a ^ 0 = a (number XORed with 0 is the number)")
            print("2. Algorithm:")
            print("   - XOR all numbers with their indices")
            print("   - Start with n since it's not in the array")
            print("   - Result is the missing number")
            print()
    
    def test_find_two_missing_numbers():
        """Test the find two missing numbers implementation."""
        # Test cases
        test_cases = [
            ([1, 2, 4, 6], "Standard case"),
            ([1, 3, 5], "Missing at start and end"),
            ([1, 2], "Missing in middle"),
            ([1], "Single element"),
        ]
        
        for nums, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"nums = {nums}")
            
            # Test find two missing numbers
            result = find_two_missing_numbers(nums)
            print(f"Missing numbers: {result}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. XOR Properties:")
            print("   - XOR all numbers from 1 to n")
            print("   - XOR all numbers in array")
            print("2. Algorithm:")
            print("   - Find a set bit in XOR of missing numbers")
            print("   - Use it to divide numbers into two groups")
            print("   - XOR each group separately")
            print()
    
    def test_find_duplicate_number():
        """Test the find duplicate number implementation."""
        # Test cases
        test_cases = [
            ([1, 3, 4, 2, 2], "Standard case"),
            ([1, 1], "Duplicate at start"),
            ([1, 2, 2], "Duplicate at end"),
            ([1, 1, 2], "Duplicate at start"),
        ]
        
        for nums, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"nums = {nums}")
            
            # Test find duplicate number
            result = find_duplicate_number(nums)
            print(f"Duplicate number: {result}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. Floyd's Cycle Finding:")
            print("   - Use two pointers: tortoise and hare")
            print("   - Hare moves twice as fast as tortoise")
            print("2. Algorithm:")
            print("   - Find intersection point of the two runners")
            print("   - Find entrance to the cycle")
            print("   - Entrance is the duplicate number")
            print()
    
    # Run all tests
    test_find_missing_number()
    test_find_two_missing_numbers()
    test_find_duplicate_number() 