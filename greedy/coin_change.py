from typing import List, Dict, Tuple
from collections import defaultdict

def coin_change_greedy(coins: List[int], amount: int) -> List[int]:
    """
    Find the minimum number of coins needed to make up the amount using a greedy approach.
    Note: This only works for certain coin denominations (e.g., standard US coins).
    
    Example:
    Input: coins = [1, 5, 10, 25], amount = 37
    Output: [25, 10, 1, 1]
    
    Time Complexity: O(n) where n is the number of coins
    Space Complexity: O(1) excluding the output
    """
    if not coins or amount < 0:
        return []
    
    # Sort coins in descending order
    coins.sort(reverse=True)
    result = []
    remaining = amount
    
    # Use largest coins first
    for coin in coins:
        while remaining >= coin:
            result.append(coin)
            remaining -= coin
    
    return result if remaining == 0 else []

def coin_change_dp(coins: List[int], amount: int) -> Tuple[int, List[int]]:
    """
    Find the minimum number of coins needed to make up the amount using dynamic programming.
    Works for any coin denominations.
    
    Time Complexity: O(n * amount) where n is the number of coins
    Space Complexity: O(amount)
    """
    if not coins or amount < 0:
        return -1, []
    
    # Initialize DP array
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    # Store the last coin used for each amount
    last_coin = [-1] * (amount + 1)
    
    # Fill DP array
    for i in range(1, amount + 1):
        for coin in coins:
            if i >= coin and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                last_coin[i] = coin
    
    if dp[amount] == float('inf'):
        return -1, []
    
    # Reconstruct the solution
    result = []
    remaining = amount
    while remaining > 0:
        coin = last_coin[remaining]
        result.append(coin)
        remaining -= coin
    
    return dp[amount], result

def coin_change_combinations(coins: List[int], amount: int) -> int:
    """
    Count the number of different ways to make up the amount using the given coins.
    
    Example:
    Input: coins = [1, 2, 5], amount = 5
    Output: 4 (ways: [1,1,1,1,1], [1,1,1,2], [1,2,2], [5])
    
    Time Complexity: O(n * amount)
    Space Complexity: O(amount)
    """
    if not coins or amount < 0:
        return 0
    
    # Initialize DP array
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    # Fill DP array
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    
    return dp[amount]

def coin_change_min_coins_with_constraints(coins: List[int], amount: int, 
                                         constraints: Dict[int, int]) -> Tuple[int, List[int]]:
    """
    Find the minimum number of coins needed to make up the amount with constraints on coin usage.
    
    Example:
    Input: coins = [1, 2, 5], amount = 11, constraints = {1: 2, 2: 3, 5: 1}
    Output: (4, [5, 2, 2, 2])
    
    Time Complexity: O(n * amount * max_constraint)
    Space Complexity: O(amount * max_constraint)
    """
    if not coins or amount < 0:
        return -1, []
    
    # Initialize DP array
    max_constraint = max(constraints.values())
    dp = [[float('inf')] * (amount + 1) for _ in range(max_constraint + 1)]
    dp[0][0] = 0
    
    # Store the last coin used for each amount and count
    last_coin = [[-1] * (amount + 1) for _ in range(max_constraint + 1)]
    
    # Fill DP array
    for i in range(max_constraint + 1):
        for j in range(amount + 1):
            if dp[i][j] == float('inf'):
                continue
            
            for coin in coins:
                if (j + coin <= amount and 
                    i + 1 <= constraints.get(coin, float('inf')) and 
                    dp[i + 1][j + coin] > dp[i][j] + 1):
                    dp[i + 1][j + coin] = dp[i][j] + 1
                    last_coin[i + 1][j + coin] = coin
    
    # Find minimum coins needed
    min_coins = float('inf')
    min_count = 0
    for i in range(max_constraint + 1):
        if dp[i][amount] < min_coins:
            min_coins = dp[i][amount]
            min_count = i
    
    if min_coins == float('inf'):
        return -1, []
    
    # Reconstruct the solution
    result = []
    remaining = amount
    count = min_count
    
    while remaining > 0:
        coin = last_coin[count][remaining]
        result.append(coin)
        remaining -= coin
        count -= 1
    
    return min_coins, result

# Example usage
if __name__ == "__main__":
    def test_coin_change():
        """Test the coin change implementations."""
        # Test cases
        test_cases = [
            ([1, 5, 10, 25], 37, "US coins"),
            ([1, 2, 5], 11, "Standard case"),
            ([2, 5, 10], 13, "No solution"),
            ([1], 5, "Single coin"),
            ([], 5, "No coins"),
        ]
        
        for coins, amount, name in test_cases:
            print(f"\nTesting {name}:")
            print(f"Coins: {coins}")
            print(f"Amount: {amount}")
            
            # Test greedy approach
            result1 = coin_change_greedy(coins, amount)
            print("\nGreedy Approach:")
            print(f"Coins used: {result1}")
            print(f"Number of coins: {len(result1)}")
            
            # Test DP approach
            num_coins, result2 = coin_change_dp(coins, amount)
            print("\nDP Approach:")
            print(f"Coins used: {result2}")
            print(f"Number of coins: {num_coins}")
            
            # Test combinations
            num_ways = coin_change_combinations(coins, amount)
            print("\nNumber of Combinations:")
            print(f"Ways to make {amount}: {num_ways}")
            
            # Test with constraints
            constraints = {coin: 3 for coin in coins}  # Allow at most 3 of each coin
            num_coins_const, result3 = coin_change_min_coins_with_constraints(
                coins, amount, constraints)
            print("\nWith Constraints (max 3 of each coin):")
            print(f"Coins used: {result3}")
            print(f"Number of coins: {num_coins_const}")
            
            # Print explanation
            print("\nExplanation:")
            print("1. Greedy Approach:")
            print("   - Always use largest coin first")
            print("   - Only works for certain coin denominations")
            print("   - O(n) time complexity")
            print("2. DP Approach:")
            print("   - Works for any coin denominations")
            print("   - O(n * amount) time complexity")
            print("   - O(amount) space complexity")
            print("3. Combinations:")
            print("   - Count different ways to make amount")
            print("   - Same time complexity as DP")
            print("4. With Constraints:")
            print("   - Handle limits on coin usage")
            print("   - O(n * amount * max_constraint) time complexity")
            print("   - O(amount * max_constraint) space complexity")
            print()
    
    # Run tests
    test_coin_change() 