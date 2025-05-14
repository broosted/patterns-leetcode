from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from heapq import heappush, heappop

@dataclass
class User:
    """Represents a user in the social network."""
    id: int
    name: str
    interests: List[str] = None
    location: str = ""
    age: int = 0

class SocialNetwork:
    """Social network implementation using union-find for connectivity."""
    
    def __init__(self, n: int):
        """
        Initialize social network with n users.
        
        Args:
            n: Number of users
        """
        self.uf = UnionFind(n)
        self.users: Dict[int, User] = {}
        self.friends: Dict[int, Set[int]] = defaultdict(set)
        self.interests: Dict[str, Set[int]] = defaultdict(set)
        self.locations: Dict[str, Set[int]] = defaultdict(set)
    
    def add_user(self, user: User) -> None:
        """
        Add a user to the network.
        
        Time Complexity: O(1)
        """
        self.users[user.id] = user
        if user.interests:
            for interest in user.interests:
                self.interests[interest].add(user.id)
        if user.location:
            self.locations[user.location].add(user.id)
    
    def add_friendship(self, user1: int, user2: int) -> bool:
        """
        Add friendship between two users.
        Returns True if friendship was added, False if already friends.
        
        Time Complexity: O(α(n)) amortized
        """
        if user1 not in self.users or user2 not in self.users:
            return False
        
        if user2 in self.friends[user1]:
            return False
        
        self.friends[user1].add(user2)
        self.friends[user2].add(user1)
        self.uf.union(user1, user2)
        return True
    
    def remove_friendship(self, user1: int, user2: int) -> bool:
        """
        Remove friendship between two users.
        Returns True if friendship was removed, False if not friends.
        
        Time Complexity: O(α(n) + F) where F is number of friends
        """
        if user1 not in self.users or user2 not in self.users:
            return False
        
        if user2 not in self.friends[user1]:
            return False
        
        self.friends[user1].remove(user2)
        self.friends[user2].remove(user1)
        
        # Rebuild connectivity without this friendship
        self.uf = UnionFind(len(self.users))
        for u in self.users:
            for v in self.friends[u]:
                if u < v:  # Process each edge once
                    self.uf.union(u, v)
        
        return True
    
    def are_connected(self, user1: int, user2: int) -> bool:
        """
        Check if two users are connected (friends or friends of friends).
        
        Time Complexity: O(α(n)) amortized
        """
        if user1 not in self.users or user2 not in self.users:
            return False
        return self.uf.connected(user1, user2)
    
    def get_connected_users(self, user_id: int) -> List[int]:
        """
        Get all users connected to a given user.
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if user_id not in self.users:
            return []
        
        return [u for u in self.users if self.uf.connected(user_id, u)]
    
    def get_friend_suggestions(self, user_id: int, max_suggestions: int = 5) -> List[Tuple[int, float]]:
        """
        Get friend suggestions based on mutual friends, interests, and location.
        Returns list of (user_id, score) tuples sorted by score.
        
        Time Complexity: O(n * log n)
        Space Complexity: O(n)
        """
        if user_id not in self.users:
            return []
        
        user = self.users[user_id]
        scores = defaultdict(float)
        
        # Score based on mutual friends
        for friend in self.friends[user_id]:
            for friend_of_friend in self.friends[friend]:
                if (friend_of_friend != user_id and 
                    friend_of_friend not in self.friends[user_id]):
                    scores[friend_of_friend] += 1.0
        
        # Score based on common interests
        if user.interests:
            for interest in user.interests:
                for other_user in self.interests[interest]:
                    if other_user != user_id and other_user not in self.friends[user_id]:
                        scores[other_user] += 0.5
        
        # Score based on location
        if user.location:
            for other_user in self.locations[user.location]:
                if other_user != user_id and other_user not in self.friends[user_id]:
                    scores[other_user] += 0.3
        
        # Get top suggestions
        suggestions = []
        for other_user, score in scores.items():
            if score > 0:
                heappush(suggestions, (score, other_user))
                if len(suggestions) > max_suggestions:
                    heappop(suggestions)
        
        return [(user_id, score) for score, user_id in sorted(suggestions, reverse=True)]
    
    def get_communities(self) -> List[List[int]]:
        """
        Get all communities in the network (connected components).
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        return self.uf.get_components()
    
    def get_largest_community(self) -> List[int]:
        """
        Get the largest community in the network.
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        communities = self.get_communities()
        return max(communities, key=len) if communities else []

class UnionFind:
    """Union-Find data structure with path compression and union by rank."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.count = n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        self.count -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)
    
    def get_components(self) -> List[List[int]]:
        components = defaultdict(list)
        for i in range(len(self.parent)):
            components[self.find(i)].append(i)
        return list(components.values())

# Example usage
if __name__ == "__main__":
    def test_social_network():
        """Test the social network implementations."""
        # Create network
        network = SocialNetwork(6)
        
        # Add users
        users = [
            User(0, "Alice", ["music", "movies"], "New York", 25),
            User(1, "Bob", ["sports", "music"], "Boston", 30),
            User(2, "Charlie", ["movies", "books"], "New York", 28),
            User(3, "David", ["sports", "books"], "Chicago", 35),
            User(4, "Eve", ["music", "books"], "Boston", 27),
            User(5, "Frank", ["sports", "movies"], "Chicago", 32)
        ]
        
        for user in users:
            network.add_user(user)
        
        # Add friendships
        friendships = [(0,1), (1,2), (2,3), (3,4), (4,5), (0,2)]
        for u, v in friendships:
            network.add_friendship(u, v)
        
        print("\nTesting Social Network Operations:")
        print("Users:")
        for user in users:
            print(f"- {user.name} (ID: {user.id}): {user.interests}, {user.location}")
        print(f"Friendships: {friendships}")
        
        # Test connectivity
        print("\nConnectivity:")
        print(f"Alice and Frank connected: {network.are_connected(0, 5)}")
        print(f"Alice and Charlie connected: {network.are_connected(0, 2)}")
        
        # Test friend suggestions
        print("\nFriend Suggestions for Alice:")
        suggestions = network.get_friend_suggestions(0)
        for user_id, score in suggestions:
            user = network.users[user_id]
            print(f"- {user.name}: score {score:.2f}")
        
        # Test communities
        communities = network.get_communities()
        print("\nCommunities:")
        for i, community in enumerate(communities):
            print(f"Community {i+1}: {[network.users[u].name for u in community]}")
        
        # Test largest community
        largest = network.get_largest_community()
        print("\nLargest Community:")
        print(f"Users: {[network.users[u].name for u in largest]}")
        
        # Print explanation
        print("\nExplanation:")
        print("1. Social Network Structure:")
        print("   - Uses union-find for efficient connectivity queries")
        print("   - Tracks friendships, interests, and locations")
        print("2. Friend Suggestions:")
        print("   - Based on mutual friends, interests, and location")
        print("   - O(n * log n) time complexity")
        print("3. Community Detection:")
        print("   - Uses connected components from union-find")
        print("   - O(n) time complexity")
        print("4. Operations:")
        print("   - Add/remove friendship: O(α(n)) amortized")
        print("   - Check connectivity: O(α(n)) amortized")
        print("   - Get suggestions: O(n * log n)")
        print()
    
    # Run tests
    test_social_network() 