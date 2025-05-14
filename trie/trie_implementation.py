from typing import List, Dict, Optional
from collections import defaultdict

class TrieNode:
    """Node for trie data structure."""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = defaultdict(TrieNode)
        self.is_word: bool = False
        self.word_count: int = 0  # Count of words with this prefix
        self.word: str = ""  # Store the complete word at leaf nodes

class Trie:
    """Trie data structure implementation."""
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """
        Insert a word into the trie.
        
        Time Complexity: O(m) where m is the length of the word
        Space Complexity: O(m)
        """
        node = self.root
        for char in word:
            node = node.children[char]
            node.word_count += 1
        node.is_word = True
        node.word = word
    
    def search(self, word: str) -> bool:
        """
        Search for a word in the trie.
        
        Time Complexity: O(m) where m is the length of the word
        Space Complexity: O(1)
        """
        node = self._find_node(word)
        return node is not None and node.is_word
    
    def starts_with(self, prefix: str) -> bool:
        """
        Check if any word in the trie starts with the given prefix.
        
        Time Complexity: O(m) where m is the length of the prefix
        Space Complexity: O(1)
        """
        return self._find_node(prefix) is not None
    
    def get_words_with_prefix(self, prefix: str) -> List[str]:
        """
        Get all words in the trie that start with the given prefix.
        
        Time Complexity: O(m + n) where m is prefix length and n is number of words
        Space Complexity: O(n) for storing results
        """
        node = self._find_node(prefix)
        if not node:
            return []
        
        result = []
        self._collect_words(node, result)
        return result
    
    def delete(self, word: str) -> bool:
        """
        Delete a word from the trie.
        
        Time Complexity: O(m) where m is the length of the word
        Space Complexity: O(1)
        """
        if not self.search(word):
            return False
        
        node = self.root
        stack = []
        
        # Find the path to the word
        for char in word:
            stack.append((char, node))
            node = node.children[char]
            node.word_count -= 1
        
        # Mark as not a word
        node.is_word = False
        node.word = ""
        
        # Remove nodes if they have no children and are not words
        while stack and not node.is_word and not node.children:
            char, parent = stack.pop()
            del parent.children[char]
            node = parent
        
        return True
    
    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Find the node corresponding to the prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def _collect_words(self, node: TrieNode, result: List[str]) -> None:
        """Collect all words from the given node."""
        if node.is_word:
            result.append(node.word)
        for child in node.children.values():
            self._collect_words(child, result)

class WordDictionary:
    """
    Word dictionary that supports adding words and searching with wildcards.
    """
    def __init__(self):
        self.trie = Trie()
    
    def add_word(self, word: str) -> None:
        """
        Add a word to the dictionary.
        
        Time Complexity: O(m) where m is the length of the word
        Space Complexity: O(m)
        """
        self.trie.insert(word)
    
    def search(self, word: str) -> bool:
        """
        Search for a word in the dictionary.
        Word may contain dots '.' where dots can be matched with any letter.
        
        Time Complexity: O(26^m) in worst case where m is the length of the word
        Space Complexity: O(m) for recursion stack
        """
        def search_helper(node: TrieNode, word: str, index: int) -> bool:
            if index == len(word):
                return node.is_word
            
            char = word[index]
            if char == '.':
                # Try all possible characters
                for child in node.children.values():
                    if search_helper(child, word, index + 1):
                        return True
                return False
            else:
                if char not in node.children:
                    return False
                return search_helper(node.children[char], word, index + 1)
        
        return search_helper(self.trie.root, word, 0)

# Example usage
if __name__ == "__main__":
    def test_trie():
        """Test the trie implementation."""
        trie = Trie()
        
        # Test cases
        words = ["apple", "app", "application", "banana", "band", "bandana"]
        print("\nTesting Trie:")
        print("Inserting words:", words)
        
        # Insert words
        for word in words:
            trie.insert(word)
        
        # Test search
        test_words = ["app", "apple", "application", "ban", "band", "xyz"]
        print("\nTesting Search:")
        for word in test_words:
            print(f"Search '{word}': {trie.search(word)}")
        
        # Test prefix
        prefixes = ["app", "ban", "xyz"]
        print("\nTesting Prefix:")
        for prefix in prefixes:
            print(f"Starts with '{prefix}': {trie.starts_with(prefix)}")
            print(f"Words with prefix '{prefix}': {trie.get_words_with_prefix(prefix)}")
        
        # Test delete
        delete_words = ["app", "banana", "xyz"]
        print("\nTesting Delete:")
        for word in delete_words:
            print(f"Delete '{word}': {trie.delete(word)}")
            print(f"Search after delete '{word}': {trie.search(word)}")
        
        # Print explanation
        print("\nExplanation:")
        print("1. Trie Structure:")
        print("   - Each node represents a character")
        print("   - Path from root to leaf represents a word")
        print("   - Nodes store word count and complete word")
        print("2. Operations:")
        print("   - Insert: O(m) time, O(m) space")
        print("   - Search: O(m) time, O(1) space")
        print("   - Prefix: O(m) time, O(1) space")
        print("   - Delete: O(m) time, O(1) space")
        print()
    
    def test_word_dictionary():
        """Test the word dictionary implementation."""
        dictionary = WordDictionary()
        
        # Test cases
        words = ["bad", "dad", "mad", "pad", "rad"]
        print("\nTesting Word Dictionary:")
        print("Adding words:", words)
        
        # Add words
        for word in words:
            dictionary.add_word(word)
        
        # Test search
        test_patterns = [
            ("bad", "Exact match"),
            (".ad", "Single wildcard"),
            ("b..", "Multiple wildcards"),
            ("xyz", "Non-existent word"),
            ("...", "All wildcards")
        ]
        
        print("\nTesting Search with Wildcards:")
        for pattern, name in test_patterns:
            print(f"{name} - Search '{pattern}': {dictionary.search(pattern)}")
        
        # Print explanation
        print("\nExplanation:")
        print("1. Word Dictionary:")
        print("   - Uses trie for efficient word storage")
        print("   - Supports wildcard matching with '.'")
        print("2. Operations:")
        print("   - Add Word: O(m) time, O(m) space")
        print("   - Search: O(26^m) worst case for wildcards")
        print("   - Space: O(n * m) where n is number of words")
        print()
    
    # Run tests
    test_trie()
    test_word_dictionary() 