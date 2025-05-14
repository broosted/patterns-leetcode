from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappush, heappop

@dataclass
class TrieNode:
    """Node for trie data structure with frequency count."""
    children: Dict[str, 'TrieNode'] = None
    is_word: bool = False
    word: str = ""
    frequency: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = defaultdict(TrieNode)

class AutocompleteSystem:
    """
    Autocomplete system that suggests words based on prefix and frequency.
    """
    def __init__(self, sentences: List[str], times: List[int]):
        """
        Initialize the autocomplete system with historical data.
        
        Time Complexity: O(n * m) where n is number of sentences and m is average length
        Space Complexity: O(n * m)
        """
        self.root = TrieNode()
        self.current_input = ""
        
        # Build trie with historical data
        for sentence, time in zip(sentences, times):
            self._insert(sentence, time)
    
    def input(self, c: str) -> List[str]:
        """
        Process input character and return top 3 suggestions.
        '#' indicates end of sentence.
        
        Time Complexity: O(m + k * log k) where m is input length and k is number of matches
        Space Complexity: O(k) for storing matches
        """
        if c == '#':
            # End of sentence, add to trie
            if self.current_input:
                self._insert(self.current_input, 1)
                self.current_input = ""
            return []
        
        self.current_input += c
        return self._get_suggestions(self.current_input, 3)
    
    def _insert(self, sentence: str, frequency: int) -> None:
        """Insert a sentence into the trie with its frequency."""
        node = self.root
        for char in sentence:
            node = node.children[char]
        node.is_word = True
        node.word = sentence
        node.frequency += frequency
    
    def _get_suggestions(self, prefix: str, k: int) -> List[str]:
        """Get top k suggestions for the given prefix."""
        # Find the node for the prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect all words with this prefix
        matches = []
        self._collect_words(node, matches)
        
        # Sort by frequency (descending) and lexicographically
        matches.sort(key=lambda x: (-x[1], x[0]))
        return [word for word, _ in matches[:k]]
    
    def _collect_words(self, node: TrieNode, matches: List[Tuple[str, int]]) -> None:
        """Collect all words from the given node with their frequencies."""
        if node.is_word:
            matches.append((node.word, node.frequency))
        for child in node.children.values():
            self._collect_words(child, matches)

class AutocompleteWithHotness:
    """
    Enhanced autocomplete system that considers both frequency and recency.
    """
    def __init__(self, sentences: List[str], times: List[int], decay_factor: float = 0.9):
        """
        Initialize the autocomplete system with hotness scoring.
        
        Args:
            sentences: List of historical sentences
            times: List of timestamps for sentences
            decay_factor: Factor to decay hotness over time (0-1)
        """
        self.root = TrieNode()
        self.current_input = ""
        self.decay_factor = decay_factor
        self.current_time = max(times) if times else 0
        
        # Build trie with historical data
        for sentence, time in zip(sentences, times):
            self._insert(sentence, time)
    
    def input(self, c: str, timestamp: int) -> List[str]:
        """
        Process input character and return top 3 suggestions based on hotness.
        '#' indicates end of sentence.
        
        Time Complexity: O(m + k * log k) where m is input length and k is number of matches
        Space Complexity: O(k)
        """
        # Update hotness scores
        self._update_hotness(timestamp)
        
        if c == '#':
            if self.current_input:
                self._insert(self.current_input, timestamp)
                self.current_input = ""
            return []
        
        self.current_input += c
        return self._get_hot_suggestions(self.current_input, 3)
    
    def _insert(self, sentence: str, timestamp: int) -> None:
        """Insert a sentence with its timestamp."""
        node = self.root
        for char in sentence:
            node = node.children[char]
        node.is_word = True
        node.word = sentence
        node.frequency = timestamp  # Store timestamp as frequency
    
    def _update_hotness(self, current_time: int) -> None:
        """Update hotness scores based on time decay."""
        def update_node(node: TrieNode) -> None:
            if node.is_word:
                # Decay the frequency based on time difference
                time_diff = current_time - node.frequency
                node.frequency *= (self.decay_factor ** time_diff)
            for child in node.children.values():
                update_node(child)
        
        update_node(self.root)
        self.current_time = current_time
    
    def _get_hot_suggestions(self, prefix: str, k: int) -> List[str]:
        """Get top k suggestions based on hotness score."""
        # Find the node for the prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Use min heap to get top k suggestions
        heap = []
        self._collect_hot_words(node, heap, k)
        
        # Convert heap to sorted list
        result = []
        while heap:
            result.append(heappop(heap)[1])
        return result[::-1]  # Reverse to get descending order
    
    def _collect_hot_words(self, node: TrieNode, heap: List[Tuple[float, str]], k: int) -> None:
        """Collect words with their hotness scores using a min heap."""
        if node.is_word:
            # Use negative frequency for max heap behavior
            heappush(heap, (-node.frequency, node.word))
            if len(heap) > k:
                heappop(heap)
        for child in node.children.values():
            self._collect_hot_words(child, heap, k)

# Example usage
if __name__ == "__main__":
    def test_autocomplete():
        """Test the basic autocomplete system."""
        sentences = [
            "i love you", "island", "ironman", "i love leetcode",
            "i love coding", "i love programming", "i love python"
        ]
        times = [5, 3, 2, 2, 1, 1, 1]
        
        print("\nTesting Basic Autocomplete:")
        print("Initial sentences:", sentences)
        print("Initial times:", times)
        
        ac = AutocompleteSystem(sentences, times)
        
        # Test inputs
        test_inputs = [
            ("i", "Single character"),
            ("i ", "With space"),
            ("i l", "Partial word"),
            ("i love", "Complete word"),
            ("#", "End of sentence"),
            ("i", "After new input")
        ]
        
        for input_char, name in test_inputs:
            print(f"\n{name} - Input '{input_char}':")
            suggestions = ac.input(input_char)
            print(f"Suggestions: {suggestions}")
        
        # Print explanation
        print("\nExplanation:")
        print("1. Basic Autocomplete:")
        print("   - Uses trie for efficient prefix matching")
        print("   - Stores frequency for each sentence")
        print("   - Returns top 3 suggestions by frequency")
        print("2. Operations:")
        print("   - Insert: O(m) time, O(m) space")
        print("   - Input: O(m + k * log k) time")
        print("   - Space: O(n * m) where n is number of sentences")
        print()
    
    def test_hot_autocomplete():
        """Test the hotness-based autocomplete system."""
        sentences = [
            "python programming", "python tutorial", "python for beginners",
            "java programming", "java tutorial", "java for beginners"
        ]
        times = [100, 90, 80, 70, 60, 50]  # Timestamps
        
        print("\nTesting Hotness-based Autocomplete:")
        print("Initial sentences:", sentences)
        print("Initial times:", times)
        
        ac = AutocompleteWithHotness(sentences, times, decay_factor=0.9)
        
        # Test inputs with timestamps
        test_inputs = [
            ("p", 110, "Single character"),
            ("py", 120, "Partial word"),
            ("python", 130, "Complete word"),
            ("#", 140, "End of sentence"),
            ("p", 150, "After new input")
        ]
        
        for input_char, timestamp, name in test_inputs:
            print(f"\n{name} - Input '{input_char}' at time {timestamp}:")
            suggestions = ac.input(input_char, timestamp)
            print(f"Suggestions: {suggestions}")
        
        # Print explanation
        print("\nExplanation:")
        print("1. Hotness-based Autocomplete:")
        print("   - Considers both frequency and recency")
        print("   - Uses time decay factor for hotness")
        print("   - Returns top 3 suggestions by hotness")
        print("2. Operations:")
        print("   - Insert: O(m) time, O(m) space")
        print("   - Input: O(m + k * log k) time")
        print("   - Update Hotness: O(n) time")
        print("   - Space: O(n * m) where n is number of sentences")
        print()
    
    # Run tests
    test_autocomplete()
    test_hot_autocomplete() 