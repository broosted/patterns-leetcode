import threading
import time
import queue
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

@dataclass
class QueueStats:
    """Statistics for the rate-limited queue."""
    total_processed: int = 0
    total_dropped: int = 0
    avg_wait_time: float = 0.0
    max_wait_time: float = 0.0
    current_size: int = 0
    peak_size: int = 0
    processing_rate: float = 0.0  # items per second

class RateLimitedQueue:
    """
    Thread-safe producer-consumer queue with rate limiting and monitoring.
    
    Features:
    - Rate limiting for both producers and consumers
    - Priority-based processing
    - Queue size monitoring
    - Performance statistics
    - Graceful shutdown
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        producer_rate: float = 100.0,  # items per second
        consumer_rate: float = 50.0,   # items per second
        num_consumers: int = 2,
        priority_levels: int = 3
    ):
        """
        Initialize the rate-limited queue.
        
        Args:
            max_size: Maximum queue size
            producer_rate: Maximum items per second for producers
            consumer_rate: Maximum items per second for consumers
            num_consumers: Number of consumer threads
            priority_levels: Number of priority levels (1 to N)
        """
        self.max_size = max_size
        self.producer_rate = producer_rate
        self.consumer_rate = consumer_rate
        self.num_consumers = num_consumers
        self.priority_levels = priority_levels
        
        # Queues for each priority level
        self.queues: List[queue.Queue] = [
            queue.Queue(maxsize=max_size // priority_levels)
            for _ in range(priority_levels)
        ]
        
        # Rate limiting
        self.producer_tokens = producer_rate
        self.consumer_tokens = consumer_rate
        self.last_producer_update = time.time()
        self.last_consumer_update = time.time()
        self.rate_lock = threading.Lock()
        
        # Statistics
        self.stats = QueueStats()
        self.stats_lock = threading.Lock()
        self.wait_times: deque = deque(maxlen=1000)  # Recent wait times
        
        # Thread management
        self.running = True
        self.consumer_threads: List[threading.Thread] = []
        self.start_consumers()
    
    def start_consumers(self) -> None:
        """Start consumer threads."""
        for i in range(self.num_consumers):
            thread = threading.Thread(
                target=self._consumer_loop,
                name=f"Consumer-{i}",
                daemon=True
            )
            thread.start()
            self.consumer_threads.append(thread)
    
    def put(
        self,
        item: Any,
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Put an item into the queue with specified priority.
        
        Args:
            item: Item to put in queue
            priority: Priority level (0 to priority_levels-1)
            timeout: Maximum time to wait (None for no timeout)
        
        Returns:
            bool: True if item was added, False if dropped
        """
        if not self.running:
            return False
        
        # Validate priority
        priority = max(0, min(priority, self.priority_levels - 1))
        
        # Check rate limit
        with self.rate_lock:
            current_time = time.time()
            time_diff = current_time - self.last_producer_update
            self.producer_tokens = min(
                self.producer_rate,
                self.producer_tokens + time_diff * self.producer_rate
            )
            
            if self.producer_tokens < 1.0:
                with self.stats_lock:
                    self.stats.total_dropped += 1
                return False
            
            self.producer_tokens -= 1.0
            self.last_producer_update = current_time
        
        # Try to put item in queue
        try:
            start_time = time.time()
            self.queues[priority].put(item, timeout=timeout)
            
            # Update statistics
            wait_time = time.time() - start_time
            with self.stats_lock:
                self.stats.current_size = sum(q.qsize() for q in self.queues)
                self.stats.peak_size = max(
                    self.stats.peak_size,
                    self.stats.current_size
                )
                self.wait_times.append(wait_time)
                self.stats.avg_wait_time = sum(self.wait_times) / len(self.wait_times)
                self.stats.max_wait_time = max(
                    self.stats.max_wait_time,
                    wait_time
                )
            
            return True
        
        except queue.Full:
            with self.stats_lock:
                self.stats.total_dropped += 1
            return False
    
    def _consumer_loop(self) -> None:
        """Consumer thread main loop."""
        while self.running:
            # Check rate limit
            with self.rate_lock:
                current_time = time.time()
                time_diff = current_time - self.last_consumer_update
                self.consumer_tokens = min(
                    self.consumer_rate,
                    self.consumer_tokens + time_diff * self.consumer_rate
                )
                
                if self.consumer_tokens < 1.0:
                    time.sleep(0.1)
                    continue
                
                self.consumer_tokens -= 1.0
                self.last_consumer_update = current_time
            
            # Try to get item from highest priority queue
            item = None
            for q in self.queues:
                try:
                    item = q.get_nowait()
                    break
                except queue.Empty:
                    continue
            
            if item is not None:
                # Process item
                try:
                    if callable(item):
                        item()
                    # Update statistics
                    with self.stats_lock:
                        self.stats.total_processed += 1
                        self.stats.current_size = sum(q.qsize() for q in self.queues)
                        # Update processing rate (exponential moving average)
                        alpha = 0.1
                        self.stats.processing_rate = (
                            (1 - alpha) * self.stats.processing_rate +
                            alpha / time_diff
                        )
                except Exception as e:
                    print(f"Error processing item: {e}")
            else:
                # No items to process, sleep briefly
                time.sleep(0.1)
    
    def get_stats(self) -> QueueStats:
        """Get current queue statistics."""
        with self.stats_lock:
            return QueueStats(
                total_processed=self.stats.total_processed,
                total_dropped=self.stats.total_dropped,
                avg_wait_time=self.stats.avg_wait_time,
                max_wait_time=self.stats.max_wait_time,
                current_size=self.stats.current_size,
                peak_size=self.stats.peak_size,
                processing_rate=self.stats.processing_rate
            )
    
    def shutdown(self, timeout: Optional[float] = None) -> None:
        """
        Shutdown the queue and wait for consumers to finish.
        
        Args:
            timeout: Maximum time to wait for consumers (None for no timeout)
        """
        self.running = False
        for thread in self.consumer_threads:
            thread.join(timeout=timeout)

# Example usage
if __name__ == "__main__":
    def test_rate_limited_queue():
        """Test the rate-limited queue implementation."""
        # Create queue with rate limits
        queue = RateLimitedQueue(
            max_size=100,
            producer_rate=50.0,  # 50 items per second
            consumer_rate=20.0,  # 20 items per second
            num_consumers=2,
            priority_levels=3
        )
        
        # Producer function
        def producer(id: int, num_items: int, priority: int):
            for i in range(num_items):
                item = f"Item-{id}-{i}"
                if queue.put(item, priority=priority):
                    print(f"Producer {id} added {item} (priority {priority})")
                else:
                    print(f"Producer {id} dropped {item}")
                time.sleep(random.uniform(0.01, 0.05))
        
        # Start producers
        producers = []
        for i in range(3):
            thread = threading.Thread(
                target=producer,
                args=(i, 20, i % 3),
                name=f"Producer-{i}"
            )
            thread.start()
            producers.append(thread)
        
        # Monitor queue
        try:
            start_time = time.time()
            while any(p.is_alive() for p in producers):
                stats = queue.get_stats()
                print("\nQueue Statistics:")
                print(f"Processed: {stats.total_processed}")
                print(f"Dropped: {stats.total_dropped}")
                print(f"Current Size: {stats.current_size}")
                print(f"Peak Size: {stats.peak_size}")
                print(f"Avg Wait Time: {stats.avg_wait_time:.3f}s")
                print(f"Max Wait Time: {stats.max_wait_time:.3f}s")
                print(f"Processing Rate: {stats.processing_rate:.1f} items/s")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            # Wait for producers
            for p in producers:
                p.join()
            
            # Shutdown queue
            queue.shutdown(timeout=5.0)
            
            # Final statistics
            stats = queue.get_stats()
            print("\nFinal Statistics:")
            print(f"Total Processed: {stats.total_processed}")
            print(f"Total Dropped: {stats.total_dropped}")
            print(f"Peak Size: {stats.peak_size}")
            print(f"Final Processing Rate: {stats.processing_rate:.1f} items/s")
            print(f"Total Time: {time.time() - start_time:.1f}s")
    
    # Run test
    test_rate_limited_queue() 