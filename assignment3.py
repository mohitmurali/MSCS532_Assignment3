import random
import timeit
import matplotlib.pyplot as plt
import sys

# Increase recursion limit for Deterministic Quicksort on large inputs
sys.setrecursionlimit(11000)

# Part 1: Randomized Quicksort
def randomized_partition(arr, low, high):
    """Partition the subarray arr[low:high+1] around a random pivot."""
    pivot_idx = random.randint(low, high)  # Choose random pivot
    arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]  # Move pivot to end
    pivot = arr[high]
    i = low - 1  # Index of smaller element
    for j in range(low, high):
        if arr[j] <= pivot:  # If current element is smaller or equal to pivot
            i += 1
            arr[i], arr[j] = arr[j], arr[i]  # Swap elements
    arr[i + 1], arr[high] = arr[high], arr[i + 1]  # Place pivot in correct position
    return i + 1

def randomized_quicksort(arr, low=None, high=None):
    """Sort array using Randomized Quicksort with average-case O(n log n).
    
    Args:
        arr: List to be sorted.
        low: Starting index of subarray (default: 0).
        high: Ending index of subarray (default: len(arr)-1).
    
    Returns:
        Sorted array (in-place).
    """
    if low is None:
        low = 0
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot_idx = randomized_partition(arr, low, high)  # Partition around pivot
        randomized_quicksort(arr, low, pivot_idx - 1)  # Sort left subarray
        randomized_quicksort(arr, pivot_idx + 1, high)  # Sort right subarray
    return arr

# Deterministic Quicksort
def deterministic_partition(arr, low, high):
    """Partition the subarray arr[low:high+1] around the first element."""
    pivot = arr[low]
    i = low + 1  # Index of smaller element
    for j in range(low + 1, high + 1):
        if arr[j] <= pivot:  # If current element is smaller or equal to pivot
            arr[i], arr[j] = arr[j], arr[i]  # Swap elements
            i += 1
    arr[low], arr[i - 1] = arr[i - 1], arr[low]  # Place pivot in correct position
    return i - 1

def deterministic_quicksort(arr, low=None, high=None):
    """Sort array using Deterministic Quicksort with first element as pivot.
    
    Args:
        arr: List to be sorted.
        low: Starting index of subarray (default: 0).
        high: Ending index of subarray (default: len(arr)-1).
    
    Returns:
        Sorted array (in-place).
        
    Note:
        May require increased recursion limit (sys.setrecursionlimit) for large
        sorted or reverse-sorted arrays due to O(n^2) worst-case behavior.
    """
    if low is None:
        low = 0
    if high is None:
        high = len(arr) - 1
    if low < high:
        pivot_idx = deterministic_partition(arr, low, high)  # Partition around pivot
        deterministic_quicksort(arr, low, pivot_idx - 1)  # Sort left subarray
        deterministic_quicksort(arr, pivot_idx + 1, high)  # Sort right subarray
    return arr

# Empirical Comparison
def generate_arrays(n):
    """Generate test arrays of size n with different distributions."""
    return {
        'random': [random.randint(0, 1000) for _ in range(n)],
        'sorted': list(range(n)),
        'reverse_sorted': list(range(n-1, -1, -1)),
        'repeated': [random.randint(0, 5) for _ in range(n)]
    }

sizes = [100, 1000, 10000]
results = {}

for n in sizes:
    arrays = generate_arrays(n)
    for arr_type, arr in arrays.items():
        arr_copy1 = arr.copy()
        arr_copy2 = arr.copy()
        time_rand = timeit.timeit(lambda: randomized_quicksort(arr_copy1, 0, len(arr_copy1)-1), number=5) / 5
        time_det = timeit.timeit(lambda: deterministic_quicksort(arr_copy2, 0, len(arr_copy2)-1), number=5) / 5
        results[(n, arr_type)] = (time_rand, time_det)

# Plotting
for arr_type in ['random', 'sorted', 'reverse_sorted', 'repeated']:
    plt.figure(figsize=(8, 6))
    rand_times = [results[(n, arr_type)][0] for n in sizes]
    det_times = [results[(n, arr_type)][1] for n in sizes]
    plt.plot(sizes, rand_times, 'ro-', label='Randomized Quicksort')
    plt.plot(sizes, det_times, 'bo-', label='Deterministic Quicksort')
    plt.title(f'Quicksort Comparison on {arr_type.capitalize()} Arrays')
    plt.xlabel('Array Size (n)')
    plt.ylabel('Average Time (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'quicksort_{arr_type}.png')
    plt.close()

# Part 2: Hash Table with Chaining
class HashTable:
    """Hash table with chaining for collision resolution."""
    
    def __init__(self, m=10):
        """Initialize hash table with m slots.
        
        Args:
            m: Number of slots (default: 10).
        """
        self.m = m
        self.table = [[] for _ in range(m)]  # List of lists for chaining
        self.p = 1000000007  # Large prime for hashing
        self.a = random.randint(1, self.p - 1)  # Random hash parameter
        self.b = random.randint(0, self.p - 1)  # Random hash parameter

    def hash_function(self, k):
        """Compute hash value for key k using universal hashing.
        
        Args:
            k: Key to hash.
        
        Returns:
            Integer slot index.
        """
        return ((self.a * k + self.b) % self.p) % self.m

    def insert(self, k, v):
        """Insert key-value pair into hash table.
        
        Args:
            k: Key.
            v: Value.
        """
        slot = self.hash_function(k)
        for i, (key, _) in enumerate(self.table[slot]):
            if key == k:  # Update existing key
                self.table[slot][i] = (k, v)
                return
        self.table[slot].append((k, v))  # Add new key-value pair

    def search(self, k):
        """Search for value associated with key k.
        
        Args:
            k: Key to search.
        
        Returns:
            Value if found, else None.
        """
        slot = self.hash_function(k)
        for key, value in self.table[slot]:
            if key == k:
                return value
        return None

    def delete(self, k):
        """Delete key-value pair with key k.
        
        Args:
            k: Key to delete.
        """
        slot = self.hash_function(k)
        for i, (key, _) in enumerate(self.table[slot]):
            if key == k:
                del self.table[slot][i]  # Remove key-value pair
                return

# Example Usage
if __name__ == "__main__":
    # Test Quicksort
    test_arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    print("Randomized Quicksort:", randomized_quicksort(test_arr.copy()))
    print("Deterministic Quicksort:", deterministic_quicksort(test_arr.copy()))

    # Test Hash Table
    ht = HashTable(5)
    ht.insert(1, "one")
    ht.insert(6, "six")
    print("Search 1:", ht.search(1))
    ht.delete(1)
    print("Search 1 after delete:", ht.search(1))