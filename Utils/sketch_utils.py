import math
import random


def isPrime(num: int):
    """
    Check if a number is prime.
    
    Args:
        num: Integer to check for primality
        
    Returns:
        bool: True if the number is prime, False otherwise
    """
    # Handle edge cases
    if num < 2:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    
    # Check odd divisors up to square root of num
    border = int(math.sqrt(num)) + 1
    for i in range(3, border, 2):
        if num % i == 0:
            return False
    return True


def calNextPrime(num: int):
    """
    Calculate the next prime number greater than or equal to num.
    
    Args:
        num: Starting number to find the next prime from
        
    Returns:
        int: The next prime number >= num
    """
    # Start with the given number
    candidate = num
    
    # Increment until we find a prime
    while not isPrime(candidate):
        candidate += 1
    return candidate

def AwareHash(data, n: int, hash_val, scale, hardener):
    """
    Computes a hash value using a polynomial rolling hash algorithm.
    
    Args:
        data: Bytes-like object containing the data to hash
        n: Number of bytes to process from data
        hash_val: Initial hash value
        scale: Scaling factor for polynomial hashing
        hardener: XOR mask applied to final result for better distribution
    
    Returns:
        int: Computed hash value
    """
    # Process n bytes of data using polynomial rolling hash
    for i in range(n):
        hash_val *= scale
        hash_val += data[i]
    # Apply hardener to improve hash distribution
    return hash_val ^ hardener


def mangle(key, nbytes: int):
    """
    Performs byte-level manipulation on a key to produce a mangled version.
    
    Args:
        key: Bytes-like object representing the key to mangle
        nbytes: Number of bytes to process
        
    Returns:
        bytes: Mangled key as bytes
    """
    # Reverse byte order and convert to integer
    new_key = 0
    for i in range(nbytes):
        new_key |= key[nbytes - i - 1] << (i * 8)
    
    # Apply multiplication and mask to 32 bits
    new_key = (new_key * 2083697005) & 0xffffffff
    
    # Convert back to bytes
    ret_key = [(new_key >> (i * 8)) & 0xff for i in range(nbytes)]
    return bytes(ret_key)


def GenHashSeed(index: int, seed=None):
    """
    Generates a hash seed based on an index and optional base seed.
    
    Args:
        index: Index value to use in seed generation
        seed: Optional base seed; if not provided, a random value is used
        
    Returns:
        int: Generated hash seed
    """
    # Generate a random seed if none provided
    if seed is None:
        seed = random.randint(0, 2**64 - 1)
    
    # Combine seed and index
    y = seed + index
    # Previously commented out: convert to bytes, mangle, then back to int
    # x = int.from_bytes(mangle(y.to_bytes(8, 'little'), 8), 'little')
    
    # Generate hash using AwareHash with fixed parameters
    return AwareHash(y.to_bytes(8, 'little'), 8, 388650253, 388650319, 1176845762)


if __name__ == '__main__':  
    print(GenHashSeed(2026))
    pass