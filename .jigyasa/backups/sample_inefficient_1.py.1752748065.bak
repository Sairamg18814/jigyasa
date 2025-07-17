
def calculate_fibonacci(n):
    """Calculate nth Fibonacci number"""
    if n <= 1:
        return n
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def find_prime_numbers(limit):
    """Find all prime numbers up to limit"""
    primes = []
    for num in range(2, limit + 1):
        is_prime = True
        for i in range(2, num):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes

def process_data(data_list):
    """Process data inefficiently"""
    result = []
    for i in range(len(data_list)):
        item = data_list[i]
        # Inefficient string concatenation
        output = ""
        for j in range(len(item)):
            output = output + item[j].upper()
        result.append(output)
    return result
