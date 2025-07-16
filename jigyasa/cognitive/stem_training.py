"""
STEM and Coding Training Generator
Generates mathematical, scientific, and coding problems dynamically
"""

import random
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class TrainingExample:
    """A single training example"""
    question: str
    answer: str
    category: str
    difficulty: str
    reasoning_steps: List[str] = None


class STEMTrainingGenerator:
    """
    Generates STEM and coding training examples dynamically
    No predefined topics - pure problem generation
    """
    
    def __init__(self):
        # Mathematical operations and concepts
        self.math_operations = {
            'basic': ['addition', 'subtraction', 'multiplication', 'division'],
            'intermediate': ['exponents', 'roots', 'logarithms', 'factorials'],
            'advanced': ['derivatives', 'integrals', 'limits', 'series']
        }
        
        # Programming concepts
        self.coding_concepts = {
            'basic': ['variables', 'loops', 'conditionals', 'functions'],
            'intermediate': ['arrays', 'dictionaries', 'classes', 'recursion'],
            'advanced': ['algorithms', 'data structures', 'complexity', 'optimization']
        }
        
        # Science concepts
        self.science_areas = {
            'physics': ['mechanics', 'thermodynamics', 'electromagnetism', 'quantum'],
            'chemistry': ['elements', 'reactions', 'bonds', 'equilibrium'],
            'biology': ['cells', 'genetics', 'evolution', 'ecology']
        }
    
    def generate_math_problem(self, difficulty: str = 'basic') -> TrainingExample:
        """Generate a mathematical problem"""
        if difficulty == 'basic':
            return self._generate_basic_math()
        elif difficulty == 'intermediate':
            return self._generate_intermediate_math()
        else:
            return self._generate_advanced_math()
    
    def _generate_basic_math(self) -> TrainingExample:
        """Generate basic arithmetic problems"""
        operation = random.choice(['add', 'subtract', 'multiply', 'divide'])
        
        if operation == 'add':
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            question = f"What is {a} + {b}?"
            answer = str(a + b)
            reasoning = [
                f"We need to add {a} and {b}",
                f"Starting with {a}",
                f"Adding {b} gives us {a + b}",
                f"Therefore, {a} + {b} = {answer}"
            ]
        
        elif operation == 'subtract':
            a = random.randint(50, 200)
            b = random.randint(1, a)
            question = f"What is {a} - {b}?"
            answer = str(a - b)
            reasoning = [
                f"We need to subtract {b} from {a}",
                f"Starting with {a}",
                f"Taking away {b} leaves us with {a - b}",
                f"Therefore, {a} - {b} = {answer}"
            ]
        
        elif operation == 'multiply':
            a = random.randint(2, 20)
            b = random.randint(2, 20)
            question = f"What is {a} × {b}?"
            answer = str(a * b)
            reasoning = [
                f"We need to multiply {a} by {b}",
                f"This means adding {a} to itself {b} times",
                f"{a} × {b} = {a * b}",
                f"Therefore, the answer is {answer}"
            ]
        
        else:  # divide
            b = random.randint(2, 10)
            result = random.randint(2, 20)
            a = b * result
            question = f"What is {a} ÷ {b}?"
            answer = str(result)
            reasoning = [
                f"We need to divide {a} by {b}",
                f"This means finding how many times {b} fits into {a}",
                f"{b} × {result} = {a}",
                f"Therefore, {a} ÷ {b} = {answer}"
            ]
        
        return TrainingExample(
            question=question,
            answer=answer,
            category='mathematics',
            difficulty='basic',
            reasoning_steps=reasoning
        )
    
    def _generate_intermediate_math(self) -> TrainingExample:
        """Generate intermediate math problems"""
        problem_type = random.choice(['quadratic', 'percentage', 'algebra', 'geometry'])
        
        if problem_type == 'quadratic':
            # Generate solvable quadratic
            root1 = random.randint(-10, 10)
            root2 = random.randint(-10, 10)
            a = 1
            b = -(root1 + root2)
            c = root1 * root2
            
            question = f"Solve for x: x² {'+' if b >= 0 else ''}{b}x {'+' if c >= 0 else ''}{c} = 0"
            answer = f"x = {root1} or x = {root2}"
            reasoning = [
                f"This is a quadratic equation in the form ax² + bx + c = 0",
                f"We can factor this as (x - {root1})(x - {root2}) = 0",
                f"Setting each factor to zero: x - {root1} = 0 or x - {root2} = 0",
                f"Therefore, x = {root1} or x = {root2}"
            ]
        
        elif problem_type == 'percentage':
            original = random.randint(50, 500)
            percent = random.randint(5, 50)
            increase = original * percent / 100
            new_value = original + increase
            
            question = f"If a price of ${original} increases by {percent}%, what is the new price?"
            answer = f"${new_value:.2f}"
            reasoning = [
                f"Original price: ${original}",
                f"Increase percentage: {percent}%",
                f"Increase amount: ${original} × {percent}/100 = ${increase:.2f}",
                f"New price: ${original} + ${increase:.2f} = ${new_value:.2f}"
            ]
        
        elif problem_type == 'algebra':
            a = random.randint(2, 10)
            b = random.randint(1, 20)
            c = random.randint(10, 50)
            x_value = (c - b) / a
            
            question = f"Solve for x: {a}x + {b} = {c}"
            answer = f"x = {x_value:.2f}"
            reasoning = [
                f"Starting with {a}x + {b} = {c}",
                f"Subtract {b} from both sides: {a}x = {c - b}",
                f"Divide both sides by {a}: x = {(c - b)}/{a}",
                f"Therefore, x = {x_value:.2f}"
            ]
        
        else:  # geometry
            radius = random.randint(3, 15)
            area = math.pi * radius ** 2
            
            question = f"What is the area of a circle with radius {radius} units? (Use π ≈ 3.14159)"
            answer = f"{area:.2f} square units"
            reasoning = [
                f"Area of a circle = π × r²",
                f"Given radius r = {radius}",
                f"Area = π × {radius}² = π × {radius**2}",
                f"Area ≈ 3.14159 × {radius**2} = {area:.2f} square units"
            ]
        
        return TrainingExample(
            question=question,
            answer=answer,
            category='mathematics',
            difficulty='intermediate',
            reasoning_steps=reasoning
        )
    
    def _generate_advanced_math(self) -> TrainingExample:
        """Generate advanced math problems"""
        problem_type = random.choice(['derivative', 'integral', 'limit'])
        
        if problem_type == 'derivative':
            power = random.randint(2, 5)
            coeff = random.randint(1, 5)
            
            question = f"Find the derivative of f(x) = {coeff}x^{power}"
            answer = f"f'(x) = {coeff * power}x^{power-1}"
            reasoning = [
                f"Using the power rule: d/dx(ax^n) = n·a·x^(n-1)",
                f"Here, a = {coeff} and n = {power}",
                f"f'(x) = {power} · {coeff} · x^({power}-1)",
                f"f'(x) = {coeff * power}x^{power-1}"
            ]
        
        elif problem_type == 'integral':
            power = random.randint(1, 4)
            coeff = random.randint(1, 5)
            
            question = f"Find the integral of f(x) = {coeff}x^{power}"
            answer = f"∫f(x)dx = {coeff/(power+1):.2f}x^{power+1} + C"
            reasoning = [
                f"Using the power rule for integration: ∫x^n dx = x^(n+1)/(n+1) + C",
                f"Here, we have {coeff}x^{power}",
                f"∫{coeff}x^{power}dx = {coeff} · x^({power}+1)/({power}+1) + C",
                f"= {coeff/(power+1):.2f}x^{power+1} + C"
            ]
        
        else:  # limit
            a = random.randint(1, 5)
            question = f"Find the limit: lim(x→{a}) (x² - {a}²)/(x - {a})"
            answer = f"{2*a}"
            reasoning = [
                f"We can factor the numerator: x² - {a}² = (x + {a})(x - {a})",
                f"So the expression becomes: (x + {a})(x - {a})/(x - {a})",
                f"Cancel (x - {a}) from numerator and denominator: x + {a}",
                f"As x → {a}, we get: {a} + {a} = {2*a}"
            ]
        
        return TrainingExample(
            question=question,
            answer=answer,
            category='mathematics',
            difficulty='advanced',
            reasoning_steps=reasoning
        )
    
    def generate_coding_problem(self, difficulty: str = 'basic') -> TrainingExample:
        """Generate a coding problem"""
        if difficulty == 'basic':
            return self._generate_basic_coding()
        elif difficulty == 'intermediate':
            return self._generate_intermediate_coding()
        else:
            return self._generate_advanced_coding()
    
    def _generate_basic_coding(self) -> TrainingExample:
        """Generate basic coding problems"""
        problem_type = random.choice(['loop', 'conditional', 'function', 'array'])
        
        if problem_type == 'loop':
            n = random.randint(5, 10)
            question = f"Write a Python function to print numbers from 1 to {n}"
            answer = f"""def print_numbers():
    for i in range(1, {n+1}):
        print(i)"""
            reasoning = [
                f"We need to print numbers from 1 to {n}",
                "Use a for loop with range function",
                f"range(1, {n+1}) generates numbers from 1 to {n}",
                "Print each number in the loop"
            ]
        
        elif problem_type == 'conditional':
            threshold = random.randint(10, 50)
            question = f"Write a function that returns 'High' if a number is greater than {threshold}, else 'Low'"
            answer = f"""def check_number(num):
    if num > {threshold}:
        return 'High'
    else:
        return 'Low'"""
            reasoning = [
                f"Compare the input number with {threshold}",
                f"If greater than {threshold}, return 'High'",
                "Otherwise, return 'Low'",
                "Use if-else conditional statement"
            ]
        
        elif problem_type == 'function':
            question = "Write a function to calculate the sum of two numbers"
            answer = """def add_numbers(a, b):
    return a + b"""
            reasoning = [
                "Define a function with two parameters",
                "Add the two parameters together",
                "Return the result",
                "Simple and straightforward implementation"
            ]
        
        else:  # array
            size = random.randint(3, 7)
            question = f"Create a list of the first {size} even numbers"
            answer = f"even_numbers = [i * 2 for i in range(1, {size + 1})]"
            reasoning = [
                f"We need the first {size} even numbers",
                "Even numbers are 2, 4, 6, 8, ...",
                "Use list comprehension for efficiency",
                f"Multiply each number from 1 to {size} by 2"
            ]
        
        return TrainingExample(
            question=question,
            answer=answer,
            category='coding',
            difficulty='basic',
            reasoning_steps=reasoning
        )
    
    def _generate_intermediate_coding(self) -> TrainingExample:
        """Generate intermediate coding problems"""
        problem_type = random.choice(['recursion', 'sorting', 'search', 'data_structure'])
        
        if problem_type == 'recursion':
            n = random.randint(5, 10)
            question = f"Write a recursive function to calculate the factorial of {n}"
            answer = """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""
            reasoning = [
                "Factorial: n! = n × (n-1) × ... × 2 × 1",
                "Base case: 0! = 1! = 1",
                "Recursive case: n! = n × (n-1)!",
                "The function calls itself with n-1"
            ]
        
        elif problem_type == 'sorting':
            question = "Write a function to sort a list using bubble sort"
            answer = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr"""
            reasoning = [
                "Compare adjacent elements",
                "Swap if they're in wrong order",
                "Repeat for all elements",
                "Larger elements 'bubble' to the end"
            ]
        
        elif problem_type == 'search':
            question = "Implement binary search for a sorted array"
            answer = """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1"""
            reasoning = [
                "Binary search works on sorted arrays",
                "Compare target with middle element",
                "Eliminate half of the array each time",
                "Time complexity: O(log n)"
            ]
        
        else:  # data_structure
            question = "Implement a simple stack class with push and pop methods"
            answer = """class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def is_empty(self):
        return len(self.items) == 0"""
            reasoning = [
                "Stack follows LIFO (Last In First Out)",
                "Use a list to store elements",
                "Push adds to the end",
                "Pop removes from the end"
            ]
        
        return TrainingExample(
            question=question,
            answer=answer,
            category='coding',
            difficulty='intermediate',
            reasoning_steps=reasoning
        )
    
    def _generate_advanced_coding(self) -> TrainingExample:
        """Generate advanced coding problems"""
        problem_type = random.choice(['dynamic_programming', 'graph', 'optimization'])
        
        if problem_type == 'dynamic_programming':
            question = "Write a function to find the nth Fibonacci number using dynamic programming"
            answer = """def fibonacci(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[0], dp[1] = 0, 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]"""
            reasoning = [
                "Fibonacci: F(n) = F(n-1) + F(n-2)",
                "Use dynamic programming to avoid recalculation",
                "Store computed values in an array",
                "Time complexity: O(n), Space: O(n)"
            ]
        
        elif problem_type == 'graph':
            question = "Implement depth-first search (DFS) for a graph"
            answer = """def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start, end=' ')
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    
    return visited"""
            reasoning = [
                "DFS explores as far as possible along each branch",
                "Use recursion to visit nodes",
                "Keep track of visited nodes",
                "Time complexity: O(V + E)"
            ]
        
        else:  # optimization
            question = "Find the maximum subarray sum (Kadane's algorithm)"
            answer = """def max_subarray_sum(arr):
    max_sum = current_sum = arr[0]
    
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum"""
            reasoning = [
                "Track current subarray sum",
                "Reset if sum becomes negative",
                "Keep track of maximum seen so far",
                "Time complexity: O(n)"
            ]
        
        return TrainingExample(
            question=question,
            answer=answer,
            category='coding',
            difficulty='advanced',
            reasoning_steps=reasoning
        )
    
    def generate_science_problem(self, difficulty: str = 'basic') -> TrainingExample:
        """Generate science problems"""
        area = random.choice(['physics', 'chemistry', 'biology'])
        
        if area == 'physics':
            if difficulty == 'basic':
                speed = random.randint(10, 50)
                time = random.randint(2, 10)
                distance = speed * time
                
                question = f"If a car travels at {speed} km/h for {time} hours, how far does it travel?"
                answer = f"{distance} km"
                reasoning = [
                    "Distance = Speed × Time",
                    f"Speed = {speed} km/h",
                    f"Time = {time} hours",
                    f"Distance = {speed} × {time} = {distance} km"
                ]
            else:
                mass = random.randint(1, 10)
                acceleration = random.randint(2, 10)
                force = mass * acceleration
                
                question = f"What force is needed to accelerate a {mass} kg object at {acceleration} m/s²?"
                answer = f"{force} N"
                reasoning = [
                    "Newton's Second Law: F = ma",
                    f"Mass (m) = {mass} kg",
                    f"Acceleration (a) = {acceleration} m/s²",
                    f"Force (F) = {mass} × {acceleration} = {force} N"
                ]
        
        elif area == 'chemistry':
            if difficulty == 'basic':
                element = random.choice(['H', 'O', 'C', 'N'])
                names = {'H': 'Hydrogen', 'O': 'Oxygen', 'C': 'Carbon', 'N': 'Nitrogen'}
                
                question = f"What element has the symbol {element}?"
                answer = names[element]
                reasoning = [
                    "Chemical symbols represent elements",
                    f"{element} is the symbol for {names[element]}",
                    "This is a fundamental element",
                    "Memorization of common elements is important"
                ]
            else:
                question = "Balance this equation: H2 + O2 → H2O"
                answer = "2H2 + O2 → 2H2O"
                reasoning = [
                    "Count atoms on each side",
                    "Left: 2 H, 2 O. Right: 2 H, 1 O",
                    "Need to balance oxygen",
                    "2H2 + O2 → 2H2O (now balanced)"
                ]
        
        else:  # biology
            if difficulty == 'basic':
                question = "What is the powerhouse of the cell?"
                answer = "Mitochondria"
                reasoning = [
                    "Cells need energy to function",
                    "Mitochondria produce ATP",
                    "ATP is the energy currency of cells",
                    "Hence: 'powerhouse of the cell'"
                ]
            else:
                question = "What is the process by which plants make their own food?"
                answer = "Photosynthesis"
                reasoning = [
                    "Plants are autotrophs",
                    "They convert light energy to chemical energy",
                    "Use CO2 and H2O to make glucose",
                    "This process is called photosynthesis"
                ]
        
        return TrainingExample(
            question=question,
            answer=answer,
            category='science',
            difficulty=difficulty,
            reasoning_steps=reasoning
        )
    
    def generate_training_batch(
        self, 
        batch_size: int = 10,
        mix: Dict[str, float] = None
    ) -> List[TrainingExample]:
        """
        Generate a batch of training examples
        
        Args:
            batch_size: Number of examples to generate
            mix: Dictionary specifying the mix of categories
                 e.g., {'math': 0.4, 'coding': 0.4, 'science': 0.2}
        """
        if mix is None:
            mix = {'math': 0.4, 'coding': 0.4, 'science': 0.2}
        
        examples = []
        
        for category, proportion in mix.items():
            count = int(batch_size * proportion)
            
            for _ in range(count):
                difficulty = random.choice(['basic', 'intermediate', 'advanced'])
                
                if category == 'math':
                    example = self.generate_math_problem(difficulty)
                elif category == 'coding':
                    example = self.generate_coding_problem(difficulty)
                elif category == 'science':
                    example = self.generate_science_problem(difficulty)
                
                examples.append(example)
        
        # Shuffle the examples
        random.shuffle(examples)
        
        return examples[:batch_size]


# Conversational training for human-like responses
class ConversationalTrainer:
    """Trains the model to speak more naturally"""
    
    def __init__(self):
        self.conversation_templates = [
            {
                "context": "greeting",
                "examples": [
                    ("Hello!", "Hey there! How's it going?"),
                    ("Hi", "Hi! What can I help you with today?"),
                    ("Good morning", "Good morning! Hope you're having a great day."),
                    ("Hey", "Hey! What's up?"),
                ]
            },
            {
                "context": "explanation",
                "examples": [
                    ("Can you explain this?", "Sure! Let me break it down for you..."),
                    ("I don't understand", "No worries, let me explain it differently..."),
                    ("What does that mean?", "Good question! So basically..."),
                    ("How does this work?", "I'll walk you through it step by step..."),
                ]
            },
            {
                "context": "problem_solving",
                "examples": [
                    ("I'm stuck on this problem", "Let's tackle this together. First..."),
                    ("Can you help me?", "Of course! Let's see what we're working with..."),
                    ("I need help with", "I'm here to help! Show me what you've got..."),
                    ("How do I solve this?", "Great question! Here's how I'd approach it..."),
                ]
            }
        ]
    
    def generate_conversational_examples(self, count: int = 5) -> List[Dict[str, str]]:
        """Generate conversational training examples"""
        examples = []
        
        for _ in range(count):
            template = random.choice(self.conversation_templates)
            user_input, response = random.choice(template['examples'])
            
            # Add variations
            if random.random() < 0.3:
                response = self._add_casual_variation(response)
            
            examples.append({
                'input': user_input,
                'response': response,
                'style': 'conversational'
            })
        
        return examples
    
    def _add_casual_variation(self, response: str) -> str:
        """Add casual variations to responses"""
        casual_additions = [
            "Actually, ",
            "You know what? ",
            "Here's the thing - ",
            "Alright, so ",
            "Okay, ",
        ]
        
        if random.random() < 0.5:
            response = random.choice(casual_additions) + response.lower()
        
        return response