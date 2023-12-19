import numpy as np

# Test case 1: lower = 1, upper = 10, fee = 0.01
lower = 1
upper = 10
fee = 0.01
expected_result = 0.05  # Replace with the expected result
result = get_iv(lower, upper, fee)
assert np.isclose(result, expected_result), f"Test case 1 failed: expected {expected_result}, got {result}"

# Test case 2: lower = 5, upper = 15, fee = 0.05
lower = 5
upper = 15
fee = 0.05
expected_result = 0.08  # Replace with the expected result
result = get_iv(lower, upper, fee)
assert np.isclose(result, expected_result), f"Test case 2 failed: expected {expected_result}, got {result}"

# Test case 3: lower = 10, upper = 20, fee = 0.1
lower = 10
upper = 20
fee = 0.1
expected_result = 0.12  # Replace with the expected result
result = get_iv(lower, upper, fee)
assert np.isclose(result, expected_result), f"Test case 3 failed: expected {expected_result}, got {result}"

# Add more test cases as needed

print("All test cases passed!")