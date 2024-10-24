import unittest
# Реалізація функції AND
def my_and(x1, x2):
    return x1 if x1 == x2 else 0

# Реалізація функції OR
def my_or(x1, x2):
    return x1 if x1 else x2

# Реалізація функції XOR через OR і AND
def my_xor(x1, x2):
    return my_or(x1, x2) and not my_and(x1, x2)

# Приклади використання:
x1 = 1  # True
x2 = 0  # False

print("AND:", my_and(x1, x2))  # AND поверне 0
print("OR:", my_or(x1, x2))    # OR поверне 1
print("XOR:", my_xor(x1, x2))  # XOR поверне 1


class TestXOR(unittest.TestCase):
    def test_xor_true_true(self):
        self.assertEqual(my_xor(1, 1), 0)
    def test_xor_true_false(self):
        self.assertEqual(my_xor(1, 0), 1)
    def test_xor_false_true(self):
        self.assertEqual(my_xor(0, 1), 1)
    def test_xor_false_false(self):
        self.assertEqual(my_xor(0, 0), 0)
if __name__ == '__main__':
    unittest.main()