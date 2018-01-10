# ------------- Classes --------------#

# Example
class V2:
    
    def __init__(self, num1, num2):
        self.num1 = num1
        self.num2 = num2
    
    def __str__(self):
        return '{0}, {1}'.format(self.num1, self.num2)
    
    def getX(self):
        return self.num1
    
    def getY(self):
        return self.num2
    
    def __add__(self, other):
        return (self.num1 + other.num1, self.num2 + other.num2)
    
    def __mul__(self, other):
        return (self.num1 * other, self.num2 * other)
    
    def __main__(self):
        return 'I am in main'