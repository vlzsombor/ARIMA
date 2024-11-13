class MyCallableClass:
    def __init__(self, value):
        self.value = value

    def __call__(self, increment):
        return self.value + increment

# Create an instance of the class
my_instance = MyCallableClass(10)(4)

# Now, call the instance like a function
print(my_instance)  # Output will be 15
print('fadfasdfsz')

input1 = input()
