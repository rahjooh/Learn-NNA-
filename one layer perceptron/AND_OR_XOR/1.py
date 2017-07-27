

from random import choice
from numpy import array, dot, random
__init__ = '__ Hadi Rahjou __'

unit_step = lambda x: 0 if x < 0 else 1

# OR data
OR_dataset = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 1),
]

# AND data
AND_dataset = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 1),
]

# XOR data
XOR_dataset = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 0),
]

def calc(dataset ,learningrate = 0.2 , numberofiteration = 100):
    # Weight :   three random numbers between 0 and 1
    weight_vector = random.rand(3)

    #used to store the error values
    errors = []

    for i in range(numberofiteration):

        #read a random row from trainset
        x, expected = choice(dataset)

        # calculate prediction by multiply on w
        result = dot(weight_vector, x)

        # calc error
        error = expected - unit_step(result)

        #update w by eta and error
        weight_vector += learningrate * error * x
    return weight_vector

print('\n=============== OR ================')
w = calc(OR_dataset)
print("w1 = {}   w2 = {}    w3 = {}".format(w[0], w[1], w[2]))
for x, _ in OR_dataset:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))


print('\n=============== AND ================')
w = calc(AND_dataset)
print("w1 = {}    w2 = {}   w3 = {}".format(w[0], w[1], w[2]))
for x, _ in AND_dataset:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

print('\n=============== XOR ================')
w = calc(XOR_dataset)
print("w1 = {}    w2 = {}   w3 = {}".format(w[0], w[1], w[2]))
for x, _ in XOR_dataset:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))