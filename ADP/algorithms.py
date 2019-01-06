import numpy as np


__all__=['PolicyIteration', 'ValueIteration']





class ValueIteration():

    def __init__(self):
        print "Value Iteration constructed"


class PolicyIteration(ValueIteration):

    def __init__(self):
        ValueIteration.__init__(self)
        print "Policy Iteration constructed"


