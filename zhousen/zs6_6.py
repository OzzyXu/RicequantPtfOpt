# practice python

import numpy as np
import pandas as pd
from math import *

#1
def drop_first_last(grades):
    first, *middle, last = grades
    return average(middle)

drop_first_last([1,3,5,4,7])

#2
def sum(items):
    head, *tail = items
    return sum(tail) if tail else head


sum([nan,1])
sum([3])


print(1) if nan else print(0)


#3
from collections import deque

def search(lines, pattern, history=5):
    previous_lines = deque(maxlen=history)
    for li in lines:
        if pattern in li:
#            yield li, previous_lines
            print(li, previous_lines)
        previous_lines.append(li)

# Example use on a file
if __name__ == '__main__':
    with open(r'../test1/shenwan_df.txt') as f:
        for line, prevlines in search(f, '电', 5):
            for pline in prevlines:
                print(pline, end='')
            print(line, end='')
            print('-' * 20)




mygenerator = (x*x for x in range(3))

for i in mygenerator:
    print(i)


def createGenerator():
    mylist = range(3)
    for i in mylist:
        yield i*i

mygenerator = createGenerator()
print(mygenerator)
for i in mygenerator:
    print(i)




with open(r'../test1/shenwan_df.txt') as f:
    d1 = f.readlines()


# 4

q = deque(maxlen=3)
q
q.append(1)
q.append(2)
q.append(3)
q.append(1)




q.appendleft(4)