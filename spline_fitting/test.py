import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_element_node_blocks(file_location):
    # Open and read the file
    with open(file_location, "r") as f:
        line = f.readlines()

    ''' get element and node bolcks '''

    NodeList = []
    ElementList = []
    AllElementList = [[0]]
    NodeFlag = 0
    ElementFlag = 0
    countNumber = 0

    for line in fh.readlines():
        if line == "*Node":
            pass

def check(name):
    print(666)
    return name + "checked"

if __name__ == '__main__':
    a = np.array([[1,2],[2,3],[3,4]])
    b = np.insert(a,1,[2,3],axis=0)
    print(b)
    # dict = {"aaaa","b","c","aaaa"}
    # set_ = set(dict)
    # set_.add("cccc")
    # set_list = list(set_)
    # print(set_)
    # print(set_list[0])
    #file_location = r'E:\Google Drive\AdditiveManufacturing\ShapeOptimization\BSplineFitting\Test\test.inp'
    #with open(file_location, "r") as f:
    #    for line in f.readlines():
    #        print(line)
    #        if line[0:4] == '*Node': 
    #            print(line)
    #        else:
    #            print('not found')
    #print([0]*6)
    #vector = numpy.array([1,1,0])
    #print(type(vector[0]))
    #angle = numpy.angle(vector)
    #print(angle)
    #import math
    #print(math.acos(0.7*math.pi))

    #list0 = [1,1,2,3]
    #dic0 = {a:1, b:2, c:3}

    ##list0 = list(set(list0))
    ##print(list0)

    #for i, e in enumerate(list0):
    #    list0[i]+=1
    #print(list0)

    #plt.plot([1,2,3,4], [1,4,9,16], 'ro')
    #plt.axis([0, 6, 0, 20])
    #plt.show()
