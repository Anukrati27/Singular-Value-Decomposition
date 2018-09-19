#Map Reducer implemented using the algorithm given in book
import sys
from collections import OrderedDict


#MapReduce class
class MapReduce:
    def __init__(self):
        self.intermediate = OrderedDict()
        self.result = []

    def emit_intermediate(self, key, value):
        self.intermediate.setdefault(key, [])
        self.intermediate[key].append(value)


    def emit(self, i, j , value):
        self.result[i][j] = value


    def execute(self, matrix1, matrix2, row1, col1, row2, col2, mapper, reducer):
        self.intermediate = OrderedDict()
        self.result = []
        for i in range(0, row1):
            self.result.append([0] * col2)

        mapper(matrix1, matrix2, row1, col1, row2, col2)

        for key in self.intermediate:
            reducer(key, self.intermediate[key])

        return self.result

mapReducer = None

#mapper function
def mapper(matrix1, matrix2, row1, col1, row2, col2):
    for i in range(0, row1):
        for j in range(0, col1):
            for k in range(0,col2):
                 mapReducer.emit_intermediate((i,k), ('L', j, matrix1[i][j])) 

    for j in range(0, row2):
        for k in range(0, col2):
            for i in range(0,row1):
                 mapReducer.emit_intermediate((i,k), ('R', j, matrix2[j][k])) 
  
#reducer function 
def reducer(key, valList):
    L = [(item[1],item[2]) for item in valList if item[0] == 'L' ]
    R = [(item[1],item[2]) for item in valList if item[0] == 'R' ]

    result = 0
    for l in L:
        for r in R:
            if l[0] == r[0]:
                result = result + l[1]*r[1]
                break

    if result != 0:
        mapReducer.emit(key[0], key[1], result) 


class MultMapReduce:
    def __init__(self):
        global mapReducer
        mapReducer = MapReduce()

    def execute(self, matrix1, matrix2, row1, col1, row2, col2):
        ret = mapReducer.execute(matrix1, matrix2, row1, col1, row2, col2, mapper, reducer)
        return ret
