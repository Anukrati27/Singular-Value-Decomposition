import numpy as np
import MultMapReduce as mpR
import math
from numpy import linalg as LA
from decimal import Decimal


TRUNC = 4


class SVD:

    def __init__(self, a, row, col):
        self.A = a
        self.row = row
        self.col = col        
    
    def compute(self):
        #calculating A Transpose
        ATr = list(map(list, zip(*self.A)))

        mult = mpR.MultMapReduce()
        #multiplying ATranspose and A
        ATrMultA = mult.execute(ATr, self.A, self.col, self.row, self.row, self.col)
	#calculating eigen values
        eigenValues, eigenVectors = self.calculateEigenVectors(ATrMultA)

        #removing eigen values which are olmost 0
        eigenVal = []
        for i in range((len(eigenValues))-1, -1, -1):
            tempEig = round(eigenValues[i], 6)
            if tempEig > 0:
                eigenVal.append(eigenValues[i])
            else:
                eigenVectors = np.delete(eigenVectors, i, 1)

        eigenVal.reverse()        
        rank = len(eigenVal)
        #calculating sigma and sigInverse matrix
        sig = [[0 for i in range(rank)] for i in range(rank)]
        sigInverse = [[0 for i in range(rank)] for i in range(rank)]
        for i in range(0, len(eigenVal)):
            val = math.sqrt(eigenVal[i])
            sig[i][i] = val
            sigInverse[i][i] = 1/val

        #calculating V and V transpose matrix
        V = eigenVectors
        VTr = list(map(list, zip(*eigenVectors)))

        #calculating U matrix
        tempU = mult.execute(V, sigInverse, self.col, rank, rank, rank)
        U = mult.execute(self.A, tempU, self.row, self.col, self.col, rank) 

        #calculating original matrix from above matrices 
        tempNewA = mult.execute(sig, VTr, rank, rank, rank, self.col)
        newA = mult.execute(U, tempNewA, self.row, rank, rank, self.col)

        self.printMatrices("Matrix U:", U, "Matrix S:", sig, "Matrix VT:", VTr, "Matrix new A:", newA, rank)
        
        #Best Low Rank APproximation
        sigmaVal = 0
        for i in range(0, rank):
            sigmaVal = sigmaVal + sig[i][i]*sig[i][i]

        k = 0
        #retaining 90% energy by reducing rank by 1 or by energy > 90%
        retainedSigVal = 0.9 * sigmaVal
        for k in range(rank-1,0,-1):
            tempVal = sigmaVal - sig[k][k]*sig[k][k]
            if (tempVal < retainedSigVal):
                break
        if(k < rank-1):
            k = k+1

        count = 1
        minFrobNormVal = float('inf')
        frobNormIndex  = -1
        bestB = []
        for i in range(rank-1, k-1, -1):
            tempB = mult.execute(U, sig, self.row, i, i, i)
            B = mult.execute(tempB, VTr, self.row, i, i, self.col)
            tempFrobNorm = self.computeFrobeniusNorm(B)
            if(tempFrobNorm < minFrobNormVal):
                minFrobNormVal = tempFrobNorm
                frobNormIndex = i
                bestB = B
            else:
                break

        if self.row > 1 and self.col > 1:
            self.printMatrices("Matrix U':", U, "Matrix S':", sig, "Matrix VT':", VTr, "Matrix M':", bestB, frobNormIndex)
        
            print("\nFrobenius norm = ", round(minFrobNormVal, TRUNC))

    #printing the matrices
    def printMatrices(self, uName, U, sigName, sig, VTrName, VTr, newAName, newA, index):
        print("\n",uName)
        for i in range(0, self.row):
            tempStr = ""
            for j in range(0, index):
                tempStr = tempStr + " " + str(round(U[i][j], TRUNC))
            print(tempStr)

        print("\n",sigName)
        for i in range(0, index):
            tempStr = ""
            for j in range(0, index):
                tempStr = tempStr + " " + str(round(sig[i][j], TRUNC))
            print(tempStr)

        print("\n",VTrName)
        for i in range(0, index):
            tempStr = ""
            for j in range(0, self.col):
                tempStr = tempStr + " " + str(round(VTr[i][j], TRUNC))
            print(tempStr)

        print("\n",newAName)
        for i in range(0, self.row):
            tempStr = ""
            for j in range(0, self.col):
                tempStr = tempStr + " " + str(round(newA[i][j], TRUNC))
            print(tempStr)

    #calculating Frobenius norm
    def computeFrobeniusNorm(self, B):
        res = 0
        for i in range(0, self.row):
            for j in range(0, self.col):
                res = res + math.pow((self.A[i][j] - B[i][j]), 2)
        return math.sqrt(res)

    #calculating Eigen vectors. If A transpose * A size =
    #2*2 or 3*3 calculating by code
    # greater then 3*3 matrix use numpy
    def calculateEigenVectors(self, mat):
        try:
            if self.col == 2:
                c = mat[0][0]*mat[1][1] - mat[1][0]*mat[0][1]
                a = 1
                b = (mat[0][0] + mat[1][1])*(-1)
                sig = self.calculateFactorsOfSquareMatrix(a, b, c)
                V = self.calculateVFor2Variables(mat, sig) 
            elif self.col == 3:
                a = -1
                b = mat[0][0] + mat[1][1] + mat[2][2]
                c = mat[0][2]*mat[2][0] + mat[0][1]*mat[1][0] + mat[2][1]*mat[1][2] - mat[1][1]*mat[2][2]\
                    - mat[0][0]*mat[2][2] - mat[0][0]*mat[1][1]
                d = mat[0][0] * ( mat[1][1]*mat[2][2] - mat[2][1]*mat[1][2] ) + \
                    mat[0][1] * ( mat[2][0]*mat[1][2] - mat[1][0]*mat[2][2] ) + \
                    mat[0][2] * ( mat[1][0]*mat[2][1] - mat[2][0]*mat[1][1] )
                sig = self.calculateFactorsOfCubicMatrix(a, b, c, d)
                print("\n\nsig values are--> ", sig)
                V = self.calculateVFor3Variable(mat, sig)              
            else:
                sig, V = self.getEigenValFromLA(mat)
            print("\n\nzzzzzzzzzzzzzzzzzzzz", sig, V)
            return sig, V
        except Exception as e:
            print("\nInException\n")
            return self.getEigenValFromLA(mat)

    #calculating Eigen values using numpy
    def getEigenValFromLA(self, mat):
        sig, V = LA.eig( np.array(mat) )
        idx = sig.argsort()[::-1]
        sig = sig[idx]
        V = V[:,idx]
        print("\n\n------ ", sig, V)
        return sig, V

    #calculating V for 2 variables
    def calculateVFor2Variables(self, matrix, sig):
        V = [[0 for i in range(0, len(sig))] for i in range(len(sig))]
        row = 0
        for i in range(0, len(sig)):
            ATrMulAMincI = self.calculateIntermediateMatrix(matrix, sig[i])
            if ATrMulAMincI[0][1] != 0:
                x1 = 1
                x2 = (-1)*ATrMulAMincI[0][0]/ATrMulAMincI[0][1]
            elif ATrMulAMincI[0][0] != 0:
                x2 = 1
                x1 = (-1)*ATrMulAMincI[0][1]/ATrMulAMincI[0][0]
            elif ATrMulAMincI[1][1] != 0:
                x1 = 1
                x2 = (-1)*ATrMulAMincI[1][0]/ATrMulAMincI[1][1]
            elif ATrMulAMincI[1][0] != 0:
                x2 = 1
                x1 = (-1)*ATrMulAMincI[1][1]/ATrMulAMincI[1][0]
            else:
                print("\nInvalid matrix provided. Exiting!!!\n")
                exit(1)

            val = math.sqrt(x1*x1 + x2*x2)
            x1 = x1/val
            x2 = x2/val
            V[0][row] = x1
            V[1][row] = x2
            row = row + 1
        return V
            
    #calculating V matrix for 3 variables
    def calculateVFor3Variable(self, matrix, sig):
        V = [[0 for i in range(0, self.col)] for i in range(len(sig))]
        row = 0
        for i in range(0, len(sig)):
            ATrMulAMincI = self.calculateIntermediateMatrix(matrix, sig[i])
            z = 2
            a = 0
            b = 0
            flag = False
            for i in range(len(ATrMulAMincI)):
                for j in range(len(ATrMulAMincI)):
                    if i == j:
                        continue
                    mul = ATrMulAMincI[i][z]/ATrMulAMincI[j][z]

                    a = ATrMulAMincI[j][0] * mul - ATrMulAMincI[i][0]
                    b = ATrMulAMincI[j][1] * mul - ATrMulAMincI[i][1]
                    if a == 0 and b == 0:
                       continue
                    elif a != 0 and b != 0:
                        flag = True
                        break
                if flag == True:
                    break
            if a == 0 or b == 0:
                x = [0, 0, 0]
            else:
                x1 = 1
                x2 = -1 * a / b
                x3 = 0
                for i in range(len(ATrMulAMincI)):
                    if ATrMulAMincI[i][2] != 0:
                        x3 = -1*((x1*ATrMulAMincI[0][0] + x2*ATrMulAMincI[0][1])/ATrMulAMincI[0][2])
                val = math.sqrt( x1*x1 + x2*x2 + x3*x3 )
                x1 = x1/val
                x2 = x2/val
                x3 = x3/val
                x =  [x1, x2, x3]
            V[0][row] = x[0]
            V[1][row] = x[1]
            V[2][row] = x[2]
            row = row + 1
        return V
                        
    #calculating (ATr*A - Sig*Identity) matrix
    def calculateIntermediateMatrix(self, matrix, sigVal):
        ATrMulAMincI = [[0 for i in range(self.col+1)] for i in range(self.col)]
        for i in range(0, self.col):
            for j in range(0, self.col):
                if i == j:
                    ATrMulAMincI[i][j] = matrix[i][i] - sigVal
                else:
                    ATrMulAMincI[i][j] = matrix[i][j]
        return ATrMulAMincI
       
    #calculating factors of a quadratic equation using mathematical formula
    def calculateFactorsOfSquareMatrix(self, a, b, c):
        d = math.sqrt(b*b - 4*a*c)
        sol1 = ((-1)*b + d )/(2*a)
        sol2 = ((-1)*b - d )/(2*a)
        if sol1 == sol2:
            return [sol1]
        elif sol1 > sol2:
            return [sol1, sol2]
        else:
            return [sol2, sol1]

    #calculating factors of a cubic equation using mathematical formula.
    #Code exists if equation has imaginary data
    def calculateFactorsOfCubicMatrix(self, a, b, c, d):
        f = ((3*c/a) - ((b*b)/(a*a)))/3
        g = ((2*math.pow(b,3)/math.pow(a,3))-(9*b*c/math.pow(a,2))+(27*d/a))/27
        h = (math.pow(g,2)/4)+(math.pow(f,3)/27)
        if h<=0:
            i = math.sqrt((g*g/4)-h)
            j = i**(1./3.)
            k = math.acos((-1)*(g/(2*i)))
            L = j *  (-1)
            M = math.cos(k/3)
            N = math.sqrt(3) * math.sin(k/3)
            P = (b/(3*a))*(-1)
            x1 = round(2 * j * math.cos(k/3) - (b / (3 * a)),6)
            x2 = round(L * (M + N) + P, 6)
            print("\n################# ", 2 * j * math.cos(k/3) - (b / (3 * a)))
            print("\n\n--- ", L * (M + N) + P)
            x3 = round(L * (M - N) + P, 6)
            retList = [x1, x2, x3]
            retList.sort(reverse=True)
            return retList
        elif f == 0 and g == 0 and h == 0:
            x1 = ((d/a)**(1./3.))*(-1)
            return [round(x1, 6)]
        elif h > 0:
            print("\nEquation has imaginary eigen values. Please check the data. Exiting!!!\n") 
            exit(0)
        else:
            print("\nError in computing eigen vectors. Exiting!!!\n")
            exit(1)
 
 


