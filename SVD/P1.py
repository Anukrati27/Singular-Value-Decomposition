import SVD
import sys

if __name__ == '__main__':
    try:

        row = 0
        col = 0
        A = []
        #Reading data until user press enter with no data
        while 1:
            origData = sys.stdin.readline().strip().strip(" ").strip("\t")
            #if data starts with comment ignore the entire line
            if origData.startswith("#"):
                continue
            #if '#' present in between, ignore the data to the right of '#'
            data = origData.split("#")[0]
            #replacing all tabs with space
            data = data.replace("\t"," ")
            #splitting data based on space
            dataList = data.split(" ")
            dataList = [x for x in dataList if x!='']
            if len(dataList) == 0:
                if A == []:
                    print ("No Valid data provided by user. Exiting!!!!")
                    exit(1)
                break
            if col == 0:
                col = len(dataList)
            else:
                if (len(dataList) != col):
                    print ("Invalid Matrix provided. Exiting!!!!")
                    exit(1)
            tempList = []
            try:
                for item in dataList:
                    tempList.append(float(item))
            except Exception as e:
                print ("Input data type is not an 'Integer'. Please check the data. Exiting!!!!")
                exit(1)
            A.append(tempList)
            row = row + 1

        svd = SVD.SVD(A, row, col)
        svd.compute()

    except Exception as e:
        print ("Error in computing SVD: ", str(e))

