Steps to run this code manually
setup python-3.4
python3 P1.py


Implementation:

1) I am calculating eigen values and eigen vectors for size = 2 and 3 matrices in the code. FOr eigen values > 3, I am using python's library numpy.
2) I have used map reduce for computing matrix multiplication
3) I am caculating best rank by reducing rank by 1 or keeping 90% of the energy

Points To Note:
1) It removes eigen values equal to 0 for computation as we are using S inverse in computation[1/0 is undefined].


Testing:
Postive cases:
  1) I have tested size 2*2, 3*3 and upto max 10*10 matrix
  2) Code ignores the data after special symbol #


error cases:
  1) I have tested for following invalid input:
     a) rows of different size
     b) Giving characters in input
     c) Giving special symbols in input
 
Note: test folder contains the test cases I used for testing the program. 
