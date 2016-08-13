#from numpy import *
from scipy import linalg,dot
#A = floor(random.rand(4, 3)*20-10) # generating a random
#A = [[x*y for x in range(3)] for y in range(4)]
A = []
A.append([3, 2, 2])
A.append([2, 3, -2])
U,s,V = linalg.svd(A) # SVD decomposition of A
s[1]=0
print "A is \n", A
print "U is \n", U
print "s is \n", s
print "V is \n", V
reconstructedMatrix= dot(dot(U,linalg.diagsvd(s,len(A),len(V))),V)
print "Reconstructed = \n", reconstructedMatrix