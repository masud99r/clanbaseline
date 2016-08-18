import numpy
from semanticpy.transform.lsa import LSA
matrix = [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
          [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

expected = [
    [0.02284739, 0.06123732, 1.20175485, 0.02284739, 0.02284739, 0.88232986, 0.03838993, 0.03838993, 0.82109254],
    [-0.00490259, 0.98685971, -0.04329252, -0.00490259, -0.00490259, 1.02524964, 0.99176229, 0.99176229, 0.03838993],
    [0.99708227, 0.99217968, -0.02576511, 0.99708227, 0.99708227, 1.01502707, -0.00490259, -0.00490259, 0.02284739],
    [-0.0486125, -0.13029496, 0.57072519, -0.0486125, -0.0486125, 0.25036735, -0.08168246, -0.08168246, 0.3806623]]

lsa = LSA(matrix)
new_matrix = lsa.transform()
print new_matrix


import numpy, scipy.sparse
from sparsesvd import sparsesvd
from scipy import linalg,dot
#mat = numpy.random.rand(200, 100) # create a random matrix
smat = scipy.sparse.csc_matrix(matrix) # convert to sparse CSC format
ut, s, vt = sparsesvd(smat, 5) # do SVD, asking for 100 factors
assert numpy.allclose(matrix, numpy.dot(ut.T, numpy.dot(numpy.diag(s), vt)))
print ut
print s
print vt
transformed_matrix = dot(dot(ut, linalg.diagsvd(s, len(matrix), len(vt))) ,vt)
print "Transform"
print transformed_matrix