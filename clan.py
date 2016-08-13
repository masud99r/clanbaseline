import numpy
from semanticpy.transform.lsa import LSA
from nose.tools import *
from semanticpy.transform.tfidf import TFIDF

def LSISimilarityMatrix(matrix):

    matrix = [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
              [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    expected = [
        [0.02284739, 0.06123732, 1.20175485, 0.02284739, 0.02284739, 0.88232986, 0.03838993, 0.03838993, 0.82109254],
        [-0.00490259, 0.98685971, -0.04329252, -0.00490259, -0.00490259, 1.02524964, 0.99176229, 0.99176229, 0.03838993],
        [0.99708227, 0.99217968, -0.02576511, 0.99708227, 0.99708227, 1.01502707, -0.00490259, -0.00490259, 0.02284739],
        [-0.0486125, -0.13029496, 0.57072519, -0.0486125, -0.0486125, 0.25036735, -0.08168246, -0.08168246, 0.3806623]]
    tfidf = TFIDF(matrix)
    new_tfidf_matrix = tfidf.transform()
    #print "TFIDF Transform Matrix\n", new_tfidf_matrix
    lsa = LSA(new_tfidf_matrix)
    new_matrix = lsa.transform(2)
    #print "Final Matrix", new_matrix
    return  new_matrix

#def similarity(lsimatrix):


def main():
    matrix = [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
              [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    transform_matrix = LSISimilarityMatrix(matrix)
    print transform_matrix

if __name__ == "__main__":
	main()