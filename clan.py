import numpy
import textmining
from util import *
from semanticpy.transform.lsa import LSA
from nose.tools import *
from semanticpy.transform.tfidf import TFIDF
try:
	from numpy import dot
	from numpy.linalg import norm
except:
	print "Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?"
	sys.exit()


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


def calculate_cosine(vector1, vector2):
    """ related documents j and q are in the concept space by comparing the vectors :
        cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
    return float(dot(vector1, vector2) / (norm(vector1) * norm(vector2)))
def similarity(lsimatrix):
    dim = len(lsimatrix)
    S = [[0 for x in range(dim)] for y in range(dim)]
    for i in range(0,len(lsimatrix)):
        for j in range(0,i+1):
            if i == j:
                S[i][j] = 1
                S[j][i] = 1

            else:
                cosine_value = calculate_cosine(lsimatrix[i],lsimatrix[j])
                S[i][j] = cosine_value
                S[j][i] = cosine_value
    return S
def get_rankedlist(query_index, similarity_matrix):
    if query_index<0 or query_index >= len(similarity_matrix):
        print "Invalid query index"
        return -1
    unorder_list = similarity_matrix[query_index]
    distances = []
    for i in range(len(unorder_list)):
        distances.append((i, unorder_list[i]))
    print "un order distance matrix = \n", distances
    distances.sort(key=lambda x: x[1],reverse=True)#sort in reverse order
    print "Sorted by cosine similarity value distance matrix = \n", distances
    return distances
def process_evaluation_data(data_dir):
    concatenatefiles([data_dir + "/trainDocForWMD.txt", data_dir + "/testDocForWMD.txt"],
                     data_dir + "/documents_lsi_data.txt")
    concatenatefiles([data_dir + "/trainProjectCategory.txt", data_dir + "/testProjectCategory.txt"],
                     data_dir + "/ProjectCategory.txt")
    concatenatefiles([data_dir + "/trainProjectDetails.txt", data_dir + "/testProjectDetails.txt"],
                     data_dir + "/ProjectDetails.txt")
    concatenatefiles([data_dir + "/trainProjectGitURL.txt", data_dir + "/testProjectGitURL.txt"],
                     data_dir + "/ProjectGitURL.txt")

def generate_and_save_term_doc_matrix(datapath):
    # Create some very short sample documents
    doc1 = 'John and Bob are brothers.'
    doc2 = 'John went to the store. The store was closed.'
    doc3 = 'Bob went to the store too.'
    # Initialize class to create term-document matrix
    tdm = textmining.TermDocumentMatrix()
    with open(datapath, "r") as ins:
        for line in ins:
            project_name, content = line.split("\t")
            tdm.add_doc(content)
    # Add the documents
    '''tdm.add_doc(doc1)
    tdm.add_doc(doc2)
    tdm.add_doc(doc3)'''
    # Write out the matrix to a csv file. Note that setting cutoff=1 means
    # that words which appear in 1 or more documents will be included in
    # the output (i.e. every word will appear in the output). The default
    # for cutoff is 2, since we usually aren't interested in words which
    # appear in a single document. For this example we want to see all
    # words however, hence cutoff=1.
    tdm.write_csv('matrix.csv', cutoff=1)
def get_tdm ( tdm_file_path):
    with open(tdm_file_path, "r") as ins:
        next(ins)#skip first header line
        array = []
        for line in ins:
            array.append(line)
    print "Array = ",array[0][20]
    print "Array Length", len(array)
    return array
def main():
    matrix = [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
              [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    process_evaluation_data("I:/Dev/PythonProjects/clan_data/project_data")
    generate_and_save_term_doc_matrix(
        "I:/Dev/PythonProjects/clan_data/project_data/documents_lsi_data.txt")  # with header
    matrix = get_tdm("matrix.csv")
    print "LSI algo started ....."
    transform_matrix = LSISimilarityMatrix(matrix)
    print "Dimension transform matrix = ", len(transform_matrix), " ", len(transform_matrix[0])

    doc_similarity_matrix = similarity(transform_matrix)
    print "Dimension doc_similarity_matrix = ", len(doc_similarity_matrix), " ", len(doc_similarity_matrix[0])

    search_result = get_rankedlist(1,doc_similarity_matrix)
    print search_result


if __name__ == "__main__":
	main()