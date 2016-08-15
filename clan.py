import numpy
import textmining
import csv
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
def generate_similarity(lsimatrix):
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

    numpy.savetxt("similarity_matrix.csv", S, delimiter=",")
def get_rankedlist(query_index, similarity_matrix,total_query_number):
    if query_index<0 or query_index >= len(similarity_matrix):
        print "Invalid query index"
        return -1
    unorder_list = similarity_matrix[query_index]

    distances = []
    for i in range(total_query_number, len(unorder_list)):#skip all queries from candidate document and similarity calculation
        distances.append((i, unorder_list[i]))
    distances.sort(key=lambda x: x[1],reverse=True)#sort in reverse order
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
def get_tdm (tdm_file_path):
    array = numpy.loadtxt(open(tdm_file_path,"rb"),delimiter=",",skiprows=1)
    return array
def get_similarity_matrix(sm_path):
    sim_matrix = numpy.loadtxt(open(sm_path,"rb"),delimiter=",")
    return sim_matrix
def readProjectDetails(projectDetailsFile):
	projectDetails = []
	projectName = []
	for line in open(projectDetailsFile):
		if '\t' in line:
			lineTokens = line.split('\t', 1)
			projectDetails.append(lineTokens[1])
			projectName.append(lineTokens[0])
		else:
			print 'no tap space: ', line
	return projectName, projectDetails
def get_category_stats(categoryfile, query_doc_number):
    category_stats = {}
    with open(categoryfile) as fc:
        for k in xrange(query_doc_number):
            next(fc)
        for line in fc:
            pname, pcategory = line.split("\t")
            pcategory = pcategory.replace("\n", "")
            count_key = category_stats.get(pcategory, 0)  # if not found will return zero
            category_stats[pcategory] = int(count_key) + 1
    return category_stats
def main():
    '''
    matrix = [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
              [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    '''
    #process_evaluation_data("I:/Dev/PythonProjects/clan_data/project_data")
    '''
    generate_and_save_term_doc_matrix(
        "I:/Dev/PythonProjects/clan_data/project_data/documents_lsi_data.txt")  # with header
    matrix = get_tdm("matrix.csv")

    print "LSI algo started ....."
    print "Matrix dim = ",len(matrix)
    transform_matrix = LSISimilarityMatrix(matrix)
    print "Dimension transform matrix = ", len(transform_matrix), " ", len(transform_matrix[0])
    generate_similarity(transform_matrix)
    '''

    doc_similarity_matrix = get_similarity_matrix("similarity_matrix.csv")
    print "Dimension doc_similarity_matrix = ", len(doc_similarity_matrix), " ", len(doc_similarity_matrix[0])
    data_dir = "I:/Dev/PythonProjects/clan_data/project_data/"
    project_name, project_description = readProjectDetails(data_dir+"/ProjectDetails.txt")
    project_name, project_category = readProjectDetails(data_dir + "/ProjectCategory.txt")
    project_name, project_giturl = readProjectDetails(data_dir + "/ProjectGitURL.txt")
    category_stats = get_category_stats(data_dir + "/ProjectCategory.txt",5)#first 5 is the query number, so training data start from 5th position starting from 0
    #print category_stats
    search_result = get_rankedlist(3,doc_similarity_matrix, 5)
    #print search_result

    #evaluation results
    save_file_search_results="clan_search_results"
    number_of_query = 5

    f = open(save_file_search_results + ".txt", "w")
    # f.write("Rank"+"\t"+"Project Name"+"\t"+"Project Description"+"\t"+"Github URL"+"\t"+"Category"+"\t"+"Judgement(0-5)"+"\n")
    # print "Rank"+"\t"+"Project Name"+"\t"+"Project Description"+"\t"+"Github URL"+"\t"+"Category"+"\t"+"Judgement(0-5)"+"\n"
    MAP = 0
    MapAt5 = 0
    MapAt1 = 0
    MapAt3 = 0

    P_MAP = 0
    P_MapAt5 = 0
    P_MapAt1 = 0
    P_MapAt3 = 0

    countQuery = 0
    for queryIndex in xrange(0, number_of_query):
        distances = get_rankedlist(3, doc_similarity_matrix, 5)#return sorted ranked
        topN = 10
        avgp = 0.0
        avgpAt5 = 0
        avgpAt1 = 0
        avgpAt3 = 0

        P_avgp = 0.0
        P_avgpAt5 = 0
        P_avgpAt1 = 0
        P_avgpAt3 = 0

        countRelavance = 0

        countRelavance_1 = 0.0
        countRelavance_3 = 0.0
        countRelavance_5 = 0.0
        countRelavance_10 = 0.0
#checking start from here
        true_relevane = category_stats.get(project_category[queryIndex].replace("\n", ""),
                                           0)  # minimum between n and total relevance document, n is top
        totalCategoryRelevance = min(topN, true_relevane)
        if totalCategoryRelevance > 0:
            countQuery = countQuery + 1  # include this query in the MAP calculation
        for i in range(1, len(distances)):
            if (project_category[queryIndex] == project_category[distances[i][0]]):
                countRelavance = countRelavance + 1
                countRelavance_10 = countRelavance_10 + 1

                avgp = avgp + (countRelavance * 1.0) / i;
                if i <= 5:
                    countRelavance_5 = countRelavance_5 + 1
                    avgpAt5 = avgpAt5 + (countRelavance * 1.0) / i;
                if i <= 3:
                    countRelavance_3 = countRelavance_3 + 1
                    avgpAt3 = avgpAt3 + (countRelavance * 1.0) / i;
                if i <= 1:
                    countRelavance_1 = countRelavance_1 + 1
                    avgpAt1 = avgpAt1 + (countRelavance * 1.0) / i;

            if i >= 10:
                break
        if totalCategoryRelevance > 0:
            avgp = (avgp * 1.0) / totalCategoryRelevance
            P_avgp = countRelavance_10 / 10.0
            P_MAP = P_MAP + P_avgp
            # f.write("Average Prevision@10 = "+str(avgp)+"\n")
            # print ("AVG@10"+str(avgp)+"\t" +str(true_relevane)+"\n")
            MAP = MAP + avgp

        totalCategoryRelevance_at_5 = min(5, true_relevane)
        if totalCategoryRelevance_at_5 > 0:
            avgpAt5 = (avgpAt5 * 1.0) / totalCategoryRelevance_at_5
            P_avgpAt5 = countRelavance_5 / 5.0
            P_MapAt5 = P_MapAt5 + P_avgpAt5
            MapAt5 = MapAt5 + avgpAt5
        totalCategoryRelevance_at_1 = min(1, true_relevane)
        if totalCategoryRelevance_at_1 > 0:
            avgpAt1 = (avgpAt1 * 1.0) / totalCategoryRelevance_at_1
            P_avgpAt1 = countRelavance_1 / 1.0
            P_MapAt1 = P_MapAt1 + P_avgpAt1
            MapAt1 = MapAt1 + avgpAt1
        totalCategoryRelevance_at_3 = min(3, true_relevane)
        if totalCategoryRelevance_at_3 > 0:
            avgpAt3 = (avgpAt3 * 1.0) / totalCategoryRelevance_at_3
            P_avgpAt3 = countRelavance_3 / 3.0
            P_MapAt3 = P_MapAt3 + P_avgpAt3
            MapAt3 = MapAt3 + avgpAt3
    if countQuery > 0:
        MAP = (MAP * 1.0) / countQuery
        MapAt5 = (MapAt5 * 1.0) / countQuery
        MapAt1 = (MapAt1 * 1.0) / countQuery
        MapAt3 = (MapAt3 * 1.0) / countQuery

        P_MAP = (P_MAP * 1.0) / countQuery
        P_MapAt5 = (P_MapAt5 * 1.0) / countQuery
        P_MapAt1 = (P_MapAt1 * 1.0) / countQuery
        P_MapAt3 = (P_MapAt3 * 1.0) / countQuery

    f.write(str(MapAt1) + "\t" + str(MapAt3) + "\t" + str(MapAt5) + "\t" + str(MAP) + "\t" + str(P_MapAt1) + "\t" + str(
        P_MapAt3) + "\t" + str(P_MapAt5) + "\t" + str(P_MAP) + "\n")
    print(str(MapAt1) + "\t" + str(MapAt3) + "\t" + str(MapAt5) + "\t" + str(MAP) + "\t" + str(P_MapAt1) + "\t" + str(
        P_MapAt3) + "\t" + str(P_MapAt5) + "\t" + str(P_MAP) + "\n")

if __name__ == "__main__":
	main()