from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator,numpy

def generate_tfidf_vector(all_documents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_documents)
    return tfidf_matrix

def find_intersection(test_doc, training_doc):
    return set(test_doc).intersection(set(training_doc))

def calculate_cosine_simi(tfidf_matrix):
    cos_array = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    #print(cos_array)
    #print(numpy.amax(cos_array))
    return numpy.amax(cos_array)

def vectorspace_classifier(test_doc,training_documents):
    max_cosine_similarity = {}
    for each_class in ['sponsored','unsponsored']:
        all_docs = []
        all_docs.append(test_doc)
        for each_doc in training_documents[each_class]:
            all_docs.append(each_doc.string)
        vectore_space_matrix = generate_tfidf_vector(all_docs)
        #print(vectore_space_matrix.shape)
        #calculate_cosine_simi(vectore_space_matrix)
        max_cosine_similarity[each_class] = calculate_cosine_simi(vectore_space_matrix)
        #print(max_cosine_similarity[each_class])
    return max(max_cosine_similarity.items(), key=operator.itemgetter(1))[0]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''
    get the test Doc and extract its words
    for each class
        for each file of that class
            get common words with the test Doc
            generate a vector space 
            calculate the cosine similarity with the test doc
        save the max cosine similarity
    the class of the max(saved cosine similarities) = predicted class
'''