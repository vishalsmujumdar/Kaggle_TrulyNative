from bs4 import BeautifulSoup
import sampler, beautifier, tfidf, naivebayes,VectorSpace, evaluation, randomforest
import sys, glob, os, heapq, csv, stat

classes = {'unsponsored':0, 'sponsored':1}

list_of_docs = {}

def build_samples(class_value, y_value, n, training_file, generate_html):
    '''
    :param class_attr: Column name in train csv that specifies class
    :param file_name_attr: Column name in train csv that specifies file name
    :param class_value: Name of class (In this case unsponsored and sponsored)
    :param y_value: The y value of the class (In this case 0 - unsponsored and 1 - sponsored)
    :param n: Number of sample files
    :param training_file: Name of the train csv
    :param generate_html: Boolean value to specify whether we want to generate the HTML file for the sampled html text files
    :return: A list of file names that get sampled. This is important because not all files sampled are available in the data.
    '''
    class_files = sampler.get_rows_of_class(training_file, y_value)
    sampled_files = sampler.get_sample(class_files, n)
    return sampler.collect_sample_data(sampled_files, class_value, generate_html=generate_html)

def build_samples_with_ratio(training_file, n, generate_html):
    return sampler.collect_samples_in_ratio(training_file, classes, n, generate_html)

def build_list_of_docs(train_directory, class_value, files):
    '''
    :param train_directory: Directory where sampled files are stored
    :param class_value: The class (In our example 'sponsored' or 'unsponsored'). This is used to get sampled files in specific class sub folder
        #{train_directory}/#{class_folder}/*
    :param files: List of all files
    :return:
    '''
    list_of_docs = {}
    for index, file_name in enumerate(files):
        file_path = glob.glob(train_directory + '/' + class_value + '/' + file_name)
        print(str(index)+': Building TextBlob for '+file_name)
        if not len(file_path) == 0:
            with open(file_path[0],'rb') as file:
                text = file.read()
                soup = BeautifulSoup(text, 'html.parser')
                list_of_docs[file_name] = beautifier.remove_stop_words_and_stem(soup)
    return list_of_docs

def build_prominent_word_list(list_of_docs):
    prominent_words = []
    word_counts = {}
    tfidf_scores = {}
    for index, doc_name in enumerate(list_of_docs):
        print(str(index)+": Processing words in document "+doc_name)
        text_blob = list_of_docs[doc_name]
        for each_word in text_blob.words:
            #tfidf_scores[each_word] = tfidf.tfidf(each_word, text_blob, list_of_docs)
            word_counts[each_word] = word_counts.get(each_word,0) + tfidf.word_count(text_blob, each_word)
        #sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        #temp_words_list = heapq.merge(prominent_words, [(v, k) for k, v in tfidf_scores.items()])
        #prominent_words = list(temp_words_list)
    #return prominent_words, word_counts
    return word_counts

def train_naive_bayes(training_file, train_directory):
    for of_class in classes.keys():

        print('Reading files from training data for Naive Bayes')
        file_samples = [os.path.basename(x) for x in glob.glob(train_directory+'/'+of_class+'/*.txt')]

        print('Building list of docs for '+of_class)
        list_of_docs[of_class] = list_of_docs[of_class] or build_list_of_docs(train_directory, of_class,file_samples) if list_of_docs.keys().__contains__(of_class) else build_list_of_docs(train_directory, of_class, file_samples)

        #prominent_words, word_counts = build_prominent_word_list(list_of_docs)
        word_counts = build_prominent_word_list(list_of_docs[of_class])

        naivebayes.init_params(of_class, word_counts)

        # sorted_words = tfidf.sorted_words(prominent_words)
        # sorted_words = list(set(sorted_words))

    naivebayes.calculate_prior_probabilities(training_file, classes)

def train_vector_space_model(train_directory):
    all_document_list = {}
    for of_class in classes.keys():
        print('Reading files from training data for Vector Space Model')
        file_samples = [os.path.basename(x) for x in glob.glob(train_directory+'/'+of_class+'/*.txt')]

        print('Building list of docs for '+of_class)
        list_of_docs[of_class] = list_of_docs[of_class] or build_list_of_docs(train_directory, of_class,file_samples) if list_of_docs.keys().__contains__(of_class) else build_list_of_docs(train_directory, of_class, file_samples)

        all_document_list[of_class] = (list_of_docs[of_class]).values()

    return all_document_list


def append_in_testCSV(output_directory,file_name, prediction, classifier):
    '''
    :param output_directory: Directory where testfileCSV and testresult CSV is stored
    :param file_name: Name of the test file
    :param prediction: Classifier result
    '''
    output_file_name = 'testDataCSV_'+classifier+'.csv'
    # os.chmod(output_directory+'/'+output_file_name,stat.S_IWRITE)
    reader = csv.reader(open(output_directory+ '/testDataCSV.csv', 'r',newline=''))

    writer_csv = csv.writer(open(output_directory+'/'+output_file_name, 'a',newline=''), delimiter=',')
    for row in reader:
        if (row[0] == file_name):
            row[2] = prediction
            writer_csv.writerow(row)

def calculate_evaluation_parameters(output_directory, classifier):
    accuracy,recall,precision,error_rate = evaluation.evaluator(output_directory, classifier)
    print('Parameters for '+classifier+':')
    print('accuracy is: ',accuracy)
    print('recall is: ',recall)
    print('precision is: ',precision)
    print('error_rate is: ',error_rate, '%')


def main(argv):
    sampler.input_raw_directory = argv[1]
    output_directory = argv[2]
    training_file_name = argv[3]

    # sampler.handle_output_directory(output_directory, clean_directory=False)
    sampler.training_file_path = sampler.input_raw_directory + '/' + training_file_name
    training_file = sampler.read_training_df(sampler.training_file_path)

    # print('Building samples in ratio')
    # samples = build_samples_with_ratio(training_file, 3000, False)

    train_directory = output_directory+'/train'

    test_files = [os.path.basename(x) for x in glob.glob(output_directory+'/'+'test'+'/*.txt')]
    testFileBlobs = build_list_of_docs(output_directory,'test',test_files)

    #===================================================================================================================
    #================================================    NAIVE BAYES    ================================================

    train_naive_bayes(training_file, train_directory)

    #for test_file_name, test_file_blob in testFileBlobs.items():
    for key in testFileBlobs:
        predicted_class = naivebayes.nb_classifier(testFileBlobs[key], classes)
        append_in_testCSV(output_directory,key,predicted_class, 'NB')

    #===================================================================================================================

    #===================================================================================================================
    #================================================    VECTOR SPACE    ===============================================

    all_document_list = train_vector_space_model(train_directory)
    # for of_class in ['unsponsored', 'sponsored']:
    #     file_samples = [os.path.basename(x) for x in glob.glob(train_directory+'/'+of_class+'/*.txt')]
    #     all_document_list[of_class] = (build_list_of_docs(train_directory, of_class,file_samples)).values()
    for index, key in enumerate(testFileBlobs):
        print(str(index)+': Testing file - '+key)
        predicted_class = VectorSpace.vectorspace_classifier(testFileBlobs[key].string,all_document_list)
        append_in_testCSV(output_directory,key,predicted_class, 'VSM')

    #===================================================================================================================

    #===================================================================================================================
    #================================================    RANDOMFOREST    ===============================================

    rfc = randomforest.random_forest_classifier(train_directory, classes)
    test_files = [os.path.basename(x) for x in glob.glob(output_directory+'/'+'test'+'/*.txt')]
    test_directory = output_directory+'/test'

    print('Testing Random Forest')
    for index, file_name in enumerate(test_files):
        print(str(index)+': Testing file - '+file_name)
        X_test = randomforest.extract_features(test_directory+'/'+file_name)
        Y_predict = rfc.predict([X_test])
        if(Y_predict[0] == 0):
            prediction = 'unsponsored'
        else:
            prediction = 'sponsored'
        append_in_testCSV(output_directory,file_name, prediction, 'RF')

    #===================================================================================================================
    #===========================================    EVALUATING CLASSIFIERS   ===========================================

    calculate_evaluation_parameters(output_directory, 'NB')

    calculate_evaluation_parameters(output_directory, 'VSM')

    calculate_evaluation_parameters(output_directory, 'RF')

    #===================================================================================================================

    # import code; code.interact(local=locals())

if __name__ == "__main__":
   main(sys.argv)

'''
    foreach c in ['sponsored', 'unsponsored'] do
        Collect samples ✓
        foreach file in samples do  ✓
            calculate tf idfs for all terms ✓
            sort according to TF IDF add to global dictionary ✓
            init params for naive bayes (word_freqs, vocab_counts, all_words_counts) ✓
            calculate conditional probabilities for all words


    Read test document ✓
    Remove stopwords and stem all words ✓
    For each word
        Calculate conditional probabilities ✓

    Calculate probability for each class for test document
    Guess class based on above probabilities

    P(C) = N(C)/N

    P(w|C) = (count(w,C)+1)/(count(C)+|V|)


'''