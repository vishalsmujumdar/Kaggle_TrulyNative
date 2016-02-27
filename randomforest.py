'''
    Raw data:           HTML file
    Data Processing:    Calculate counts of tokens from HTML file to form arrays of features
    Classifier:         Train Random Forest classifier from input feature table
'''

from sklearn.ensemble import RandomForestClassifier
import re
import glob

def extract_features(file_path):
    file = open(file_path, 'r')
    text = file.read()
    values = []
    values.append(text.count('\n'))
    values.append(text.count(' '))
    values.append(text.count('\t'))
    values.append(text.count('{'))
    values.append(text.count('['))
    values.append(len(re.split('\s+', text)))
    values.append(len(text))
    values.append(text.count('<'))
    values.append(text.count('('))
    values.append(len(re.split('\w+', text)))
    values.append(text.count('\n\t'))
    values.append(text.count('//'))
    values.append(text.count('/'))
    values.append(text.count('<!'))
    values.append(text.count('/>'))
    values.append(text.count('&'))
    values.append(text.count(';'))
    values.append(text.count('=='))
    values.append(text.count('==='))
    values.append(text.count('css'))
    values.append(text.count('#'))
    values.append(text.count('@'))
    values.append(text.count('$'))
    values.append(text.count('%'))
    values.append(text.count('^'))
    values.append(text.count('+'))
    values.append(text.count('?'))
    values.append(text.count('|'))
    values.append(text.count('\\'))
    values.append(text.count('*'))
    values.append(text.count('||'))
    values.append(text.count('\t\t'))
    values.append(text.count('\t\t\t'))

    return values

def build_train_data(train_directory, classes):
    train_data = []
    train_classes = []
    for each_class in classes.keys():
        print('Processing files of class '+each_class)
        train_files = glob.glob(train_directory + '/' + each_class + '/*.txt')
        for index, each_train_file in enumerate(train_files):
            train_data.append(extract_features(each_train_file))
            train_classes.append(classes[each_class])
            print(str(index)+' Extracted features from '+each_train_file)

    return train_data, train_classes

def build_test_data(test_directory):
    test_data = []
    test_files = glob.glob(test_directory + '/*.txt')
    for index, each_test_file in enumerate(test_files):
        test_data.append(extract_features(each_test_file))
        print(str(index)+' Extracted features from '+each_test_file)

    return test_data

def random_forest_classifier(train_directory, classes):
    print('RANDOM FOREST CLASSIFIER')
    X_train, Y_train = build_train_data(train_directory, classes)
    rfc = RandomForestClassifier(n_jobs=-1)
    rfc = rfc.fit(X_train, Y_train)
    return rfc