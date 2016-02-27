'''
    This module is used to sample data from huge input data of 300000 files spread across
    5 folders.
'''

import os, sys, glob, shutil
import pandas as pd
import numpy, math
from bs4 import BeautifulSoup as bs

processed_data_path = ''
input_raw_directory = ''
training_file_path = ''

class_attr = 'sponsored'
file_name_attr = 'file'

def read_file(path):
    # Returns a pandas DataFrame object as the file
    return pd.read_csv(path)

def clean_up_folder(folder):
    # Takes a folder and deletes the sub folder tree
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def handle_output_directory(output_directory_path, clean_directory=True):
    global processed_data_path
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)
    processed_data_path = output_directory_path
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    if clean_directory:
        clean_up_folder(output_directory_path)
    return

def read_training_df(file_path):
     return read_file(file_path)

def get_rows_of_class(training_df, y):
    return training_df[training_df[class_attr]==y]

def get_sample(df, num_of_samples):
    return df.sample(num_of_samples)

def get_samples_of_class(df, y, num_of_samples):
    return get_sample(get_rows_of_class(df, y), num_of_samples)

def collect_sample_data(df, class_value, generate_html=False):
    file_names = df[file_name_attr]
    class_files_folder = processed_data_path+'/train/'+class_value
    class_files_html_folder = class_files_folder+'/html'

    if not os.path.exists(class_files_folder):
        os.makedirs(class_files_folder)
    if generate_html and not os.path.exists(class_files_html_folder):
        os.makedirs(class_files_html_folder)
    sample_names = []
    for index, file in enumerate(file_names):
        input_file = glob.glob(input_raw_directory+'/*/'+file)

        if not len(input_file) == 0:
            shutil.copyfile(input_file[0], class_files_folder+'/'+file)
            if generate_html:
                html_file_name = file.split('.')[0] + '.html'
                shutil.copyfile(input_file[0], class_files_html_folder+'/'+html_file_name)
            sample_names.append(file)
            print(str(index)+": Created: "+file)
        else:
            print(str(index)+": File "+file+" is not available")
    return sample_names

def count_of_class(training_file, y):
    return len(get_rows_of_class(training_file, y))

def collect_samples_in_ratio(training_df, classes, num_of_samples, generate_html):
    '''
    This method will take the original training file as an input and divide num_of_samples in the ratio of classes
    in the training file, for example - If num_of_samples is 1000 according to input ratio it will fetch say
    Class_1: 9/10 = 900 files, Class_2: 1/10 = 100 files and so on by calling get_sample() rows.
    Then collect samples from the sampled dfs returned by get_samples() by calling collect_sample_data()
    :param training_df: The input training file which has the original ratio of the two classes.
    :param classes: A list of the classes in the data
    :return:
    '''
    total_num_of_rows = len(training_df)
    samples_of = {}
    for each_class, y_value in classes.items():
        num = math.floor(num_of_samples * count_of_class(training_df, y_value)/total_num_of_rows)
        samples_df = get_samples_of_class(training_df, y_value, num)
        print('Collecting samples of class - '+each_class)
        samples_of[each_class] = collect_sample_data(samples_df, each_class, generate_html)

    return samples_of

def fetch_random_file_for(y=None):
    df = read_training_df(training_file_path)
    if y is None:
        random_index = numpy.random.choice(df.index)
    else:
        random_index = numpy.random.choice(get_rows_of_class(df, y).index)
    random_row = df.ix[random_index]
    file_path = glob.glob(input_raw_directory+'/*/'+random_row.file)
    try:
        random_file = open(file_path[0], 'rb')
        print('File: '+random_row.file)
        return random_file
    except:
        print('Could not find file corresponding to selected sample. Re-trying...')
        fetch_random_file_for(y)
        # print("Unexpected error:", sys.exc_info()[0])
        # raise

def fetch_random_soup_for(y=None):
    try:
        soup = bs(fetch_random_file_for(y), 'html.parser')
        return soup
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

def sample_test_data(training_df, num_of_samples):
    testing_files = get_sample(training_df,num_of_samples)
    test_data_path = input_raw_directory+'/'+'Test'
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    unavailable_file_indices = []

    for index, file_name in enumerate(testing_files[file_name_attr].tolist()):
        test_file = glob.glob(input_raw_directory+'/*/'+file_name)
        if not len(test_file) == 0:
            shutil.copyfile(test_file[0], test_data_path+'/'+file_name)
            print(str(index)+": Created: "+file_name)
        else:
            unavailable_file_indices.append(index)
            print(str(index)+": File "+file_name+" is not available")

    testing_files = testing_files.drop(unavailable_file_indices)
    testing_files['output_classes'] = ''
    testing_files.to_csv(test_data_path+'/testDataCSV.csv', sep=',', index=False)

def sample_equal_classes_test_data(training_df, num_of_samples, classes):
    test_data_path = input_raw_directory+'/'+'Test'
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    unavailable_file_indices = []
    testing_files = []
    for each_class in classes.keys():
        testing_files.append(get_samples_of_class(training_df, classes[each_class], num_of_samples/2))

    testing_files = pd.concat(testing_files)

    for index, file_name in enumerate(testing_files[file_name_attr].tolist()):
        test_file = glob.glob(input_raw_directory+'/*/'+file_name)
        if not len(test_file) == 0:
            shutil.copyfile(test_file[0], test_data_path+'/'+file_name)
            print(str(index)+": Created: "+file_name)
        else:
            unavailable_file_indices.append(index)
            print(str(index)+": File "+file_name+" is not available")

    testing_files = testing_files.drop(unavailable_file_indices)
    testing_files['output_classes'] = ''
    testing_files.to_csv(test_data_path+'/testDataCSV.csv', sep=',', index=False)


def main(argv):
    '''
    :param argv:
        input_raw_directory: Path of raw input data
        output_directory: Path where output should be stored
        training_file_name: Name of the file with the classes that should be present in input raw directory
    :return:
    '''

    # If less than two arguments are specified print an error and return program
    if len(argv)<3:
        print("Usage: sampler <input_raw_directory> <output_directory> <training_file_name>")
        return

    else:
        global input_raw_directory
        input_raw_directory = argv[1]
        output_directory = argv[2]
        training_file_name = argv[3]

        # If input directory does not exist print an error and return the program
        if not os.path.exists(input_raw_directory):
            print(input_raw_directory," does not exist")
            return

        # If output directory does not exist create it and the create Processed folder in output directory
        handle_output_directory(output_directory, clean_directory=False)

        # Read training file from input path and training file name
        global training_file_path
        training_file_path = input_raw_directory + '/' + training_file_name
        training_file = read_training_df(training_file_path)

        ## Sampling files
        classes = {'unsponsored':0, 'sponsored':1}
        sample_equal_classes_test_data(training_file, 1000, classes)

if __name__ == "__main__":
   main(sys.argv)