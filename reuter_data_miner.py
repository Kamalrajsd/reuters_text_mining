import os
import xml.etree.ElementTree as ET
import re, unicodedata
from constants import *
import functools
from trie import Trie
import time
from nltk.stem import WordNetLemmatizer
import contractions
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import LinearSVC   
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

class ReuterDataMiner:
    obj_lemmatizer = WordNetLemmatizer()

    def __init__(self):
        # stores the distinct places
        self.dict_places = {}
        # stores the distinct topics
        self.dict_topics = {}
        # stores the count of articles with no places
        self.i_no_places_cnt = 0
        # stores the count of articles with no topics
        self.i_no_topics_cnt = 0
        # trie which holds the baseline word count
        self.obj_Trie = Trie()
        # stores the distinct words
        self.arr_words = set()

        # topics classifier
        obj_cls_topics = None
        # places classifier
        obj_cls_places = None
        
        # places vectorizer
        obj_tfidf_places = None

        # topics vectorizer
        obj_tfidf_topics = None
     
    
    def doPreprocessing(self, b_is_verbose):
        """
        Performs the preprocessing required compute baseline wordcounts
        Args:
            str_flag (str): flag to enable logging.
        """
        file_count = 1

        f_places = open(PATH_PRED_PLACES_CSV, 'w', newline='')
        f_topics = open(PATH_PRED_TOPICS_CSV, 'w', newline='')
        f_labeled = open(PATH_LABELED_CSV, 'w', newline='')
        f_places_writer = csv.writer(f_places, quotechar='"', quoting=csv.QUOTE_ALL)
        f_topics_writer = csv.writer(f_topics, quotechar='"', quoting=csv.QUOTE_ALL)
        f_labeled_writer = csv.writer(f_labeled, quotechar='"', quoting=csv.QUOTE_ALL)
        
        # iterates through the list of data files
        for str_file_name in os.listdir(PATH_DATA_DIR):
            str_file_path = PATH_DATA_DIR + str_file_name
            
            with open(str_file_path, 'r') as obj_file:
                
                # reads a file
                str_file_content = obj_file.read()

                # truncates noise in data which prevents reading data into XML objects 
                str_file_content = self._prepareFileForParsing(str_file_content)

                # maps data into XML objects
                obj_xml_data = ET.fromstring(str_file_content)

                # adds the number of articles with no places to a global variable
                self.i_no_places_cnt += self._countEmptyElement(obj_xml_data, XPATH_PLACES)

                # adds the number of articles with no topics to a global variable
                self.i_no_topics_cnt += self._countEmptyElement(obj_xml_data, XPATH_TOPICS)

                # appends the distinct places in the file to a dictionary of global distinct places
                self.dict_places = { **self.dict_places, **self._computeDistElement(obj_xml_data, XPATH_PLACES) }

                # appends the distinct topics in the file to a dictionary of global distinct topics
                self.dict_topics = { **self.dict_topics, **self._computeDistElement(obj_xml_data, XPATH_TOPICS) }

                arr_f_places = []
                arr_f_topics = []
                arr_f_labeled = []
                
                for obj_elem in  obj_xml_data.findall("REUTERS"):
                    
                    # clean the data to perform selection and transformation 
                    str_body_content, str_title_content = self._doCleanSymbols(obj_elem)
                    
                    str_body_content = self._expandAbbreviation(str_body_content)

                    arr_places = [p.text for p in obj_elem.findall('PLACES//D')]
                    
                    arr_topics = [t.text for t in obj_elem.findall('TOPICS//D')]

                    str_doc_id = obj_elem.attrib['NEWID']

                    if len(arr_places) == 0:
                        arr_f_places.append([str_doc_id, str_body_content, str_title_content])
                    if len(arr_topics) == 0:
                        arr_f_topics.append([str_doc_id, str_body_content, str_title_content])
                    if len(arr_places) != 0 or len(arr_topics) != 0:
                        for str_place in arr_places:
                            arr_f_labeled.append([str_doc_id, str_place, CLASS_SEP.join(arr_places), EMPTY_STRING, EMPTY_STRING, str_body_content, str_title_content])
                        for str_topics in arr_topics:
                            arr_f_labeled.append([str_doc_id, EMPTY_STRING, EMPTY_STRING, str_topics, CLASS_SEP.join(arr_topics), str_body_content, str_title_content])
                        

                    # construct the trie
                    self._populateTrie(str_body_content, str_doc_id) 
                
                f_labeled_writer.writerows(arr_f_labeled)
                f_places_writer.writerows(arr_f_places)
                f_topics_writer.writerows(arr_f_topics) 

            if b_is_verbose:
                print("Processed file: " + str(file_count))
            file_count += 1
        f_topics.close()
        f_labeled.close()
        f_places.close()

    def _doCleanText(self, str_body_content):
        # transform contractions to full version of the word
        # eg. I'm to I am; Could've to Could have   
        str_body_content = contractions.fix(str_body_content)
        
        # replace undesired symbols like , (comma), . (full stop), ? (question mark) with whitespace 
        # extensive list of symbols available, 
        str_regex = "|".join(ARR_IGNORED_CHAR)
        str_regex = '('+str_regex+')'
        str_body_content = re.sub(str_regex, r' ', str_body_content)

        # remove - (hypen) in places where a it is followed by a character 
        # eg. -word to word
        str_body_content = re.sub(r'(\s{1}-(\w))', r'\2', str_body_content)

        # remove - (hypen) in places where a it is preceded by a character 
        # eg. following- to following
        str_body_content = re.sub(r'((\w)-\s)', r'\2', str_body_content)

        # remove - (hyped) in places where is it not preceded and followed by a character
        str_body_content = re.sub(r'((-){2,})', r'', str_body_content)
        str_body_content = re.sub(r'(~)', r'', str_body_content)

        # remove whitespace and newlines
        str_body_content = re.sub(r'((\s|\n)+)', r' ', str_body_content)

        # remove unicode characters 
        str_body_content = unicodedata.normalize('NFKD', str_body_content).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        # convert the cleaned data into lower case
        str_body_content = str_body_content.lower()

        return str_body_content

    def _doCleanSymbols(self, obj_xml_data):
        """
        Removes symbols & unicode characters that are not of interest
        Args:
            obj_xml_data (obj): object mapped XML data
        """
        str_body_content = ''

        # extract content from all articles in a file
        for obj_elem in  obj_xml_data.findall(XPATH_TITLE):
            str_body_content += obj_elem.text

        str_title = self._doCleanText(str_body_content)

        for obj_elem in  obj_xml_data.findall(XPATH_BODY):
            str_body_content += obj_elem.text
        if str_body_content == '' and obj_xml_data.find('TEXT').text is not None and obj_xml_data.find('TEXT').text != '':
            str_body_content += obj_xml_data.find('TEXT').text
        
        str_body =  self._doCleanText(str_body_content)

        return str_body, str_title
        

    def _expandAbbreviation(self, str_body_content):
        """
        Replaces common abbreviations with expanded words
        Args:
            str_body_content (str): entire article content in string format
        """
        dict_abbreviations = {
            'dlrs': 'dollar',
            'mln': 'million',
            'pct': 'percent',
            'cts': 'cent',
            'shr': 'share',
            'u s ': ' usa ',
            'u k ': ' uk ',
            'vs': 'versus',
            'mths': 'month',
            'avg':'average'
        }
        # uses each of abbreviation mapping defined in the dictionary above
        for abb in dict_abbreviations:
            str_body_content = re.sub(abb, dict_abbreviations[abb], str_body_content)
        return str_body_content

    def _populateTrie(self, str_body_content, str_doc_num):
        """
        Lemmatizes the word to add the it to Trie
        Args:
            str_body_content (str): entire article content in string format
        """
        arr_clean_words = []

        #  tokenize the words
        arr_raw_words = str_body_content.split(" ")

        # for each word, lemmatization is performed 
        # and it is added to Trie
        for i, str_word in enumerate(arr_raw_words):
            # hyphenated words like three-time should be treated as two different words
            # while words like co-operate should be treated as a single word
            # hence an hyphenated word "word1-word2" here is considered 
            # as three words - word1, word2 and word1word2
            if '-' in str_word:
                arr_words = str_word.split('-')
                arr_words.append(str_word.replace('-', ''))
            else:
                arr_words = [str_word]

            # first, the word is treated as a noun
            # if the lemmatization happens the transformed word is inserted into Trie
            # else the word is treated as a verb and it is lemmatized.
            for w in arr_words:	
                str_lemm = self.obj_lemmatizer.lemmatize(w)
                if str_lemm == w:
                    str_lemm = self.obj_lemmatizer.lemmatize(w, pos='v')
                arr_clean_words.append(str_lemm)
                
                # insert the word into Trie
                self.obj_Trie.insertWord(str_doc_num, str_lemm)

        # add the distinct words from the file to the global set of distince words
        self.arr_words = self.arr_words.union(set(arr_clean_words))

    def _prepareFileForParsing(self, str_file_content):
        """
        Prepares each data file for parsing 
        Args:
            str_file_content (str): entire file content in string format
        """
        # removes the UNKNOWN tag which contains tag like elements eg. <SDR> on its own 
        # which prevents parsing of XML data
        str_file_content = re.sub(r'<UNKNOWN>(.|\n)+?<\/UNKNOWN>', r'', str_file_content)

        # remove invalid XML characters
        str_file_content = re.sub(r'(&#\d+;)', r'', str_file_content)

        # remove the DOCTYPE tage
        str_file_content = re.sub(r'<!DOCTYPE lewis SYSTEM "lewis.dtd">', r'', str_file_content)  

        # python's ElementTree XML parser requires a root XML tag in order to enable it 
        # parse XML data. Hence, here a dummy root tag is added
        str_file_content = "<data>" + str_file_content + "</data>"	
        return str_file_content  

    def _countEmptyElement(self, obj_xml_data, str_xpath):
        """
        Counts elements without children in the given XPATH  
        Args:
            obj_xml_data (obj): entire file content mapped as python objects
            str_xpath   (str): XPATH in string format
        """
        i_count = 0 
        # extracts all elements in the specified XPATH
        for obj_elem in obj_xml_data.findall(str_xpath):
            # checks if an element has children
            if len(obj_elem) == 0:
                # increments count when there are no children
                i_count+=1
        return i_count

    def _computeDistElement(self, obj_xml_data, str_xpath):
        """
        Identifies distinct text of elements in the given XPATH  
        Args:
            obj_xml_data (obj): entire file content mapped as python objects
            str_xpath   (str): XPATH in string format
        """
        dict_elem = {}
        # extracts all elements in the specified XPATH
        for obj_elem in obj_xml_data.findall(str_xpath + '/D'):
            # checks if the element's text is already present in the dictionary
            if obj_elem.text not in dict_elem:
                # adds the text to the dictionary if it is not already present
                dict_elem[obj_elem.text] = True
        return dict_elem

    def writeDistinctToFile(self, str_flag, str_file_name):
        """
        Writes PLACES/TOPICS to the specified file  
        Args:
            str_flag (str): flag that mentiones places/topics
            str_file_name   (str): file path
        """

        # identifies the dictionary to write from
        dict = {}
        if str_flag == 'places':
            dict = self.dict_places
        elif str_flag == 'topics':
            dict = self.dict_topics
        else:
            raise BaseException('Invalid flag:' + str_flag)

        # sorts the keys in the dictionary
        arr_keys = list(dict.keys())
        arr_keys.sort()
        with open(PATH_OUTPUT_DIR +  str_file_name, 'w') as f:
            # writes each key to the file
            for key in arr_keys:
                f.write(key + '\n')
            
    def writeWordCountToFile(self, str_file_name):
        """
        Writes word counts to the specified file  
        Args:
            str_file_name   (str): file path to write to 
        """

        # fetches the list of distinct words
        self.arr_words = list(self.arr_words)
        self.arr_words.sort()
        f = open(PATH_OUTPUT_DIR + str_file_name, 'w')

        # fetches its word count from Trie and
        # writes it to the file
        for w in self.arr_words:
            wc, dc = self.obj_Trie.getWordCount(w)
            f.write(w+ " : " + str(wc) + ", " + str(dc) + '\n')

    def getNoDataCount(self, str_flag):
        """
        Fetches count of articles with no places/topics  
        Args:
            str_flag   (str): flag that denotes places/topics
        """
        if str_flag == 'places':
            return self.i_no_places_cnt
        elif str_flag == 'topics':
            return self.i_no_topics_cnt
        else:
            raise BaseException('Invalid flag:' + str_flag)

    def trainModel(self):
        """
        Trains Topics & places classifier based using TfIdf as the feature vector
        """

        # reads labled data
        obj_df = pd.read_csv(PATH_LABELED_CSV, header=None, names=['id', 'places', 'all_places', 'topics', 'all_topics', 'body', 'title'])

        # constructs Topics classifier
        self.obj_tfidf_topics = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words='english')

        obj_df_topics = obj_df.dropna(subset=['topics'])

        obj_df_feat_topics_train, obj_df_feat_topics_test, obj_df_label_topics_train, obj_df_label_topics_test = train_test_split(obj_df_topics[['body', 'title']], obj_df_topics[['topics', 'all_topics']], test_size = 0.25)
        
        # construct tf-idf matrix
        obj_tfidf_topics_train = self.obj_tfidf_topics.fit_transform(obj_df_feat_topics_train['body'])

        # obj_mi_cv = CountVectorizer(max_df=0.95, min_df=2,
        #                              max_features=10000,
        #                              stop_words='english')
        # mi_vector = cv.fit_transform(obj_df_feat_topics_train['title'])

        # mi_feature_vector = mutual_info_classif(mi_vector, obj_df_label_topics_train['topics'], discrete_features=True)

        self.obj_cls_topics = LinearSVC().fit(obj_tfidf_topics_train, obj_df_label_topics_train['topics'])

        obj_tfidf_topics_test = self.obj_tfidf_topics.transform(obj_df_feat_topics_test['body'])


        arr_predictions_topics = self.obj_cls_topics.predict(obj_tfidf_topics_test)

        print(self.accuracy(arr_predictions_topics, obj_df_label_topics_test['all_topics']))

        # constructs Topics classifier
        self.obj_tfidf_places = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words='english')
        obj_df_places = obj_df.dropna(subset=['places'])

        obj_df_feat_places_train, obj_df_feat_places_test, obj_df_label_places_train, obj_df_label_places_test = train_test_split(obj_df_places[['body', 'title']], obj_df_places[['places', 'all_places']], test_size = 0.3)
        
        # construct tf-idf matrix
        obj_tfidf_places_train = self.obj_tfidf_places.fit_transform(obj_df_feat_places_train['body'])

        self.obj_cls_places = LinearSVC().fit(obj_tfidf_places_train, obj_df_label_places_train['places'])

        obj_tfidf_places_test = self.obj_tfidf_places.transform(obj_df_feat_places_test['body'])

        arr_predictions_places = self.obj_cls_places.predict(obj_tfidf_places_test)

        print(self.accuracy(arr_predictions_places, obj_df_label_places_test['all_places']))

        # cross-validation score
        # accuracies = cross_val_score(clf, features, labels, scoring='accuracy', cv=5)
        # for fold_idx, accuracy in enumerate(accuracies):
        #     entries.append(('L_SV', fold_idx, accuracy))

    def predict(self, str_flag):
        """
        Predicts the class based on the classifier requested 
        Args:
            str_flag   (str): classifier - topics/places
        """
        obj_cls = None
        str_in_file_path = '' 
        str_out_file_path = ''
        obj_tfidf_vectorizer = None

        if str_flag == 'topics':
            obj_cls = self.obj_cls_topics
            str_in_file_path = PATH_PRED_TOPICS_CSV
            str_out_file_path = PATH_OUT_TOPICS_CSV
            obj_tfidf_vectorizer = self.obj_tfidf_topics
        elif str_flag == 'places':
            obj_cls = self.obj_cls_places
            str_in_file_path = PATH_PRED_PLACES_CSV
            str_out_file_path = PATH_OUT_PLACES_CSV
            obj_tfidf_vectorizer = self.obj_tfidf_places
        else:
            raise BaseException('Invalid flag: '+ str_flag)

        obj_df_to_predict = pd.read_csv(str_in_file_path, header=None, names=['id', 'body', 'title'])        

        obj_df_to_predict['predicted_'+str_flag] = obj_cls.predict(obj_tfidf_vectorizer.transform(obj_df_to_predict['body']))

        obj_df_to_predict[['id', 'predicted_'+str_flag]].to_csv(str_out_file_path, index = False)

        
    def accuracy(self, predicted, class_set):
        """
        Computes the accuracy of the predicted data on test data 
        Args:
            predicted   (list): list containing predicted class lables
            class_set   (list): list containing the possible classes for the data
        """
        class_set = list(class_set)
        if len(predicted) != len(class_set):
            raise BaseException('Length not equal')
        right_pred  = 0
        for i, c in enumerate(predicted):
            if c in class_set[i]:
                right_pred += 1
        # print(str(right_pred)+ ' ' + str(len(predicted)) )
        return right_pred/ len(predicted)


# initialize the program
obj_mine = ReuterDataMiner()

start_time = time.time()
# perform preprocessing 
# set b_is_verbose to False if no status on preprocessing is needed
# obj_mine.doPreprocessing(b_is_verbose = True)

# trains the classifiers
obj_mine.trainModel()

obj_mine.predict("places")
obj_mine.predict("topics")
# # write distinct places to a file
# obj_mine.writeDistinctToFile('places', 'places.txt')

# # write distince topic to a file
# obj_mine.writeDistinctToFile('topics', 'topics.txt')

# # write baseline wordcount to a file
# obj_mine.writeWordCountToFile('output.txt')

# # print articles with no places to the screen
# print("No of articles with no places" + str((obj_mine.getNoDataCount('places'))))

# # print articles with no topics to the screen
# print("No of articles with no topics"+ str(obj_mine.getNoDataCount('topics')))

print("--- %s seconds ---" % (time.time() - start_time))


