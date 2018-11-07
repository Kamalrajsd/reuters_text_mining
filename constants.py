
PATH_DATA_DIR = 'data\\'
PATH_OUTPUT_DIR = 'output\\'
PATH_INTERIM_DIR = 'interim\\'
CLASS_SEP = '|'
XPATH_PLACES = './/PLACES'
XPATH_TOPICS = './/TOPICS'
XPATH_BODY = './/BODY'
XPATH_TITLE = './/TITLE'
CNT_ALPHABETS = 26
ASCII_A = 97
EMPTY_STRING = ''
ARR_IGNORED_CHAR = [
    r'<', r'>', r'=',r'_', r'\|',
    r'\^', r'\[', r'\]', r'\{', r'\}', r"'", r':', 
    r';', r'\d', r'!', r'#', r'\$' ,r'%', r'&', 
    r'\(', r'\)', r'\*', r'\.',r'\"', r'\\', r'\/', 
    r'\,', r'\+', r'\?', r'@', r'\s{1}-\s{1}'
 ]

PATH_LABELED_CSV = PATH_INTERIM_DIR + 'labled.csv'
PATH_PRED_TOPICS_CSV = PATH_INTERIM_DIR + 'predict_topics.csv'
PATH_PRED_PLACES_CSV = PATH_INTERIM_DIR + 'predict_places.csv'
PATH_OUT_TOPICS_CSV = PATH_OUTPUT_DIR + 'predicted_topics.csv'
PATH_OUT_PLACES_CSV = PATH_OUTPUT_DIR + 'predicted_places.csv'