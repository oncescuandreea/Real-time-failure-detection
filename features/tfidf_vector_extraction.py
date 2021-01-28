# -*- coding: utf-8 -*-
# used to be called TFIDF_final_scores_newlaptop

# same file as the previous version but use already extracted info to get the vectors for clustering
# input info needed: 
# a folder with reports .docx
# folder with datasets   .csv   
# correlation table sql already created  
# ============================================================================= 
# output results:
# tf-idf for each word within a document  
# ranking based on bow for each document
# directly added to sql as a table
# =============================================================================
# improved lemmatizer with tf-idf but manually implemented then select 15 words 
# after disregarding tf-idf scores less than 0.0019
# importing useful libraries

import mysql.connector  # connect to database

import nltk  # preprocessing text
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter, defaultdict

import numpy as np  # maths equations
import docxpy  # word extraction of text
import math
from utils.utils_ml import id_to_name
from pathlib import Path
import argparse
from utils.feature_extraction_utils import delete_table


def compute_tf(wordDict: dict, words_number: int):
    """
    Create a dictionary for tf scores for each document.
    @param wordDict: dictionary containing words and number of appearances withing all the documents
    @param words_number: number of total words in doc after lemmatization and removal of stop words
    @return: dictionary of tf scored for document
    """
    tf_dict = {}
    wordDict_items = wordDict.items()
    for word, count in wordDict_items:
        # for each word in a document calculate tf score
        tf_dict[word] = count / float(words_number)
    # import pdb; pdb.set_trace()
    return tf_dict


def compute_idf(documents_list):
    """
    Function generated IDF dictionary for the list of existing documents
    @param documents_list: list of dictionaries, each corresponding to a document, of the form (word, noApp)
    @return idf_dict: dictionary of idf scores for each word
    """
    n = len(documents_list)  # number of documents in the corpus <=> number of dictionaries in the list

    idf_dict = dict.fromkeys(documents_list[0].keys(), 0)  # initialises each word's count to 0
    for document in documents_list:
        for word, number_of_appearances in document.items():
            if number_of_appearances > 0:
                idf_dict[word] += 1  # increase the number of appearances of the word in the corpus

    for word, number_of_appearances in idf_dict.items():
        # for each word in corpus calculate idf score update the dictionary values
        idf_dict[word] = math.log10(n / float(number_of_appearances))

    return idf_dict


def compute_tf_idf(tf_dict: dict, idf_dict: dict):
    """
    Create dictionary of TF-IDF scores for each word in corpus
    @param tf_dict: TF dictionary for each document
    @param idf_dict: IDF word dictionary for whole corpus
    @return tf_idf: dictionary of tf-idf scores for each word in current document
    """
    tf_idf = {}
    tf_items = tf_dict.items()
    for word, tf_word_score in tf_items:
        tf_idf[word] = tf_word_score * idf_dict[word]
    return tf_idf


def word_lemmatization(word: str, word_pos, lemmatizer):
    """
    Takes word and part of speech and returns the lemmatized version of word
    @param word: word to be lemmatized
    @param word_pos: part of speech of given word
    @return: lemmatized version of given word
    """
    if 'NN' in word_pos:
        x = wordnet.NOUN
    else:
        if 'VB' in word_pos:
            x = wordnet.VERB
        else:
            if 'JJ' in word_pos:
                x = wordnet.ADJ
            else:
                x = wordnet.ADV
    return lemmatizer.lemmatize(word, x)


def text_preprocessing(file_path: Path):
    """
    Extract text, pre-process text and add part of speech to no_stop words.
    @param file_path: file location
    @return: vector of tuples of words and the corresponding POS
    """
    text = docxpy.process(str(file_path))
    file_words = [w for w in word_tokenize(text.lower()) if w.isalpha()]  # tokenize and discard non words
    no_stops = [w for w in file_words if w not in stopwords.words('english')]  # discard stopwords
    vec = nltk.pos_tag(no_stops)  # add part of speech tagging
    return vec


def create_corpus(report_names: list, lemmatizer, report_folder_location: Path):
    """
    Function generated the most relevant words for all documents in corpus
    @param report_names: list or report names
    @param lemmatizer: WordNetLemmatizer used to lemmatize all words in all documents
        based on their Part of Speech (pos)
    @return: @param number_of_words_per_document: list of number of lemmatized words for each doc
             @param new: list of string version of lemmatized words describing each document
             @param word_set: set of representative words for all documents
             @param lemmatized_document_words: dictionary of list of lemmatized words for each doc
    """
    number_of_words_per_document = []
    new = []
    word_set = []
    word_and_pos = defaultdict()  # dictionary of list of word-POS tuples for each document
    lemmatized_document_words = defaultdict(list)

    for report_name in report_names:
        report_path = report_folder_location / report_name[0]
        word_and_pos[report_name] = text_preprocessing(report_path)

        for word, pos in word_and_pos[report_name]:
            lemmatized_document_words[report_name].append(word_lemmatization(word, pos, lemmatizer))
        number_of_words_per_document.append(len(lemmatized_document_words[report_name]))
        # create a new string containing all lemmatized preprocessed words separated by space
        string_of_lemmatized_words = ' '.join(lemmatized_document_words[report_name])
        new.append(string_of_lemmatized_words)  # concatenate the formed string to a list containing all the documents
        # discard words that appear more times to create word_set corpus
        word_set = set(word_set).union(set(lemmatized_document_words[report_name]))

    return number_of_words_per_document, new, word_set, lemmatized_document_words


def create_document_wordsets(report_names: list,
                             word_set: set,
                             lemmatized_document_words: dict):
    """
    Function counts which words of the word set appear in each document
    @param report_names: list of report names
    @param word_set: set of most relevant words describing all documents
    @param lemmatized_document_words: dictionary of list of lemmatized words for each doc
    @return: @param document_word_vectors: vector of word appearances from the word set
    """
    document_word_vectors = []  # list of document's word vectors
    for report_name in report_names:
        # create dictionary with 0 for each word in corpus; this will be the document's word vector
        corpus_dictionary_for_report = dict.fromkeys(word_set, 0)
        for word in lemmatized_document_words[report_name]:
            corpus_dictionary_for_report[word] += 1
        # add document's word vector (actually dictionary) to the list of vectors
        document_word_vectors.append(corpus_dictionary_for_report)
    return document_word_vectors


def create_tf_idf_document_vectors(document_word_vectors: list,
                                   report_names: list,
                                   number_of_words_per_document: list):
    """
    Function takes word appearance frequency word set for each document and calculates the tf-idf
    word vector for each document.
    @param document_word_vectors: vector of word appearances from the word set
    @param report_names: list of report names
    @param number_of_words_per_document: list of number of relevant words per document
    @return: list of tf-idf document vectors
    """
    length = len(report_names)  # number of documents
    idf_dict = compute_idf(document_word_vectors)  # obtain idf dictionary
    list_tf_idf = []  # create list of tfidf scores
    for i in range(0, length):
        # loop through documents
        tf_dict = compute_tf(document_word_vectors[i], number_of_words_per_document[i])
        list_tf_idf.append(compute_tf_idf(tf_dict, idf_dict))  # append to list tfidf scores of each document
    return list_tf_idf


def filter_tf_idf_words(report_names, lemmatized_document_words, final, mycursor, cnx):
    """
    Function returns lemmatized representative words for each document before
    and after being filtered by the tfidf score. This is used to print
    representative words in the NLP section. Tables 3 and 4.
    @param report_names: list of report names
    @param lemmatized_document_words: dictionary of lists of lemmatized words
    @param final: panda dataframe to easier visualise word vectors and their
        corresponding report name
    """
    create_table_command = "CREATE TABLE wordrep300 (MeasID varchar(255), Word varchar(255), NoApp int, " \
                           "TFIDF float(8), count int)"
    try:
        print("Table wordrep300 is being created")
        mycursor.execute(create_table_command)
    except mysql.connector.errors.ProgrammingError:
        print("Table wordrep300 already created. Data will be replaced")
        delete_table('wordrep300', cnx)
        mycursor.execute(create_table_command)
    create_table_command = "CREATE TABLE wordrep400 (MeasID varchar(255), Word varchar(255), NoApp int, " \
                           "TFIDF float(8), count int)"
    try:
        print("Table wordrep400 is being created")
        mycursor.execute(create_table_command)
    except mysql.connector.errors.ProgrammingError:
        print("Table wordrep400 already created. Data will be replaced")
        delete_table('wordrep400', cnx)
        mycursor.execute(create_table_command)
    id2name = id_to_name(mycursor)
    name2id = {v: k for k, v in id2name.items()}
    for idx, report_name in enumerate(report_names):
        full_word_vector = []  # new list of words with any tfidf
        filtered_word_vector = []  # new list of words where tfidf>0.0019
        # now create a new list of words for each document that contains words with tfidf>0.0019
        for lemmatized_word in lemmatized_document_words[report_name]:
            if final[lemmatized_word][idx] > 0.0019:
                filtered_word_vector.append(lemmatized_word)
            full_word_vector.append(lemmatized_word)

        full_word_vector = np.asarray(full_word_vector)  # transform list to array
        filtered_word_vector = np.asarray(filtered_word_vector)  # transform list to array
        # keep only most commonly met 15 words with tfidf>0.0019
        filtered_word_vector_15 = Counter(filtered_word_vector).most_common(15)
        # keep only most commonly met 15 words with any tfidf
        full_word_vector_15 = Counter(full_word_vector).most_common(15)
        report_id = name2id[report_name[0]]

        add_to_sql_command = f"INSERT INTO wordrep300 (MeasID, Word, NoApp, TFIDF, count) VALUES " \
                             f"(%s, %s, '%s', '%s', '%s')"
        word_counter = 0
        for word, count in filtered_word_vector_15:
            word_vector = [report_id, str(word), count, float(final.loc[report_name][word]), word_counter]
            mycursor.execute(add_to_sql_command, word_vector)
            cnx.commit()
            word_counter += 1

        add_to_sql_command = f"INSERT INTO wordrep400 (MeasID, Word, NoApp, TFIDF, count) VALUES " \
                             f"(%s, %s, '%s', '%s', '%s')"
        word_counter = 0
        for word, count in full_word_vector_15:
            word_vector = [report_id, str(word), count, float(final.loc[report_name][word]), word_counter]
            mycursor.execute(add_to_sql_command, word_vector)
            cnx.commit()
            word_counter += 1


def save_to_sql(final2, mycursor, report_names, cnx):
    # create an sql table containing the tf-idf scores with column names being the vector of words describing all documents and the id column containing the name of the report
    addf = "CREATE TABLE tfidfpd2 (measID VARCHAR(255)"
    for ii in range(0, 258):
        addf = addf + ",_" + final2.columns[ii] + " Float(15,14)"
    addf = addf + ")"

    try:
        print("Table tfidfpd2 is being created")
        mycursor.execute(addf)
    except mysql.connector.errors.ProgrammingError:
        print("Table tfidfpd2 already created. Data will be replaced")
        delete_table('tfidfpd2', cnx)
        mycursor.execute(addf)

    # create the string representing the command for sql for adding values
    addf2 = "INSERT INTO tfidfpd2 (measID"

    for ii in range(0, 258):
        addf2 = addf2 + ",_" + final2.columns[ii]
    addf2 = addf2 + ") VALUES (%s"

    for ii in range(0, 258):
        addf2 = addf2 + ",'%s'"
    addf2 = addf2 + ")"

    # insert into the previously created table the tfidf scores
    col = final2.columns
    length = len(report_names)
    for ii in range(0, length):
        listToAdd = [final2.index[ii]]
        for el in col:
            listToAdd.append(float(final2[el][ii]))
        mycursor.execute(addf2, listToAdd)
        cnx.commit()


def main():
    lemmatizer = WordNetLemmatizer()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sql_password",
        type=str,
    )
    parser.add_argument(
        "--report_folder_location",
        type=Path,
        default='C:/Users/oncescu/OneDrive - Nexus365/Reports',
    )
    args = parser.parse_args()
    cnx = mysql.connector.connect(user='root', password=args.sql_password,
                                  host='127.0.0.1',
                                  database='final_bare')
    mycursor = cnx.cursor()

    # select reports from table which contains number of report, title and name of data file associated
    sql = "select nameRep from corr"
    mycursor.execute(sql)
    report_names = list(mycursor.fetchall())
    number_of_words_per_document, _, word_set, lemmatized_document_words = \
        create_corpus(report_names, lemmatizer, args.report_folder_location)

    document_word_vectors = create_document_wordsets(report_names, word_set, lemmatized_document_words)

    list_tf_idf = create_tf_idf_document_vectors(document_word_vectors, report_names, number_of_words_per_document)
    indexname = []
    for report_name in report_names:
        indexname.append(report_name[0])
    # create data frame and set index as indexname
    final = pd.DataFrame(list_tf_idf, index=indexname)
    save_to_sql(final, mycursor, report_names, cnx)
    filter_tf_idf_words(report_names, lemmatized_document_words, final, mycursor, cnx)


if __name__ == '__main__':
    main()
