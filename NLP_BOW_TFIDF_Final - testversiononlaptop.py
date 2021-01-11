# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:53:56 2019

@author: ball4472
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:06:15 2019

@author: ball4472
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:02:04 2019

@author: ball4472
"""



# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:05:26 2019

@author: ball4472
"""
#input info needed: 
# a folder with reports .docx
# folder with datasets   .csv   
# correlation table sql already created  
# ============================================================================= 
# output results:
# tf-idf for each word within a document  
# ranking based on bow for each document 
# retain only 1st 7 words most informative in wordrep4
# directly added to sql as a table
# =============================================================================
# improved lemmatizer with tf-idf but manually implemented then select 15 words 
#after disregarding tf-idf scores less than 0.0019
#importing useful libraries

import mysql.connector #connect to database
import re
import csv

import nltk # preprocessing text
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter  #create dictionary for bow


import numpy as np #maths equations
import docxpy #word extraction of text

#defining tf-idf functions

def computeTF(wordDict, lenbow):
    # wordDict is the dictionary containing words and number of appearances withing all the documents
    # create a dictionary for tf scores
    tfDict = {} #create dictionary of tf scores for each document
    for word, count in wordDict.items():
        #for each word in a document calculate tf score
        tfDict[word] = count/float(lenbow)
    return tfDict


def computeIDF(docList):
    #docList represents a list of dictionaries, each corresponding to a document, of the form (word, noApp)
    import math
    idfDict = {} #create a dictionary of idf scores for each word
    N = len(docList) #number of documents in the corpus <=> number of dictionaries in the list
    
    idfDict = dict.fromkeys(docList[0].keys(), 0) #initialises each word's count to 0
    for doc in docList:
        #for each dictionary in the list
        for word, val in doc.items():
            # for each word and number of appearances in a document
            if val > 0:
                #if the words shows up in the document
                idfDict[word] += 1 #increase the number of appearances of the word in the corpus
    
    for word, val in idfDict.items():
        # for each word in corpus calculate idf score
        idfDict[word] = math.log10(N / float(val)) #update the dictionary values
        
    return idfDict


def computeTFIDF(tfBow, idfs):
    tfidf = {} # create dictionary of tf-idf scores for each word
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

lemmatizer = WordNetLemmatizer()

def lema(word,posw):
    #takes word and part of speach and creates a label specifially saying what that part
    #of speach is. useful for pr
    if 'NN' in posw:
        x=wordnet.NOUN
    else:
        if 'VB' in posw:
            x=wordnet.VERB
        else:
            if 'JJ' in posw:
                x=wordnet.ADJ
            else:
                x=wordnet.ADV
    return lemmatizer.lemmatize(word, x)


def prepro(file):
    # extract text and preprocesses and adds part of speach
    text = docxpy.process(file) #extract text
    allwords=[w for w in word_tokenize(text.lower()) if w.isalpha()] #tokenize and discard non words
    #discard stopwords
    no_stops = [w for w in allwords if w not in stopwords.words('english')]
    vec=nltk.pos_tag(no_stops) #add part of speach tagging
    return vec
# now move data to sql
####################################################################
#connect to database
cnx = mysql.connector.connect(user='root', password='sqlAmonouaparola213',
                          host='127.0.0.1',
                          database='final')
#get data from temperature sensor
mycursor = cnx.cursor()
sql="delete from wordrep100" #wordrep100 saves number of report, first 15 words with tfidf>0.0019 and no of appearances
mycursor.execute(sql)
mycursor = cnx.cursor()
sql="delete from wordrep200" #wordrep200 saves number of report, first 15 words with any tfidf
mycursor.execute(sql)
sql="select nameRep from corr" # select reports from table which contains number of report, title and name of data file associated
mycursor.execute(sql)
results=mycursor.fetchall() # create vector of report names
lengthsOfvec=[] #list containing lengths of each document vector of words
new=[] # list containing all lematized words from the preprocessed documents from all documents
wordset=[] # list containing only one of each word from all preprocessed documents

for namerep in results:
      #namerep is the name of the report
    file = 'Reports/'+namerep[0] #find path to report
    vec=prepro(file) # create vector containing preprocessed words and part of speach tags
    
    vnew=[] #vector containig lemmatized words of vec
    for val in vec:
        vnew.append(lema(val[0],val[1])) # for each word and pos in the report, add to vnew the lemmatized version of the word based on pos
    lengthsOfvec.append(len(vnew))   # add length of the word vector describing the document to a list
    line = ' '.join(vnew) #create a new string containing all lemmatized preprocessed words sepparated by space
    new.append(line) #concatenate the formed string (line) to a list containing all the documents
    wordset = set(wordset).union(set(vnew)) # discards words that appear more times, becomes corpus

listdic=[] # list of document's word vectors
lengt=len(results) # number of documents
listIDs=[] #list of ids of documents in order
for namerep in results:
    dictio=dict.fromkeys(wordset, 0) #create dictionaty with 0 for each word in corpus; this will be the document's word vector
    
    file = 'Reports/'+namerep[0] #location of report
    vec=prepro(file) #preprocess again
    
    vnew=[] #vector containig lemmatized words of vec
    for val in vec:
        wrd=lema(val[0],val[1]) #get lematized form of word
        dictio[wrd]+=1 #increase count for word 
    listIDs.append(namerep[0])
    
    listdic.append(dictio) #add document's word vector (actually dictionary) to the list of vectors
#andr=pd.DataFrame(listdic)

idf=computeIDF(listdic) #obtain idf dictionary
listTFIDF=[] #create list of tfidf scores
for i in range(0,lengt):
      # loop through documents
    tfs=computeTF(listdic[i], lengthsOfvec[i]) #for each document calculate tf scores of words
    listTFIDF.append(computeTFIDF(tfs,idf)) #append to list tfidf scores of each document
final=pd.DataFrame(listTFIDF) #create dataframe using pandas for visualisation can add listIDs
count=0 #counter to keep track of how many documents have been processed 
for namerep in results:
    
    file = "C:/Users/oncescu/OneDrive - Nexus365/Reports/"+namerep[0]
    vec=prepro(file)
    text = docxpy.process(file)
    vnewnew=[] # new list of words with any tfidf 
    vnew=[] # new list of words where tfidf>0.0019  
    # now create a new list of words for each document that contains words with tfidf>0.0019
    for val in vec:
        wrd=lema(val[0],val[1])
        if final[wrd][count]>0.0019:
            vnew.append(wrd)
        vnewnew.append(wrd)
    
    vnewnew = np.asarray(vnewnew) #transform list to array
    vnew = np.asarray(vnew) #transform list to array
    vn = Counter(vnew).most_common(15) # keep only most commonly met 15 words with tfidf>0.0019
    vnn= Counter(vnewnew).most_common(15) # keep only most commonly met 15 words with any tfidf
    #a = []
    #for i in range(len(vn)):
     #   a.append(vn[i][0])
    #print(a)
    v1=[] # find report number
    match_digits_and_words = ('(\d{5,115})') #define what youre looking for
    v1 = re.findall(match_digits_and_words, text) #actually find report number

    df = pd.DataFrame(vn) #create dataframe from vn
    df.to_csv("BagOfWords2.csv") #transform to csv file to then import in sql
    df = pd.DataFrame(vnn) #create dataframe from vn
    df.to_csv("BagOfWords3.csv") #transform to csv file to then import in sql
    
    
    ################ here update table with words with tfidf>0.0019
    input1 = open('BagOfWords2.csv', 'r') #open file to read from
    output = open('C:/Program Files/MySQL/first_edit2.csv', 'w', newline='') #create new file which contains info from input 1 and the report number
    z='C:/Program Files/MySQL/first_edit2.csv' #save the location of new file into variable z
    writer = csv.writer(output) #initialise a writing function
    for row in csv.reader(input1):
        if row[1]!='0':
            row[0]=v1[0]
            writer.writerow(row)
    input1.close()
    output.close()
    mycursor.execute("load data infile \'"+z+"\' into table wordrep100 fields terminated by \',\' enclosed by \'\"\' lines terminated by \'\\r\\n\' (MeasID, Word, NoApp)")
    bla=0
# =============================================================================
#     if namerep[0]=='Failure_Report137.docx': #this is for accelerometer ground fail 080320191037030137
#         bla=1
#       23022019203103029 #for gsr resistor burnt
# =============================================================================
    for i in range(0,len(vn)):
        mycursor.execute("update wordrep100 set TFIDF = "+str(listTFIDF[count][vn[i][0]])+" where Word = \'"+vn[i][0]+"\' and MeasID="+v1[0]) #add tfidf score
        mycursor.execute("update wordrep100 set count = "+str(i)+" where Word = \'"+vn[i][0]+"\' and MeasID="+v1[0]) # use this to be able to print wordrep100 and wordrep200 side by side

    
    ############## here update table with words with any tfidf
    input1 = open('BagOfWords3.csv', 'r') #open file to read from
    output = open('C:/Program Files/MySQL/first_edit3.csv', 'w', newline='') #create new file which contains info from input 1 and the report number
    z='C:/Program Files/MySQL/first_edit3.csv' #save the location of new file into variable z
    writer = csv.writer(output) #initialise a writing function
    for row in csv.reader(input1):
        if row[1]!='0':
            row[0]=v1[0]
            writer.writerow(row)
    input1.close()
    output.close()
    mycursor.execute("load data infile \'"+z+"\' into table wordrep200 fields terminated by \',\' enclosed by \'\"\' lines terminated by \'\\r\\n\' (MeasID, Word, NoApp)")
    
    for i in range(0,len(vnn)):
        mycursor.execute("update wordrep200 set TFIDF = "+str(listTFIDF[count][vnn[i][0]])+" where Word = \'"+vnn[i][0]+"\' and MeasID="+v1[0])
        mycursor.execute("update wordrep200 set count = "+str(i)+" where Word = \'"+vnn[i][0]+"\' and MeasID="+v1[0])
    
    cnx.commit()
    count+=1 
# =============================================================================
# from sklearn.metrics.pairwise import cosine_similarity
# dist = 1 - cosine_similarity(final)
# =============================================================================
from sklearn.cluster import KMeans
ok=0
it=0
indccl=[3,20,35,68,71,76,90]
listcl2=[]
listcl1=[]
for i in indccl:
    listcl1.append(np.asarray(final.loc[i+1,:]))
arrayn1=np.asarray(listcl1)
for i in indccl:
    listcl2.append((np.asarray(final.loc[i+2,:])+np.asarray(final.loc[i+1,:]))/2)
arrayn2=np.asarray(listcl2)
# =============================================================================
# while ok==0:
# =============================================================================
it+=1
num_clusters = 7

# =============================================================================
# km = KMeans(n_clusters=num_clusters, init=arrayn2, max_iter=1) #init=arrayn to set initial cluster centers to labelled ones, when done initial test for report used 300
# =============================================================================
# =============================================================================
# km = KMeans(n_clusters=num_clusters,max_iter=10,random_state=32)
# =============================================================================
km = KMeans(n_clusters=num_clusters, init=arrayn2, max_iter=2, random_state=32)

km.fit(final)

clusters = km.labels_.tolist()
countl={}
countvec=[]
for i in range(0,num_clusters):
    countl[i]=0
for i in range(0,num_clusters):
    countvec.append(dict(countl))
# =============================================================================
# final2=final.sort_values(by=[117],axis=1)
# =============================================================================
label1=[]

tot1=0
label2=[]

tot2=0
label3=[]

tot3=0
label4=[]

tot4=0
label5=[]

tot5=0
label6=[]

tot6=0
label7=[]

tot7=0
maxim=dict(countl) #used to detect the main diagonal by finding the maximum entry for each manually set label

labels={}
#create confusion matrices by finding coresponding clusters
#at every run the computer changes the ids of the clusters and we want to find the correlation between our counting sysetem
for i in range(0,lengt):
    if (i>=0 and i<=11) or (41<=i and i<=46) or (59<=i and i<=64): #this is label 1
        countvec[0][clusters[i]]+=1 #unordered confusion matrix 
        if countvec[0][clusters[i]] >maxim[0]:
            maxim[0]=countvec[0][clusters[i]]
            l1=clusters[i]
        tot1+=1
        label1='gsr ground pin'
    else:
        if (i>=12 and i<=29) or (53<=i and i<=58): #this is label 2
            countvec[1][clusters[i]]+=1
            if countvec[1][clusters[i]] >maxim[1]:
                maxim[1]=countvec[1][clusters[i]]
                l2=clusters[i]
            tot2+=1
            label2='gsr analog pin'
        else:
            if (i>=30 and i<=40) or (47<=i and i<=52): #this is label 3
                countvec[2][clusters[i]]+=1
                if countvec[2][clusters[i]] >maxim[2]:
                    maxim[2]=countvec[2][clusters[i]]
                    l3=clusters[i]
                tot3+=1
                label3='gsr resistor burnt'
            else:
                if i==65 or (i>=74 and i<=79) or (i>=100 and i<=105): #this is label 4
                    countvec[3][clusters[i]]+=1
                    if countvec[3][clusters[i]] >maxim[3]:
                        maxim[3]=countvec[3][clusters[i]]
                        l4=clusters[i]
                    tot4+=1
                    label4='temperature ground pin'
                else:
                    if (i>=66 and i<=70) or (i>=106 and i<=111): #this is label 5
                        countvec[4][clusters[i]]+=1
                        if countvec[4][clusters[i]] >maxim[4]:
                            maxim[4]=countvec[4][clusters[i]]
                            l5=clusters[i]
                        tot5+=1
                        label5='acceleration power pin'
                    else:
                        if (i>=71 and i<=73) or (i>=80 and i<=81) or (i>=112 and i<=117): #this is label 6
                            countvec[5][clusters[i]]+=1
                            if countvec[5][clusters[i]] >maxim[5]:
                               maxim[5]=countvec[5][clusters[i]]
                               l6=clusters[i]
                            tot6+=1
                            label6='acceleration ground pin'
                        else:
                            countvec[6][clusters[i]]+=1 
                            if countvec[6][clusters[i]] >maxim[6]:
                               maxim[6]=countvec[6][clusters[i]]
                               l7=clusters[i]
                            tot7+=1
                            label7='humidity power pin'


val=0.75 #accuracy value; used when looping through to find the number of times needed to run the code to get above 75% accuracy
if l5==l6:        #sometimes accelerometer ground/power pin get clustered in the same cluster because of the reports similarity       
    labels[0]=l1
    labels[1]=l2
    labels[2]=l3
    labels[3]=l4
    labels[4]=l5
    labels[5]=21-l1-l2-l3-l4-l5-l7
    labels[6]=l7
    ok=0
else:
    labels[0]=l1
    labels[1]=l2
    labels[2]=l3
    labels[3]=l4
    labels[4]=l5
    labels[5]=l6
    labels[6]=l7
    if maxim[0]/tot1>val and maxim[1]/tot2>val and maxim[2]/tot3>val and maxim[3]/tot4>val and maxim[4]/tot5>val and maxim[5]/tot6>val and maxim[6]/tot7>val:
        ok=1 #used to stop the loop when if conditions are met

for i in range(0,7):
    for j in range(0,7):
        print(countvec[i][labels[j]], end=" ")
    print()
    
print(it)

