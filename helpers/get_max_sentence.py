#!/usr/bin/env python3

''' returns the max length of sentences in the AEM corpus '''


path_to_corpus = "/Users/anne/surfdrive/Shared/JEDS/Data/AEM_corpus/"

with open(path_to_corpus + 'AEM_corpus') as f: 
    data = f.readlines()

print("\n\n\n .... Reading the data .....")
    
print("The longest sentence in the AEM corpus is: {}".format(max(map(len, data))))