#!/usr/bin/env python3
import inca
import gensim
import logging
import re
from collections import defaultdict
import json

lettersanddotsonly = re.compile(r'[^a-zA-Z\.]')

PATH = "/home/anne/tmpanne/"

#outlets = ['ad (print)' , 'ad (www)' , 'anp' ,  'bd (www)' , 'bndestem (www)' , 'destentor (www)' , 'ed (www)' , 'fd (print)' , 'frieschdagblad (www)' , 'gelderlander (www)' , 'metro (print)' , 'metro (www)' ,  'nos' , 'nos (www)' , 'nrc (print)' , 'nrc (www)' , 'nu' , 'parool (www)' , 'pzc (www)' ,'spits (www)' , 'telegraaf (print)' , 'telegraaf (www)'  , 'trouw (print)' , 'trouw (www)' , 'tubantia (www)' , 'volkskrant (print)' , 'volkskrant (www)' , 'zwartewaterkrant (www)' ]

outlets = ['ad (print)' , 'fd (print)' ,  'metro (print)' ,  'nrc (print)' , 'telegraaf (print)' , 'trouw (print)' ,  'volkskrant (print)'  ]


def preprocess(s):
    s = s.lower().replace('!','.').replace('?','.')  # replace ! and ? by . for splitting sentences
    s = lettersanddotsonly.sub(' ',s)
    return s

class train_model():

    def __init__(self, doctype, fromdate,todate):
        self.doctype = doctype
        self.fromdate = fromdate
        self.todate = todate
        self.numberofarticles = defaultdict(int)
        self.numberoffailedarticles = defaultdict(int)
        self.failed_document_reads = 0
        self.documents = 0
        if type(self.doctype) is list:
            self.query = {
                  "query": {
                          "bool": {
                                    "filter": [ {'bool': {'should': [{ "term": { "doctype": d}} for d in self.doctype]}},
                                                { "range": { "publication_date": { "gte": self.fromdate, "lt":self.todate }}}
                                              ]
                                  }
                        }
                }

        self.sentences = set()
        for sentence in self.get_sentences():
            self.sentences.add(" ".join(sentence))



    def get_sentences(self):
        for d in inca.core.database.scroll_query(self.query):
            self.numberofarticles[d['_source']['doctype']] +=1
            try:
                self.documents += 1
                sentences_as_lists = (s.split() for s in preprocess(d['_source']['text']).split('.'))
                for sentence in sentences_as_lists:
                    yield sentence
            except Exception as e:
                # print(e)
                logger.debug(d['_source']['doctype'])
                self.numberoffailedarticles[d['_source']['doctype']] +=1
                self.failed_document_reads +=1
                continue

def train_and_save(fromdate,todate,doctype):
    filename = "{}uniekezinnen_{}_{}".format(PATH,fromdate,todate)

    casus = train_model(doctype,fromdate,todate)

    with open(filename, mode='w') as fo:
        for sentence in casus.sentences:
            fo.write(sentence)
            fo.write('\n')
    with open(filename+'_numberofarticles.json', mode = 'w') as fo:
        json.dump({"succes":casus.numberofarticles, "fail":casus.numberoffailedarticles}, fo)

    print('Created file with sentences: {}'.format(filename))


if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    train_and_save(fromdate = "2000-01-01", todate = "2017-12-31", doctype = outlets)
