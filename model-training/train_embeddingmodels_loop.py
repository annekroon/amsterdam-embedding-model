#df

    
import gensim
import logging
import re
import itertools

lettersanddotsonly = re.compile(r'[^a-zA-Z\.]')
PATH = "/home/anne/tmpanne/AEM_small_sample/"
FILENAME = "uniekezinnen_2000-01-01_2017-12-31"
# n sentences uniekezinnen_2000-01-01_2017-12-31: 4289076/4289076

def get_parameters(model_number):
    nr = int(model_number)

    window = [5, 10, 47615]
    negative = [5, 15]
    size = [100, 300]

    w2v_parameters = []

    for w, n, s in list(itertools.product(window, negative, size)):
        my_dict = {}
        my_dict['window'] = w
        my_dict['negative'] = n
        my_dict['size'] = s
        w2v_parameters.append(my_dict)

    return w2v_parameters[nr]

#################

def preprocess(s):
    s = s.lower().replace('!','.').replace('?','.')  # replace ! and ? by . for splitting sentences
    s = lettersanddotsonly.sub(' ',s)
    return s

class train_model():

    def __init__(self, model_nr):
        self.sentences = gensim.models.word2vec.PathLineSentences(PATH + FILENAME)
        self.w2v_params = get_parameters(model_nr)
        print("estimating model with the following parameter settings: {}".format(self.w2v_params))

        self.model = gensim.models.Word2Vec(**self.w2v_params)
        self.model.build_vocab(self.sentences)
        print('Build Word2Vec vocabulary for Model {}'.format(model_nr))
        self.model.train(self.sentences,total_examples=self.model.corpus_count, epochs=self.model.iter)
        print('Estimated Word2Vec model')

def train_and_save():
    for model_nr in range(0,12):
        logging.info("Starting w model:{}\n\n\n\nLET'S GO.".format(model_nr))
        parameters = get_parameters(model_nr)
        print("set parameters:", parameters)
        filename = "{}w2v_model_nr_{}_window_{}_size_{}_negsample_{}".format(PATH, model_nr, parameters['window'], parameters['size'],parameters['negative'] )

        casus = train_model(model_nr)

        with open(filename, mode='wb') as fo:
            casus.model.save(fo)
        print('Saved model')
        print("reopen it with m = gensim.models.word2vec.load('{}')".format(filename))

        del(casus)

if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    train_and_save()
