import gensim
import os
from collections import defaultdict

class word2vec_intrinsic_evaluation():
    '''This class tests the intrinsic accuracy of word2vec models.'''

    def __init__(self, path_to_embeddings, sample_size, path_to_evaluation_data):
        self.nmodel = 0
        self.samplesize = sample_size
        self.basepath = path_to_embeddings
        self.path_to_evaluation_data = path_to_evaluation_data

    def get_w2v_model(self):
        '''yields a dict with one item. key is the filename, value the gensim model'''

        filenames = [e for e in os.listdir(self.basepath) if not e.startswith('.')]

        for fname in filenames:
            model = {}
            path = os.path.join(self.basepath, fname)
            print("\nLoading gensim model")

            if fname.startswith('w2v'):
                mod = gensim.models.Word2Vec.load(path)
            else:
                mod = gensim.models.KeyedVectors.load_word2vec_format(path)

            model['gensimmodel'] = mod
            model['filename'] = fname
            self.nmodel +=1
            print("loaded gensim model nr {}, named: {}".format(self.nmodel, model['filename']))
            yield model

    def get_analogy_accuracy(self, model, evaluation_file):
        results = {}
        acc = []
        for i in model.wv.accuracy(evaluation_file):
            results[i['section']] = len(i['correct']) / (len(i['incorrect']) + len(i['correct'])) * 100
        acc.append(results)
        return acc

    def get_scores(self):
        final_results = []
        evaluation_data = defaultdict(int)
        for model in self.get_w2v_model():
            print('starting evaluation data')
            evaluation_data[model['filename']] = self.get_analogy_accuracy(model['gensimmodel'], self.path_to_evaluation_data)
            final_results.append(evaluation_data)
            print(final_results)
        return final_results
