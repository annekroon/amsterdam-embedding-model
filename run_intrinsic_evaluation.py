from src.analysis import intrinsic
import logging
import argparse
import json
import pandas as pd

def main(args):
	analyzer = intrinsic.word2vec_intrinsic_evaluation(path_to_embeddings = args.word_embedding_path, sample_size = args.word_embedding_sample_size, path_to_evaluation_data = args.path_to_evaluation_data)
	my_results = analyzer.get_scores()

	ed = {k:v for element in my_results for k,v in element.items()}
	d = {k: {k: v for ed in L for k, v in ed.items()} for k, L in ed.items()}
	df = pd.DataFrame.from_records(d).transpose()

	print('Created dataframe')
	print(df)

	df.to_pickle('{}_instrinic_evaluation_{}.pkl'.format(args.output, args.word_embedding_sample_size))

	with open('{}_instrinic_evaluation_{}.json'.format(args.output, args.word_embedding_sample_size),mode='w') as fo:
		fo.write('[')
		for result in my_results:
			fo.write(json.dumps(result))
			fo.write(',\n')
		fo.write('[]]')
		print("\n\n\nSave results\n\n\n")

if __name__ == '__main__':

	logger = logging.getLogger()
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
	logging.root.setLevel(level=logging.INFO)

	parser = argparse.ArgumentParser(description='Compute Intrinsic Accuracy of Pretrained Word Embedding Models')
	parser.add_argument('--word_embedding_sample_size', type=str, required=False, default = 'large', help='Size of sample of pretrained word embedding (small or large)')
	parser.add_argument('--word_embedding_path', type=str, required=True, help='Path of pretrained word embedding.')
	parser.add_argument('--output', type=str, required=False, default='output/output', help='Path of output file (JSON formatted scores)')
	parser.add_argument('--path_to_evaluation_data', type=str, required=False, default='data/raw/question-words.txt')
	args = parser.parse_args()

	print('Arguments:')
	print('word_embedding_sample_size:', args.word_embedding_sample_size)
	print('word_embedding_path:', args.word_embedding_path)
	print('path_to_evaluation_data:', args.path_to_evaluation_data)
	print('output.path:', args.output)
	print()

	main(args)
