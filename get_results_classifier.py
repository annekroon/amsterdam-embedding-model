from lib import classification
import numpy as np
import argparse
import json
import pandas as pd
import seaborn as sns


parser = argparse.ArgumentParser(description='Compute Table and Figure summarizing results of pretrained word embedding models')
parser.add_argument('--dataset', type=str, required=False, default='vermeer', help='Path of dataset with annotated data to be classified')
parser.add_argument('--output', type=str, required=False, default='tables_figures/', help='Path of output file (CSV formatted classification scores)')
args = parser.parse_args()

print('Arguments:')
print('dataset:', args.dataset)
print('output.path:', args.output)
print()

def rename_models(x):
    variable_name = x
    if x == 'w2v_model_nr_5_window_10_size_300_negsample_5':
        variable_name = 'w2v 5 (w = 10, d = 300, ns = 5)'
    if x =='w2v_model_nr_1_window_5_size_300_negsample_5':
        variable_name = 'w2v 1 (w = 5, d = 300, ns = 5)'
    if x == 'w2v_model_nr_8_window_47615_size_100_negsample_5':
        variable_name = 'w2v 8 (w = sentence, d = 100, ns = 5)'
    if x =='w2v_model_nr_11_window_47615_size_300_negsample_15':
        variable_name = 'w2v 11 (w = sentence, d = 300, ns = 15)'
    if x =='w2v_model_nr_4_window_10_size_100_negsample_5':
        variable_name = 'w2v 4 (w = 10, d = 100, ns = 5)'
    if x == 'w2v_model_nr_9_window_47615_size_300_negsample_5':
        variable_name = 'w2v 9 (w = sentence, d = 300, ns = 5)'
    if x == 'w2v_model_nr_3_window_5_size_300_negsample_15':
        variable_name = 'w2v 3 (w = 5, d = 300, ns = 15)'
    if x == 'w2v_model_nr_10_window_47615_size_100_negsample_15':
        variable_name = 'w2v 10 (w = sentence, d = 100, ns = 15)'
    if x == 'w2v_model_nr_7_window_10_size_300_negsample_15':
        variable_name = 'w2v 7 (w = 10, d = 100, ns = 15)'
    if x == 'w2v_model_nr_6_window_10_size_100_negsample_15':
        variable_name = 'w2v 6 (w = 10, d = 100, ns = 15)'
    if x == 'w2v_model_nr_2_window_5_size_100_negsample_15':
        variable_name = 'w2v 2 (w = 5, d = 100, ns = 15)'
    if x == 'w2v_model_nr_0_window_5_size_100_negsample_5':
        variable_name = 'w2v 0 (w = 5, d = 100, ns = 5)'
    if x == 'baseline':
        variable_name = 'baseline model'
    if x == 'wiki.nl.vec':
        variable_name = 'pre-trained 01: Wiki FastText'
    if x == 'cow-320.txt':
        variable_name = 'pre-trained 02: COW small'
    return variable_name

def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val > 0.80 else 'black'
    return 'color: %s' % color

def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

def create_pointplot(df, x_value, y_value, hue=None,  col=None, row = None, wrap=None, size=8, aspect=1.5, title=""):
    p = sns.factorplot(x=x_value, y=y_value, kind='point',  hue=hue, row = row,
                       col=col, col_wrap=wrap, size=size, aspect=aspect, data=df, legend_out=False, palette=sns.color_palette('Set1'))
    p.fig.subplots_adjust(top=0.9)
    p.fig.suptitle(title, fontsize=16)
    p.despine(left=True)
    for ax in p.axes.ravel():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    return p

def per_type(x):
    var = x
    if x.split('~')[-1] == ' w2v_ET ' and x.startswith('w2v'):
        var = 'w2v_ET_mean'
    elif x.split('~')[-1] == ' w2v_svm' and x.startswith('w2v'):
        var = 'w2v_svm_mean'
    return var

df = pd.read_pickle("output/output_training_size_large_AEM_data_{}.pkl".format(args.dataset))

df['model'] = df['model'].map(rename_models)
df_n = pd.melt(df, id_vars = ['accuracy', 'train_size', 'model'], var_name='model classifier')
df_n['type'] = df_n['model'] + ' ~ ' + df_n['value'].astype(str)
df_n['n'] = 1

print("number of observations per model:\n\n")
print(df_n.groupby('model').agg({'n' : sum, 'accuracy': np.mean}))

df['type'] = df['model'] + ' ~ ' + df['classifier']
df_n = df.pivot(index = 'type', columns = 'train_size', values = 'accuracy')

tab = df_n.style.\
    applymap(color_negative_red).\
    apply(highlight_max)

# write results to Table

f=open("{}{}_table_results_classification.html".format(args.output, args.dataset),"w")
f.write(tab.render()) # df is the styled dataframe
f.close()
print("\n\n\nSave results\n\n\n")

# write results to Figure

a = df
a['new_label'] = a['type'].apply(per_type)
a = a.groupby(['new_label', 'train_size']).agg(np.mean)
a = a.reset_index(level=[0,1])

fig = create_pointplot(a, 'train_size', 'accuracy', hue='new_label', size=10, aspect=1.5,
                 title="accuracy across different classifiers")

fig.savefig("{}{}_figure_classification.png".format(args.output, args.dataset))
print("\n\n\nSave fig\n\n\n")
