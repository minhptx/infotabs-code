import sys
import os
import requests
import re,csv
import json
import numpy as np
from collections import OrderedDict
import inflect,argparse
import re, datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wget
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel, BertModel
from gensim.models import KeyedVectors
# # Import and download stopwords from NLTK.
from nltk.corpus import stopwords
from nltk import download

if not os.path.exists('GoogleNews-vectors-negative300.bin.gz'):
	pass


# Remove stopwords.
try:
	stop_words = stopwords.words('english')
except:
	download('stopwords')  # Download stopwords list.
	stop_words = stopwords.words('english')


inflect = inflect.engine()

def is_date(string):
	match = re.search('\d{4}-\d{2}-\d{2}', string)
	if match:
		return True
	else:
		return False

def write_csv(data,split):
    with open(args['save_dir']+split+'.tsv', 'at') as outfile:
        writer = csv.writer(outfile,delimiter='\t')
        writer.writerow(data)


def config(parser):
    parser.add_argument('--json_dir', default="./../../data/tables/json/", type=str)
    parser.add_argument('--data_dir', default="./../../data/infotabs_tsv/", type=str)
    parser.add_argument('--save_dir', default="./../../temp/wmdpremise", type=str)
    parser.add_argument('--splits',default=["train"],  action='store', type=str, nargs='*')
    parser.add_argument('--topk', default=3, type=int)
    return parser

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser = config(parser)
	args = vars(parser.parse_args())
	num_top = args['topk']

	model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
	model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.

	for split in args["splits"]:
		data = pd.read_csv(args['data_dir']+"infotabs_"+split+".tsv",sep="\t")

		with open(args['save_dir']+split+".tsv", 'wt') as out:
			writer = csv.writer(out, delimiter='\t')
			writer.writerow(["index","table_id","annotator_id","premise","hypothesis","label"])

		for index,row in data.iterrows():
			file = args['json_dir'] +str(row['table_id'])+".json"
			json_file = open(file,"r")
			data = json.load(json_file)

			data = {x: [str(a) for a in y] for x, y in data.items()}


			try:
				title = data["title"][0]
			except KeyError:
				print(row)
				exit()

			del data["title"]

			para = ""
			hypo = str(row['hypothesis'])
			dot_prods = []
			candidates = []

			sentlist = []
			dislist = []

			for key in data:
				line = ""
				values = data[key]

				if isinstance(key, tuple):
					key = " ".join(tuple)

				try:
					res = inflect.plural_noun(key)
				except:
					res = False
				if (len(values) > 1) or res:
					verb_use = "are"

					if is_date("".join(values)):
						para += title+" was "+ str(key) +" on "
						line += title+" was "+ str(key) +" on "
					else:
						try:
							para += "The "+str(key)+" of "+title+" "+verb_use+" "
							line += "The "+str(key)+" of "+title+" "+verb_use+" "
						except TypeError:
							print(row)
							print(key)
							print(title)
							exit()

					for value in values[:-1]:
						para += value +", "
						line += value +", "
					if len(values) > 1:
						para += "and "+values[-1] + ". "
						line += "and "+values[-1] + ". "
					else:
						para += values[-1] + ". "
						line += values[-1] + ". "
				else:
					verb_use = "is"
					if is_date(values[0]):
						para += title+" was "+ str(key) +" on "+values[0] +". "
						line += title+" was "+ str(key) +" on "+values[0] +". "
					else:
						para +="The "+str(key)+" of "+title+" "+verb_use+" "+values[0] +". "
						line +="The "+str(key)+" of "+title+" "+verb_use+" "+values[0] +". "

				premsent = line
				premsplit = premsent.lower().split()
				hyposplit = hypo.lower().split()

				distance = model.wmdistance(premsplit, hyposplit)

				dislist += [distance]
				sentlist += [premsent]

			dislist, sentlist = (list(t) for t in zip(*sorted(zip(dislist , sentlist))))
			newpara = ""

			for i in range(0,min(num_top,len(sentlist))):
				newpara += sentlist[i]

			label = row["label"]
			if row["label"] == "E":
				label = 0
			if row["label"] == "N":
				label = 1
			if row["label"] == "C":
				label = 2

			data = [index,row['table_id'],row['annotator_id'],newpara,row["hypothesis"],label]
			write_csv(data,split)
