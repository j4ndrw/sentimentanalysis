#SENTIMENT ANALYZER.

#tensorflow = librarie utilizata pentru crearea grafurilor computationale si crearea modelelor specifice ML (Machine Learning)

#nltk = librarie folosita pentru manipularea cuvintelor
#tokenize = functie care sparge sirul de caractere in cuvinte
#stem = scoate terminatii specifice diferitelor acorduri si specifice timpurilor(e.g: Calling -> Call ; Called -> Call)
#lemmatizer = ia cuvantul fara terminatie si ii foloseste sensul din dictionar, tinand cont de context (e.g: "To call" poate sa ia forme de "called, calling")
#pickle = librarie specifica limbajului python pentru a salva date
#numpy = librarie folosita pentru modelarea vectorilor

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

n_noduri_hl1 = 500
n_noduri_hl2 = 500

n_clase = 2

batch_size = 1

x = tf.placeholder('float')
y = tf.placeholder('float')

hl1 = {'n_strat':n_noduri_hl1,
	   'weight':tf.Variable(tf.random_normal([2569, n_noduri_hl1])),
	   'bias':tf.Variable(tf.random_normal([n_noduri_hl1]))}

hl2 = {'n_strat':n_noduri_hl2,
	   'weight':tf.Variable(tf.random_normal([n_noduri_hl1, n_noduri_hl2])),
	   'bias':tf.Variable(tf.random_normal([n_noduri_hl2]))}

strat_output = {'n_strat':None,
			    'weight':tf.Variable(tf.random_normal([n_noduri_hl2, n_clase])),
			    'bias':tf.Variable(tf.random_normal([n_clase]))}

def model_retea_neuronala(data):

	l1 = tf.matmul(data, hl1['weight']) + hl1['bias']
	l1 = tf.nn.relu(l1)

	l2 = tf.matmul(l1, hl2['weight']) + hl2['bias']
	l2 = tf.nn.relu(l2)

	output = tf.matmul(l2, strat_output['weight']) + strat_output['bias']
	return output

#Definim SENTIMENT ANALYZER-UL!
def sentiment_analysis(input):
	predictie = model_retea_neuronala(x)
	with open('lexicon.pickle', 'rb') as f:
			lexicon = pickle.load(f)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		tf.train.Saver().restore(sess, "./reteaNeuronala.ckpt")

		cuvinte_actuale = word_tokenize(input.lower())
		cuvinte_actuale = [lemmatizer.lemmatize(i) for i in cuvinte_actuale]

		caracteristici = np.zeros(len(lexicon))

		for cuvant in cuvinte_actuale:
			if cuvant.lower() in lexicon:
				valoare_index = lexicon.index(cuvant.lower())
				caracteristici[valoare_index] += 1
		caracteristici = np.array(list(caracteristici))
		rezultat = (sess.run(tf.argmax(predictie.eval(feed_dict={x:[caracteristici]}), 1)))
		if rezultat[0] == 1:
			print('(+) Comentariu POZITIV:', input)
		elif rezultat[0] == 0:
			print('(-) Comentariu NEGATIV:', input)

running = True
while(running):
	sentence = input("-> Introduceti comentariul (English only): \n >>>")
	sentiment_analysis(sentence)
	valid = False
	while valid == False:
		decision = input("--- Doriti sa introduceti un comentariu nou? [y/n] --- \n")
		if decision == 'y':
			running = True
			valid = True
		elif decision == 'n':
			running = False
			valid = True
		else:
			print("Alegere invalida! :(")
			valid = False