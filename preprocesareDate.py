#PREPROCESARE DATE. PROCES UTILIZAT: NLP (Natural Language Processing)

#nltk = librarie folosita pentru manipularea cuvintelor
#tokenize = functie care sparge sirul de caractere in cuvinte
#stem = scoate terminatii specifice diferitelor acorduri si specifice timpurilor(e.g: Calling -> Call ; Called -> Call)
#lemmatizer = ia cuvantul fara terminatie si ii foloseste sensul din dictionar, tinand cont de context (e.g: "To call" poate sa ia forme de "called, calling")
#pickle = librarie specifica limbajului python pentru a salva date
#numpy = librarie folosita pentru modelarea vectorilor
#pandas = librarie folosita pentru analiza datelor

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd

#Definim lemmatizer-ul
lemmatizer = WordNetLemmatizer()


#Definim o functie pentru initializarea si crearea seturilor de date pentru ANTRENAMENT si TEST
def init_process(fin, fout):
	outfile = open(fout, 'a')
	with open(fin, buffering=200000, encoding='latin-1') as f:
		for rand in f:
			try:
				rand = rand.replace('"', '')

				#Definim ce inseamna POZITIV si NEGATIV conform datelor date.
				#Datele folosite sunt luate de pe site-ul Universitatii Stanford
				#care au delimitat polaritatile astfel: 0 - negativ, 4 - pozitiv
				polaritate_initiala = rand.split(',')[0]
				if polaritate_initiala == '0':
					polaritate_initiala = [1, 0]
				elif polaritate_initiala == '4':
					polaritate_initiala = [0, 1]

				tweet = rand.split(',')[-1]
				prefix_tweet = str(polaritate_initiala) + '>>>' + tweet
				outfile.write(prefix_tweet)
			except:
				pass
	outfile.close()

#Vom crea seturile de date pentru ANTRENAMENT si TEST
init_process('training.1600000.processed.noemoticon.csv', 'setAntrenament.csv')
init_process('testdata.manual.2009.06.14.csv', 'setTestare.csv')

#Definim o functie pentru crearea lexiconului
#lexicon = dictionar (in cazul nostru, set de cuvinte necesare analizei = nu vom lua in considerare cuvinte gen "the" sau "and" etc, ci doar cuvinte "cheie")
def creare_lexicon(fin):
	#Cream lexiconul
	lexicon = []

	#Vom deschide fisierul de interes cu encoding-ul "latin-1" pentru a nu lua in considerare caracterele ce nu sunt litere.
	#De asemenea vom lua un buffering foarte mare deoarece, lucrand cu 1.6mil de tweet-uri, procesul de creare a lexiconului va fi foarte incet.
	with open(fin, 'r', buffering=20000, encoding='latin-1') as f:
		try:
			cnt = 1
			continut = ''
			for rand in f:
				cnt += 1

				#Vom lua cate 2500 de cuvinte pe rand pentru ca vrem sa omitem cuvinte uzuale (e.g: "the", "and") si cuvinte rar folosite (e.g: "clepsydra")
				if(cnt / 2500.0).is_integer():
					tweet = rand.split('>>>')[1]
					continut += ' ' + tweet
					cuvinte = word_tokenize(continut)
					cuvinte = [lemmatizer.lemmatize(i) for i in cuvinte]
					lexicon = list(set(lexicon + cuvinte))
					print(cnt, len(lexicon))
		except:
			pass
	#Vom crea pickle-ul care contine lexiconul. De asemenea, il vom scrie in binar din moment ce acesta este un savefile ('wb' = write binary)
	with open('lexicon.pickle', 'wb') as f:
         pickle.dump(lexicon, f) 

#Vom crea setul de ANTRENAMENT
creare_lexicon('setAntrenament.csv')

#Definim o functie Word2Vec
#Word2Vec este un grup de modele care ne ajuta sa gasim relatii 
#intre un cuvant si posibilele lui contexte. Putem face asta prin 
#atribuirea unor caracteristici si a unor etichete fiecarui cuvant analizat.
def convert_to_vec(fin, fout, lexicon_pickle):

	#Deschidem pickle-ul pe care il citim in binar si il atribuim lexiconului cu care vom lucra. 'rb' = read binary
    with open(lexicon_pickle, 'rb') as f:
        lexicon = pickle.load(f)

    #Definim fisierul de iesire ca fiind gata de stocare a rezultatului. 'a' = append
    outfile = open(fout, 'a')
    with open(fin, buffering=20000, encoding='latin-1') as f:
    	cnt = 0
    	for rand in f:
    		cnt += 1
    		eticheta = rand.split('>>>')[0]
    		tweet = rand.split('>>>')[1]
    		cuvinte_actuale = word_tokenize(tweet.lower())
    		cuvinte_actuale = [lemmatizer.lemmatize(i) for i in cuvinte_actuale]

    		#Initializam un vector de zero-uri de marimea lexiconului in care vom stoca caracteristicile (e.g: caracteristici = 1 0 3 2....)
    		caracteristici = np.zeros(len(lexicon))

    		#Formam lista caracteristicilor
    		for cuvant in cuvinte_actuale:
    			if cuvant.lower() in lexicon:
    				valoare_index = lexicon.index(cuvant.lower())
    				caracteristici[valoare_index] += 1
    		caracteristici = list(caracteristici)

    		#Ii vom da forma outputului
    		#Ex. output: [1, 0] >>> @asdb "tha pizza was so dlicious"
    		#(greselile de scriere sunt intentionate deoarece 
    		#reteaua neuronala trebuie sa inteleaga contextul chiar daca 
    		#textul este imperfect)
    		prefix_tweet = str(caracteristici) + '>>' + str(eticheta) + '\n'
    		outfile.write(prefix_tweet)

    	print(cnt)

convert_to_vec('setTestare.csv', 'setTestareProcesat.csv', 'lexicon.pickle')

#Definim o functie care amesteca datele cu scopul de a nu lasa reteaua neuronala sa recurga la copierea datelor.
#Astfel, la fiecare iteratie, antrenamentul acesteia va fi diferit si rezultatul va fi mai precis.
def amesteca_date(fin):
	df = pd.read_csv(fin, encoding='latin-1', error_bad_lines=False)
	df = df.iloc[np.random.permutation(len(df))]
	print(df.head())
	df.to_csv('setAntrenamentAmestecat.csv', index=False)

amesteca_date('setAntrenament.csv')

#Definim o functie care va crea date de test conform caracteristicilor si etichetelor.
def creare_date_test(fin):
	set_caracteristici = []
	set_etichete = []
	cnt = 0
	with open(fin, buffering=20000) as f:
		for rand in f:
			try:
				caracteristici = list(eval(rand.split('>>')[0]))
				eticheta = list(eval(rand.split('>>')[1]))

				set_caracteristici.append(caracteristici)
				set_etichete.append(eticheta)
				cnt += 1
			except:
				pass
	print(cnt)
	set_caracteristici = np.array(set_caracteristici)
	set_etichete = np.array(set_etichete)

creare_date_test('setTestareProcesat.csv')