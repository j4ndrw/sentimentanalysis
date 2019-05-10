#CREAREA, ANTRENAREA SI TESTAREA RETELEI NEURONALE PENTRU SENTIMENT ANALYZER.

#tensorflow = librarie utilizata pentru crearea grafurilor computationale si crearea modelelor specifice ML (Machine Learning)

#nltk = librarie folosita pentru manipularea cuvintelor
#tokenize = functie care sparge sirul de caractere in cuvinte
#stem = scoate terminatii specifice diferitelor acorduri si specifice timpurilor(e.g: Calling -> Call ; Called -> Call)
#lemmatizer = ia cuvantul fara terminatie si ii foloseste sensul din dictionar, tinand cont de context (e.g: "To call" poate sa ia forme de "called, calling")
#pickle = librarie specifica limbajului python pentru a salva date
#numpy = librarie folosita pentru modelarea vectorilor

import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

#Pentru aceasta retea neuronala am folosit 2 straturi ascunse a cate 500 de noduri prin care se vor calcula rezultatele.
#Cu cat straturile sunt mai multe, cu atat rezultatul este mai precis, insa procesul va dura mai mult si va avea nevoie de 
#mai mult RAM si putere din parte procesorului / placii video
#Aceastei retele neuronale i-a luat ~24h sa isi termine antrenamentul.
n_noduri_hl1 = 500
n_noduri_hl2 = 500

#Numarul de clase necesare este 2 deoarece reteaua noastra neuronala va decide daca un comentariu este POZITIV sau NEGATIV.
n_clase = 2

#Batch = succesiune / serie / iteratie.
#Batch_size = 1 = online learning (ia fiecare bucatica de data pe rand si o analizeaza)
#Daca batch_size-ul era mai mare, erau analizate batch_size / total_batch-uri date deodata
batch_size = 1
total_batchuri = int(1600000 / batch_size)

#Epoch = ciclu de antrenament
cate_epochuri = 10

#Am definit 2 placeholdere pe care le vom folosi mai tarziu in antrenamentul retelei si in testarea acesteia
x = tf.placeholder('float')
y = tf.placeholder('float')

#Definim straturile
#tf.random_normal returneaza un tensor (vector) de o anumita forma(dimensiune. e.g [x,y]) cu valori random.
#Weight: matrice
#Bias: vector
#Weight-ul in cazul nostru va fi lungimea lexiconului X numarul de noduri al stratului.

hl1 = {'n_strat':n_noduri_hl1,
	   'weight':tf.Variable(tf.random_normal([2569, n_noduri_hl1])),
	   'bias':tf.Variable(tf.random_normal([n_noduri_hl1]))}

hl2 = {'n_strat':n_noduri_hl2,
	   'weight':tf.Variable(tf.random_normal([n_noduri_hl1, n_noduri_hl2])),
	   'bias':tf.Variable(tf.random_normal([n_noduri_hl2]))}

strat_output = {'n_strat':None,
			    'weight':tf.Variable(tf.random_normal([n_noduri_hl2, n_clase])),
			    'bias':tf.Variable(tf.random_normal([n_clase]))}

#Definim modelul pe care il vom folosi la antrenament
def model_retea_neuronala(data):

	#layer = weight * x + bias, unde x = input (valoare)
	#ReLU = Rectified Linear Unit: ReLU(x) = {0, x < 0; 
	# 										  x, x >= 0}

	l1 = tf.matmul(data, hl1['weight']) + hl1['bias']
	l1 = tf.nn.relu(l1)

	l2 = tf.matmul(l1, hl2['weight']) + hl2['bias']
	l2 = tf.nn.relu(l2)

	output = tf.matmul(l2, strat_output['weight']) + strat_output['bias']
	return output

#Vom crea un checkpoint saver in care ne vom salva modelul odata ce isi termina antrenamentul pentru a nu trebui sa se antreneze inca odata la a 2-a sau a x-a rulare
#De asemenea vom defini un logfile in care isi va scrie cate epochuri a completat.
#Odata terminat, modelul va verifica checkpoint-ul si logfile-ul si va decide daca a terminat antrenamentul sau nu
saver = tf.train.Saver()
log_file = 'log_file'

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
config.intra_op_parallelism_threads = 44


#Definim functia de ANTRENAMENT
def antreneaza_retea_neuronala(x):
	predictie = model_retea_neuronala(x)

	#Cost sau Loss = marja de eroare
	#reduce_mean = medie aritmetica
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = predictie, labels = y))

	#Optimizatorul face in asa fel incat sa minimizeze erorile facute, astfel spunandu-i retelei cand face ceva bine si cand nu.
	#AdamOptimizer-ul se mai numeste si Gradient Descent pentru ca minimizeaza costul.
	#Acest timp de antrenament se numeste SUPRAVEGHEAT (Supervised learning) tocmai pentru ca exista un optimizator care regleaza rezultatele
	#De obicei, cu cat learning rate-ul este mai mic, cu atat modelul este mai precis, insa cazul variaza depinzand de operatiile necesare
	optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		try:
			#Scriem in logfile la ce epoch suntem
			epoch = int(open(log_file,'r').read().split('\n')[-2])+1
			print('Incepe:',epoch)
		except:
			epoch = 1

		while epoch <= cate_epochuri:
			if epoch != 1:
				saver.restore(sess, "./reteaNeuronala.ckpt")
			epoch_loss = 1
			with open('lexicon.pickle', 'rb') as f:
				lexicon = pickle.load(f)
			with open('setAntrenamentAmestecat.csv', buffering=20000, encoding='latin-1') as f:
				batch_x = []
				batch_y = []
				batch_run = 0
				for rand in f:
					eticheta = rand.split('>>>')[0]
					tweet = rand.split('>>>')[1]
					cuvinte_actuale = word_tokenize(tweet.lower())
					cuvinte_actuale = [lemmatizer.lemmatize(i) for i in cuvinte_actuale]

					caracteristici = np.zeros(len(lexicon))

					for cuvant in cuvinte_actuale:
						if cuvant.lower() in lexicon:
							valoare_index = lexicon.index(cuvant.lower())
							caracteristici[valoare_index] += 1
					rand_x = list(caracteristici)
					#eval = VALUE.eval() = afiseaza rezultatul.
					#in tensorflow, daca afisam VALUE, vom avea un output care ne zice forma tensorului VALUE, tipul de data si o gramada de chestii legate de tensor, 
					#mai putin 
					#rezultatul. De aceea, cand vrem sa afisam tensorul, trebuie sa il evaluam.
					rand_y = eval(eticheta)
					batch_x.append(rand_x)
					batch_y.append(rand_y)
					if len(batch_x) >= batch_size:
						#feed_dict ii spune programului cui variabila sa ii atribuie ce.
						_, batch_loss = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x), y: np.array(batch_y)})
						epoch_loss += batch_loss
						batch_x = []
						batch_y = []
						batch_run += 1
						print('Batch:', batch_run, ' / ', total_batchuri, '| Epoch:', epoch, '| Eroare batch:', batch_loss)

			saver.save(sess, "./reteaNeuronala.ckpt")
			print('Epoch', epoch, ' / ', cate_epochuri, 'Eroare epoch:', epoch_loss)
			with open(log_file, 'a') as f:
				f.write(str(epoch) + '\n')
			epoch += 1

antreneaza_retea_neuronala(x)

#Definim functia de TESTARE
def testeaza_retea_neuronala():
	predictie = model_retea_neuronala(x)
	with tf.Session(config = config) as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(cate_epochuri):
			try:
				saver.restore(sess, "./reteaNeuronala.ckpt")
			except:
				pass
			epoch_loss = 0

		#Verificam daca argumentul maxim al predictiei este egal cu cel computat
		corect = tf.equal(tf.argmax(predictie, 1), tf.argmax(y, 1))

		#acuratete = media aritmetica dintre iteratiile in care programul a avut dreptate.
		acuratete = tf.reduce_mean(tf.cast(corect, 'float'))
		set_caracteristici = []
		set_etichete = []
		cnt = 0
		with open('setTestareProcesat.csv', buffering=20000) as f:
			for rand in f:
				try:
					caracteristici = list(eval(rand.split('>>')[0]))
					eticheta = list(eval(rand.split('>>')[1]))
					set_caracteristici.append(caracteristici)
					set_etichete.append(eticheta)
					cnt += 1
				except:
					pass
		print('----------------------------------------------------------------------------')
		print(cnt, 'mostre testate')
		print('----------------------------------------------------------------------------')
		test_x = np.array(set_caracteristici)
		test_y = np.array(set_etichete)
		acc = acuratete.eval({x: test_x, y: test_y}) * 100
		print('----------------------------------------------------------------------------')
		print('Acuratete: {0}%'.format(acc))
		print('----------------------------------------------------------------------------')

testeaza_retea_neuronala()