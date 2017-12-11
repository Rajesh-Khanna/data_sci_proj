from sklearn.cluster import MiniBatchKMeans,KMeans,SpectralClustering,MeanShift,AffinityPropagation,AgglomerativeClustering,DBSCAN,Birch,FeatureAgglomeration
from sklearn.mixture import GaussianMixture
import numpy as np
import data_modification as dm
import pywt
import matplotlib.pyplot as plt
trls = 168
duration = 300
reduction = 0
sub_sets = 1
dc = dm.ex_d('data_set_IVa_aa_cnt.txt','data_set_IVa_aa_mrk.txt',trls,duration,reduction)
dpz =[]
#rd = dc.red_data()
a = dc.BF()
dc2 = dc.i_data(1)
dpz.append(dc.MEAN(a))
dpz.append(dc.MEDIAN(a))
#dpz.append(dc.VARIANCE())
#dpz.append(dc.coeff_var())
dpz.append(dc.wave_let(a))
dpz.append(dc.skew_let())
targets = dc.Labels()

max = 0

for dp in dpz:

	print('\n\n\n****************     *******************\n\n\n')
	mkm = MiniBatchKMeans(n_clusters=2,max_iter = 5000,n_init = 50)
	mkm.fit(dp)
	labels = mkm.labels_
	
	count = 0
	for i in range(trls):
		if(labels[9]==0):
			if(labels[i] == 1-targets[i]):
				count+=1
		else:
			if(labels[i] == targets[i]):
				count+=1


	print('MiniBatchKMeans : ',end='')
	acuracy = count*100/trls
	if(max<acuracy):
		max = acuracy
	print(acuracy,end='')
	print('%')
	print('')

	mkm = KMeans(n_clusters=2,max_iter = 5000,n_init = 50,algorithm = 'elkan')
	mkm.fit(dp)
	bels = mkm.labels_
	

	count = 0
	for i in range(trls):
		if(labels[9]==0):
			if(labels[i] == 1-targets[i]):
				count+=1
		else:
			if(labels[i] == targets[i]):
				count+=1

	print('KMeans : ',end='')
	acuracy = count*100/trls
	if(max<acuracy):
		max = acuracy
	print(acuracy,end='')
	print('%\n')
	
	mkm = SpectralClustering(n_clusters=2)
	mkm.fit(dp)
	labels = mkm.labels_
	
	
	count = 0
	for i in range(trls):
		if(labels[9]==0):
			if(labels[i] == 1-targets[i]):
				count+=1
		else:
			if(labels[i] == targets[i]):
				count+=1

	print('SpectralClustering : ',end='')
	acuracy = count*100/trls
	if(max<acuracy):
		max = acuracy
	print(acuracy,end='')
	print('%\n')
	
	mkm = MeanShift()
	mkm.fit(dp)
	labels = mkm.labels_
	
	
	count = 0
	for i in range(trls):
		if(labels[9]==0):
			if(labels[i] == 1-targets[i]):
				count+=1
		else:
			if(labels[i] == targets[i]):
				count+=1
	print('MeanShift : ',end='')
	acuracy = count*100/trls
	if(max<acuracy):
		max = acuracy
	print(acuracy,end='')
	print('%\n')

	mkm = AffinityPropagation()
	mkm.fit(dp)
	labels = mkm.labels_
	
	
	count = 0
	for i in range(trls):
		if(labels[9]==0):
			if(labels[i] == 1-targets[i]):
				count+=1
		else:
			if(labels[i] == targets[i]):
				count+=1
	print('AffinityPropagation : ',end='')
	acuracy = count*100/trls
	if(max<acuracy):
		max = acuracy
	print(acuracy,end='')
	print('%\n')
	

	mkm = AgglomerativeClustering(n_clusters =2)
	mkm.fit(dp)
	labels = mkm.labels_
	
	count = 0
	for i in range(trls):
		if(labels[9]==0):
			if(labels[i] == 1-targets[i]):
				count+=1
		else:
			if(labels[i] == targets[i]):
				count+=1
	print('AgglomerativeClustering : ',end='')
	acuracy = count*100/trls
	if(max<acuracy):
		max = acuracy
	print(acuracy,end='')
	print('%\n')

	mkm = DBSCAN()
	mkm.fit(dp)
	labels = mkm.labels_
	
	count = 0
	for i in range(trls):
		if(labels[9]==0):
			if(labels[i] == 1-targets[i]):
				count+=1
		else:
			if(labels[i] == targets[i]):
				count+=1
	print('DBSCAN : ',end='')
	acuracy = count*100/trls
	if(max<acuracy):
		max = acuracy
	print(acuracy,end='')
	print('%\n')

	mkm = Birch()
	mkm.fit(dp)
	labels = mkm.labels_
	
	count = 0
	for i in range(trls):
		if(labels[9]==0):
			if(labels[i] == 1-targets[i]):
				count+=1
		else:
			if(labels[i] == targets[i]):
				count+=1
	print('Birch : ',end='')
	acuracy = count*100/trls
	if(max<acuracy):
		max = acuracy
	print(acuracy,end='')
	print('%\n')



	mkm = GaussianMixture()
	mkm.fit(dp)
	labels = mkm.predict(dp)
	
	count = 0
	for i in range(trls):
		if(labels[9]==0):
			if(labels[i] == 1-targets[i]):
				count+=1
		else:
			if(labels[i] == targets[i]):
				count+=1
	print('GaussianMixture : ',end='')
	acuracy = count*100/trls
	if(max<acuracy):
		max = acuracy
	print(acuracy,end='')
	print('%\n')


print('best accuracy achieved : ',end='')
print(max)
