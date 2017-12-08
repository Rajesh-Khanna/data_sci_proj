from sklearn.cluster import MiniBatchKMeans
import numpy as np
import data_modification as dm

trls = 168
duration = 300

dc = dm.ex_d('data_set_IVa_aa_cnt.txt','data_set_IVa_aa_mrk.txt',trls,duration)
dp= dc.MEAN()
mkm = MiniBatchKMeans(n_clusters=2)
mkm.fit(dp)
labels = mkm.labels_
targets = dc.labels
for i in range(trls):
	print(labels[i]-targets[i]+1,end='')