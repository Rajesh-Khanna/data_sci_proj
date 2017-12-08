import numpy as np

class ex_d:
	def __init__(self,data_file,time_file,l,data_repetation):
		self.data_file = data_file
		self.time_file = time_file
		self.l = l
		self.dr = data_repetation
		d = open(self.data_file,'r')
		t = open(self.time_file,'r')
		dr = data_repetation
		dp = []
		data = d.read().split('\n')
		ti = t.read().split('\n')
		time=[]
		labels = []
		for i in range(l):
			time.append(int(ti[i].split('\t')[0]))
			labels.append(int(ti[i].split('\t')[1]))
		self.labels = labels
		for j in range(len(time)):
			dp.append([])
			for i in range(50,dr-100):
				dp[j].append(data[time[j]-1+i].split('\t'))
		dp2 = []
		self.d2 = dp2
		for i in range(len(dp)):
			dp2.append([])
			for j in range(50,dr-100):
				dp2[i].append(list(map(float, dp[i][j])))	
		for i in dp2:
			for j in range(50,dr-100):
				i[j] = np.array(i[j])
		
	def MEAN(self):
		dr = self.dr
		dp2 = self.d2
		l = self.l
		data_point = np.zeros((l,118))
		for i in range(len(dp2)):
			for j in range(50,dr-100):		
				data_point[i] += dp2[i][j]
			data_point[i] = data_point[i]/100
		return data_point
	def Labels(self):
		return self.labels