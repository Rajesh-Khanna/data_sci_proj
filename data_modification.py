import numpy as np
import pywt
import matplotlib.pyplot as plt
import scipy.stats as sp
from scipy.signal import butter, lfilter, freqz,filtfilt

e = [32,33,34,36,37,38,42,43,44,45,46,47,50,51,52,54,55,56,59,60,61,62,63,64,68,69,70,72,73,74]

class ex_d:
	def __init__(self,data_file,time_file,l,data_repetation,reduction):
		self.data_file = data_file
		self.time_file = time_file
		self.l = l  #168
		self.r = reduction #0
		self.dr = data_repetation  #300
		r = reduction
		d = open(self.data_file,'r')
		t = open(self.time_file,'r')
		dr = data_repetation
		dp = []
		r_data = []
		data = d.read().split('\n')
		ti = t.read().split('\n')
		time=[]
		labels = []
		for i in range(l):
			time.append(int(ti[i].split('\t')[0]))
			a = int(ti[i].split('\t')[1])
			if(a == 2):
				labels.append(0)
			else:
				labels.append(1)
		self.labels = labels
		for j in range(len(time)):
			dp.append([])
			#r_data.append([])
			for i in range(r//2	,dr-r//2):
				fg = data[time[j]-1+i].split('\t')
				dp[j].append([])
				for w in e:
					dp[j][-1].append(float(fg[w]))
		for i in dp:
			for j in range(len(i)):
				i[j] = np.array(i[j])
			i = np.array(i)
		dp = np.array(dp)
		self.d2 = dp
		ed = np.zeros((l,30,dr))
		for i in range(len(dp)):
			for j in range(dr):
				for k in range(30):
					ed[i,k,j] += dp[i,j,k]
		self.ed = ed

	def i_data(self,z):
		d = self.d2
		l = self.l
		dr = self.dr
		self.z = z
		q = 0
		dp = np.zeros((l*z,dr//z,30))
		for i in range(l):
			for w in range(z):
				for j in range(dr//z):
					dp[q][j] += d[i][j + w*dr//z]
				q+=1
		return(dp)

	def MEAN(self,dp2):
		dr = self.dr
		l = self.l
		r = self.r
		z = self.z
		data_point = np.zeros((l*z,30))
		for i in range(len(dp2)):
			for j in range(dr//z):
				
				#data_point[i] += (-0.5+j/(dr//z))*(dp2[i][j])
				
				data_point[i] += (dp2[i][j])
				
			data_point[i] = data_point[i]/(dr//z)

		return data_point

	def Labels(self):
		return self.labels
	def MEDIAN(self,dp2):
		dr = self.dr
		#dp2 = self.d2
		l = self.l
		z = self.z 
		data_point = np.zeros((l*z,30))
		for i in range(len(dp2)):		
			data_point[i] += dp2[i][dr//(2*z)]
		return data_point

	def VARIANCE(self):
		dr = self.dr
		dp2 = self.d2
		l = self.l
		r = self.r
		data_point = np.zeros((l,30))
		mean_points=self.MEAN(dp2)
		for i in range(len(dp2)):
			for j in range(dr-r):		
				da2ta_point[i] += abs(dp2[i][j]-mean_points[i])
			data_point[i] = data_point[i]/(dr-r)
		return data_point
	def coeff_var(self):
		dp2 = self.d2
		return (self.VARIANCE()/self.MEAN(dp2))

	def wavelet_features(self,z,dp):
		#d = self.d2
		dr = self.dr
		'''for i in range(len(d)):
			for j in d[i]:
				j = np.array(j)
			d[i]=np.array(d[i])
		'''
		#dp = np.array(d)
		
		r = self.r
		l = self.l		
		epoch = np.zeros((30,dr))
		for i in range(30):
			for j in range(dr):
				epoch[i][j] = 1*dp[z][j][i]
		
		#epoch += self.ed[z]

		channels=30	
		cA_values=[]
		cD_values=[]
		cA_mean=[]
		cA_std = []
		cA_Energy =[]
		cD_mean = []
		cD_std = []
		cD_Energy = []
		Entropy_D = []
		Entropy_A = []
		wfeatures = []
		for i in range(channels):
			cA,cD=pywt.dwt(epoch[i,:],'coif1')
			cA_values.append(cA)
			cD_values.append(cD)		#calculating the coefficients of wavelet transform.
		for x in range(channels):   
			cA_mean.append(np.mean(cA_values[x]))
			wfeatures.append(np.mean(cA_values[x]))
			cA_std.append(abs(np.std(cA_values[x])))
			wfeatures.append(abs(np.std(cA_values[x])))
			cA_Energy.append(abs(np.sum(np.square(cA_values[x]))))
			wfeatures.append(abs(np.sum(np.square(cA_values[x]))))
			cD_mean.append(np.mean(cD_values[x]))		# mean and standard deviation values of coefficents of each channel is stored .
			wfeatures.append(np.mean(cD_values[x]))

			cD_std.append(abs(np.std(cD_values[x])))	
			wfeatures.append(abs(np.std(cD_values[x])))
		
			cD_Energy.append(abs(np.sum(np.square(cD_values[x]))))
			wfeatures.append(abs(np.sum(np.square(cD_values[x]))))
		
			Entropy_D.append(abs(np.sum(np.square(cD_values[x]) * np.log(np.square(cD_values[x])))))
			wfeatures.append(abs(np.sum(np.square(cD_values[x]) * np.log(np.square(cD_values[x])))))
		
			Entropy_A.append(abs(np.sum(np.square(cA_values[x]) * np.log(np.square(cA_values[x]))))) 
			wfeatures.append(abs(np.sum(np.square(cA_values[x]) * np.log(np.square(cA_values[x])))))
		return wfeatures
	def wave_let(self,d):
		#d = self.d2
		dr = self.dr
		r = self.r
		l = self.l
		dp = np.array([self.wavelet_features(0,d)])
		for i in range(1,len(d)):
			dp = np.vstack([dp, self.wavelet_features(i,d)])
		return dp

	def skew(self,z):
		d = self.d2
		dr = self.dr
		r = self.r
		l = self.l
		for i in range(len(d)):
			for j in d[i]:
				j = np.array(j)
			d[i]=np.array(d[i])
		dp = np.array(d)
		data = np.zeros((30,dr))
		for i in range(30):
			for j in range(dr):
				data[i][j] = 1*dp[z][j][i]
		skew_array = np.zeros((len(data))) #Initialinling the array as all 0s
		index = 0; #current cell position in the output array
		for i in data:
			x=sp.stats.skew(i,axis=0,bias=True)
			skew_array[index]=x
			index+=1 #updating the cell position
		return np.sum(skew_array)/index

	def skew_let(self):
		d = self.d2
		dr = self.dr
		r = self.r
		l = self.l
		dp = np.array([self.skew(0)])
		for i in range(1,len(d)):
			dp = np.vstack([dp, self.skew(i)])
		return dp

	def butter_bandpass(self ,lowcut, highcut, fs, order=3):
		nyq = 0.5 * fs
		low = lowcut / nyq
		high = highcut / nyq
		b, a = butter(order, [low, high], btype='band')
		return b, a


	def butter_bandpass_filter(self,z, lowcut, highcut, fs, order=3):
		data = self.ed[z]
		b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
		y = np.array([filtfilt(b, a, data[0])])
		for i in range(1,len(data)):	
			b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
			y=np.vstack([y,filtfilt(b, a, data[i])])
		#print(len(y[0]))
		return y

	def BF(self):

		fs = 300/3
		lowcut = 7.0
		highcut = 30.0
		#t = range(len(eeg))
		l = self.l
		y = np.array(self.butter_bandpass_filter(0, lowcut, highcut, fs, order=3))
		for i in range(1,l):
			y=np.vstack([y,self.butter_bandpass_filter(i, lowcut, highcut, fs, order=3)])
		dp = np.zeros((168,300,30))
		for i in range(168):
			for j in range(300):
				for k in range(30):
					dp[i][j][k] = 1*y[30*(i//30) + k][j]
		return dp