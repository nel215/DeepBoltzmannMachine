#coding:utf-8
import numpy
import random
import math

class DeepBoltzmannMachine:
	def __init__(self,Layer,trainData):
		self.Layer = Layer
		self.trainData = trainData
				
	def pretrain(self,_Alpha=0.01):
		def calcHidList(visList,weight,scale=1.0):
			res = []
			sigmoid = lambda x:1.0/(1.0+math.exp(-x)) 
			for vis in visList:
				res.append(numpy.array([sigmoid(scale*numpy.dot(vis,w)) for w in numpy.transpose(weight)]))
			return res
			
		def calcVisList(hidList,weight,scale=1.0):
			res = []
			sigmoid = lambda x:1.0/(1.0+math.exp(-x)) 
			for hid in hidList:
				res.append(numpy.array([sigmoid(scale*numpy.dot(hid,w)) for w in weight]))
			return res
			
		L = len(self.Layer)
		self.weight = []
		for l in xrange(L-1):
			V = self.Layer[l]
			H = self.Layer[l+1]
			weight = numpy.array([numpy.array([random.uniform(-0.1,0.1) for h in xrange(H)]) for v in xrange(V)])
			print numpy.transpose(weight).shape
			print 'Layer',l,'-',l+1,'train.'
			# 入力データを生成
			visList = self.trainData
			for i in xrange(l):
				visList = calcHidList(visList,self.weight[i],2.0)
			# PersistentContrastiveDivergence用の可視層を生成
			visHatList = numpy.copy(visList)
			# 中間層は2倍する
			visCalcScale = 1.0 if l==0 else 2.0
			hidCalcScale = 1.0 if l==L-2 else 2.0
			
			N = len(visList)
			Alpha = _Alpha
			for iter in xrange(1000):
				# ContrastiveDivergence実行
				visHatList = calcVisList(calcHidList(visHatList,weight,hidCalcScale),weight,visCalcScale)
				#高速化のためのキャッシュ
				cache1 = calcHidList(visList,weight,hidCalcScale)
				cache2 = calcHidList(visHatList,weight,hidCalcScale)
				weightDelta = [numpy.array([0.0 for h in xrange(H)]) for v in xrange(V)]
				for k in xrange(N):
					for v in xrange(V):
						weightDelta[v] += visList[k][v]*cache1[k]
						weightDelta[v] -= visHatList[k][v]*cache2[k]
						#for h in xrange(H):
						#	weightDelta[v][h] += visList[k][v]*cache1[k][h]-visHatList[k][v]*cache2[k][h]
							
						
				wsum = 0
				# 勾配降下
				for v in xrange(V):
					weight[v] += weightDelta[v]/N*Alpha
					wsum += abs(weightDelta[v].sum())
				print wsum,Alpha
				Alpha *= 0.999
			self.weight.append(weight)
	def fixedpointeq(self,vis):
		sigmoid = lambda x:1.0/(1.0+math.exp(-x)) 
		L = len(self.Layer)
		# 固定点方程式をとく
		# 求められるものはp(h|v)の推定値
		# イテレーション回数
		T = 20
		dp = [[numpy.array([random.uniform(0,1) for i in xrange(l)]) for l in self.Layer] for t in xrange(T)]
		# 初期値の設定
		for t in xrange(T):dp[t][0] = numpy.copy(vis)
		for t in xrange(1,T):
			for l in xrange(1,L):
				sum = numpy.copy(self.bias[l])
				for i,w in enumerate(numpy.transpose(self.weight[l-1])):
					sum[i] += numpy.dot(dp[t-1][l-1],w)
				if l+1< L:
					for i,w in enumerate(self.weight[l]):
						sum[i] += numpy.dot(dp[t-1][l+1],w)
				dp[t][l] = numpy.array(map(sigmoid,sum))
		return dp[T-1]
					
	def sampling(self,prev):
		sigmoid = lambda x:1.0/(1.0+math.exp(-x)) 
		N = len(self.trainData)
		L = len(self.Layer)
		# PersistentContrastiveDivergenceでdp表は使いまわす
		# 求められるものはp(v,h)の推定値
		# イテレーション回数
		T = 10
		for t in xrange(T):
			next = [[numpy.array([0.0 for i in xrange(l)]) for l in self.Layer] for n in xrange(N)]
			for n in xrange(N):
				for l in xrange(L):
					sum = numpy.copy(self.bias[l])
					if l-1>=0:
						for i,w in enumerate(numpy.transpose(self.weight[l-1])):
							sum[i] += numpy.dot(prev[n][l-1],w)
					if l+1< L:
						for i,w in enumerate(self.weight[l]):
							sum[i] += numpy.dot(prev[n][l+1],w)
					next[n][l] = numpy.array(map(sigmoid,sum))
			prev,next = next,prev
		return prev
		
	def output(self):
		sigmoid = lambda x:1.0/(1.0+math.exp(-x)) 
		L = len(self.Layer)
		# 求められるものはp(v,h)の推定値
		# イテレーション回数
		T = 20
		dp = [[numpy.array([random.uniform(0,1) for i in xrange(l)]) for l in self.Layer] for t in xrange(T)]
		for t in xrange(1,T):
			for l in xrange(L):
				sum = numpy.copy(self.bias[l])
				if l-1>=0:
					for i,w in enumerate(numpy.transpose(self.weight[l-1])):
						sum[i] += numpy.dot(dp[t-1][l-1],w)
				if l+1< L:
					for i,w in enumerate(self.weight[l]):
						sum[i] += numpy.dot(dp[t-1][l+1],w)
				dp[t][l] = numpy.array(map(sigmoid,sum))
		return dp[T-1]
					
	def finetune(self,Alpha=0.005):
		print 'fine tune.'
		N = len(self.trainData)
		L = len(self.Layer)
		self.bias = [numpy.array([random.uniform(-0.1,0.1) for i in xrange(l)]) for l in self.Layer]
		
		sample = [[numpy.array([random.uniform(0,1) for i in xrange(l)]) for l in self.Layer] for n in xrange(N)]
		for i,vis in enumerate(self.trainData):
			sample[i][0] = numpy.copy(vis)
		for iter in xrange(200):
			weightDelta = [[numpy.array([0.0 for v in xrange(self.Layer[l+1])]) for h in xrange(self.Layer[l])] for l in xrange(L-1)]
			biasDelta = [numpy.array([0.0 for i in xrange(l)]) for l in self.Layer]
			# 固定点方程式
			for vis in self.trainData:
				mu = self.fixedpointeq(vis)	
				for l in xrange(L-1):
					for v in xrange(self.Layer[l]):
						for h in xrange(self.Layer[l+1]):
							weightDelta[l][v][h] += mu[l][v]*mu[l+1][h]/N
				for l in xrange(L):
					biasDelta[l] += mu[l]/N
							
			# サンプリング
			sample = self.sampling(sample)
			for s in sample:
				for l in xrange(L-1):
					for v in xrange(self.Layer[l]):
						for h in xrange(self.Layer[l+1]):
							weightDelta[l][v][h] -= s[l][v]*s[l+1][h]/N
				for l in xrange(L):
					biasDelta[l] -= s[l]/N
			
			wsum = 0
			bsum = 0
			# 更新
			for l in xrange(L-1):
				for v in xrange(self.Layer[l]):
					self.weight[l][v] += Alpha*weightDelta[l][v]
					wsum += abs(weightDelta[l][v].sum())
			for l in xrange(L):
				self.bias[l] += Alpha*biasDelta[l]
				bsum += abs(biasDelta[l].sum())
			
			print wsum,bsum
			Alpha *= 0.999
			
			
if __name__=='__main__':
	V = 5
	N = 20
	input = [numpy.array([1.0*random.randint(0,1) for j in xrange(V)]) for i in xrange(N)]
	
	dbm = DeepBoltzmannMachine([V,10,10],input)
	
	dbm.pretrain()
	
	dbm.finetune()
	
	sum = numpy.array([0.0 for j in xrange(V)])
	for inp in input:
		sum += inp
	print sum/N,dbm.output()[0]
	
	
	