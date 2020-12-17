import os
import numpy as np
import pandas as pd



if __name__ == '__main__':

	cols = [ 
			'AU01_r','AU02_r','AU04_r','AU05_r','AU06_r','AU07_r','AU09_r','AU10_r',
			'AU12_r','AU14_r','AU15_r','AU17_r','AU20_r','AU23_r','AU25_r','AU26_r','AU45_r' 
		]

	train_files = os.listdir('../data/train')
	val_files   = os.listdir('../data/val')
	test_files  = os.listdir('../data/test')

	for ftrain in train_files:
		player    = ftrain.split('_')[1]
		keep_cols = cols[:]
		for i in range(len(cols)):
		    keep_cols[i] = player+'_'+cols[i]
		df = pd.read_csv(os.path.join('../data/train', ftrain), usecols=keep_cols)
		np.savez(os.path.join('../data/train', ftrain), df.to_numpy())


	for fval in val_files:
		player    = fval.split('_')[1]
		keep_cols = cols[:]
		for i in range(len(cols)):
		    keep_cols[i] = player+'_'+cols[i]
		df = pd.read_csv(os.path.join('../data/val', fval), usecols=keep_cols)
		np.savez(os.path.join('../data/val', fval), df.to_numpy())
		# name = fval.split('.')[0]
		# os.rename(os.path.join('../data/val', fval), os.path.join('../data/val', name+'.npz'))


	for ftest in test_files:
		player    = ftest.split('_')[1]
		keep_cols = cols[:]
		for i in range(len(cols)):
		    keep_cols[i] = player+'_'+cols[i]
		df = pd.read_csv(os.path.join('../data/test', ftest), usecols=keep_cols)
		np.savez(os.path.join('../data/test', ftest), df.to_numpy())
		# name = ftest.split('.')[0]
		# os.rename(os.path.join('../data/test', ftest), os.path.join('../data/test', name+'.npz'))
