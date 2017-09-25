import scipy.io as sio
import h5py
import numpy as np
from scipy.stats.stats import pearsonr
import sys
from resnet50 import *



"""
#Datascience FMRI data
P1_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Michell_Data/NewNumpy/P1_MRI.npy"
P2_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Michell_Data/NewNumpy/P2_MRI.npy"
P3_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Michell_Data/NewNumpy/P3_MRI.npy"
P4_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Michell_Data/NewNumpy/P4_MRI.npy"
P5_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Michell_Data/NewNumpy/P5_MRI.npy"
P6_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Michell_Data/NewNumpy/P6_MRI.npy"
P7_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Michell_Data/NewNumpy/P7_MRI.npy"
P8_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Michell_Data/NewNumpy/P8_MRI.npy"
P9_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Michell_Data/NewNumpy/P9_MRI.npy"



#MEG Data
A_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Meg_data/NewNumpy/A_MEG.npy"
B_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Meg_data/NewNumpy/B_MEG.npy"
C_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Meg_data/NewNumpy/C_MEG.npy"
D_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Meg_data/NewNumpy/D_MEG.npy"
E_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Meg_data/NewNumpy/E_MEG.npy"
F_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Meg_data/NewNumpy/F_MEG.npy"
G_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Meg_data/NewNumpy/G_MEG.npy"
I_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Meg_data/NewNumpy/I_MEG.npy"
J_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Meg_data/NewNumpy/J_MEG.npy"
"""



#ORG MEG datascience correlation matrices
A_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/MEG_org/a.npy"
B_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/MEG_org/b.npy"
C_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/MEG_org/c.npy"
D_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/MEG_org/d.npy"
E_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/MEG_org/e.npy"
F_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/MEG_org/f.npy"
G_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/MEG_org/g.npy"
I_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/MEG_org/i.npy"
J_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/MEG_org/j.npy"

#ORG FMRI datascience correlation matrices
P1_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/fMRI_org/b1.npy"
P2_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/fMRI_org/b2.npy"
P3_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/fMRI_org/b3.npy"
P4_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/fMRI_org/b4.npy"
P5_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/fMRI_org/b5.npy"
P6_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/fMRI_org/b6.npy"
P7_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/fMRI_org/b7.npy"
P8_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/fMRI_org/b8.npy"
P9_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/fMRI_org/b9.npy"
"""
#Abstract only Anderson_Data
P1_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Anderson_Data/NewNumpy/P1_MRIA.npy"
P2_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Anderson_Data/NewNumpy/P2_MRIA.npy"
P3_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Anderson_Data/NewNumpy/P3_MRIA.npy"
P4_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Anderson_Data/NewNumpy/P4_MRIA.npy"
P5_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Anderson_Data/NewNumpy/P5_MRIA.npy"
P6_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Anderson_Data/NewNumpy/P6_MRIA.npy"
P7_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Anderson_Data/NewNumpy/P7_MRIA.npy"
P8_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Anderson_Data/NewNumpy/P8_MRIA.npy"
P9_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Anderson_Data/NewNumpy/P9_MRIA.npy"



#EEG DATA
A_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/EEG_data/NewNumpy/A_EEG.npy"
B_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/EEG_data/NewNumpy/B_EEG.npy"
C_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/EEG_data/NewNumpy/C_EEG.npy"
D_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/EEG_data/NewNumpy/D_EEG.npy"
E_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/EEG_data/NewNumpy/E_EEG.npy"
F_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/EEG_data/NewNumpy/F_EEG.npy"
G_MAT = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/EEG_data/NewNumpy/G_EEG.npy"
"""



DICTIONARY ="/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Michell_Data/Dictionary/dictionary_org.txt"
#DICTIONARY = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Anderson_Data/Dictionary/Anderson_words.txt"
#DICTIONARY = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Anderson_Data/Dictionary/Anderson_italian.txt"
#DICTIONARY = "/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/Anderson_Data/Dictionary/A.txt"
#DICTIONARY = '/Users/Dhanush/Desktop/Projects/Brain_Bench/GIT_DATA/EEG_Data/Dictionary/EEG_dictionary.txt'


IMG_PATH ='/Users/Dhanush/Desktop/Projects/CNNs/deep-learning-models/DataScienceImages/'
layer_name ='conv1'
#layer_name = 'flatten_1'
#layer_name = 'fc1000'

size_words = 60


def two_vs_two (input_mat, brain_mat, length):
	brain_1 = input_mat
	brain_2 = brain_mat
	#brain_2 = np.load(open(brain_2_name, 'r'))
	s = 0
	total = 0

	for line_a in range (0,length):
		b_1_a = brain_1[line_a,:]
		b_2_a = brain_2[line_a,:]
		for line_b in range (line_a+1, length):
			b_1_b = brain_1[line_b,:]
			b_2_b = brain_2[line_b,:]
			mask = np.ones(len(b_1_a), dtype=bool)
			mask[[line_a, line_b]] = False
			#b_1_a_masked = ma.masked_array(b_1_a, mask = mask)
			#b_2_a_masked = ma.masked_array(b_2_a, mask = mask)
			#b_1_b_masked = ma.masked_array(b_1_b, mask = mask)
			#b_2_b_masked = ma.masked_array(b_2_b, mask = mask)
			b_1_a_masked = b_1_a[mask]
			b_2_a_masked = b_2_a[mask]
			b_1_b_masked = b_1_b[mask]
			b_2_b_masked = b_2_b[mask]
			#print mask
			#print len(b_2_b_masked)
			#print mask

			part_a = pearsonr(b_1_a_masked, b_2_a_masked)[0] + pearsonr(b_1_b_masked, b_2_b_masked)[0]
			part_b = pearsonr(b_1_a_masked, b_2_b_masked)[0] + pearsonr(b_1_b_masked, b_2_a_masked)[0]
			# part_a = distance.cosine(b_1_a_masked, b_2_a_masked) + distance.cosine(b_1_b_masked, b_2_b_masked)
			# part_b = distance.cosine(b_1_a_masked, b_2_b_masked) + distance.cosine(b_1_b_masked, b_2_a_masked)
			
			#print part_a
			#print part_b

			total += 1
			if part_a > part_b:
				s += 1		
	#print s/float(total)	
	# print "%d out of %d" % (s, total)
	return s/float(total)


def get_score(input_mat, brain_data_filename, mask, length):
	brain_file = open(brain_data_filename, 'r')
	brain_mat = np.load(brain_file)
	brain_mat = np.reshape(brain_mat[mask], (length, length))
	return two_vs_two(input_mat, brain_mat, length)


def get_fMRI_average(input_mat, mask, length):
	fMRI_score = get_score(input_mat,  P1_MAT, mask, length)
	fMRI_score += get_score(input_mat, P2_MAT, mask, length)
	fMRI_score += get_score(input_mat, P3_MAT, mask, length)
	fMRI_score += get_score(input_mat, P4_MAT, mask, length)
	fMRI_score += get_score(input_mat, P5_MAT, mask, length)
	fMRI_score += get_score(input_mat, P6_MAT, mask, length)
	fMRI_score += get_score(input_mat, P7_MAT, mask, length)
	fMRI_score += get_score(input_mat, P8_MAT, mask, length)
	fMRI_score += get_score(input_mat, P9_MAT, mask, length)
	return fMRI_score/9.0
	#return fMRI_score/7.0

def get_MEG_average(input_mat, mask, length):
	MEG_score = get_score(input_mat, A_MAT, mask, length)
	MEG_score += get_score(input_mat, B_MAT, mask, length)
	MEG_score += get_score(input_mat, C_MAT, mask, length)
	MEG_score += get_score(input_mat, D_MAT, mask, length)
	MEG_score += get_score(input_mat, E_MAT, mask, length)
	MEG_score += get_score(input_mat, F_MAT, mask, length)
	MEG_score += get_score(input_mat, G_MAT, mask, length)
	MEG_score += get_score(input_mat, I_MAT, mask, length)
	MEG_score += get_score(input_mat, J_MAT, mask, length)
	return MEG_score/9.0

def get_EEG_average(input_mat, mask, length):
	EEG_score = get_score(input_mat, A_MAT, mask, length)
	EEG_score += get_score(input_mat, B_MAT, mask, length)
	EEG_score += get_score(input_mat, C_MAT, mask, length)
	EEG_score += get_score(input_mat, D_MAT, mask, length)
	EEG_score += get_score(input_mat, E_MAT, mask, length)
	EEG_score += get_score(input_mat, F_MAT, mask, length)
	EEG_score += get_score(input_mat, G_MAT, mask, length)
	return EEG_score/7.0

def run_test ():
	obj = get_CNN_vector(IMG_PATH,DICTIONARY,layer_name,size_words)
	input_mat = obj['input_mat']
	mask = obj['mask']
	length = obj['length']
	fMRI_score = 0
	fMRI_score = get_fMRI_average(input_mat, mask, length)
	MEG_score = 0
	MEG_score = get_MEG_average(input_mat, mask, length)
	EEG_score = 0
	#EEG_score = get_EEG_average(input_mat, mask, length)
	return [fMRI_score, MEG_score, length,EEG_score]

def main():
		scores = run_test()
		print "There are %d words found in BrainBench from the input" % scores[2]
		print "The fMRI score is %f" % scores[0]
		print "The MEG score is %f" % scores[1]
		#print "The EEG score is %f" % scores[3]

if __name__ == "__main__":
    main()