import sys, os, pdb

file_root = '/cs/vml4/xla193/cross1_list'
file_head = 'list_frm-veri-fix-train-shuffle-0-'
all_file  = 'list_frm-labelvect-fix-veri-0ft-random.txt'
SUM = 0
with open(os.path.join(file_root, all_file), 'r') as inf:
	lines = inf.readlines()
	SUM = len(lines)

for i in range(10):
	print i+1,'starting:', SUM
	filename = file_head + str(i+1) + '.txt'
	with open(os.path.join(file_root, filename), 'r') as inf:
		lines = inf.readlines()
		print (len(lines))
		print int(round(float(len(lines))/50))
		print (SUM - len(lines))
		print int(round(float(SUM-len(lines))/50))


