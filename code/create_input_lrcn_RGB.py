import sys, os, optparse, array
import numpy as np 
import scipy.io as sio
import struct
import pdb
from sklearn.preprocessing import normalize

optparser = optparse.OptionParser()
optparser.add_option("--d", "--database", dest="database", default=None, help="Input database name")
optparser.add_option("--i", "--input", dest="input", default='input_UCF-101_list_frm-20_0ft.txt', help="Input file name")
optparser.add_option("--o", "--output", dest="output", default=None, help="Output file name")

(opts, _) = optparser.parse_args()

data_root     = '/cs/vml2/xla193/cluster_video/output'

def create_iofile(database, inputfile, outputfile):
	file_path = os.path.join( data_root, database )
	assert os.path.isdir(file_path), "File holder dose not exist"
	ori_file = os.path.join( file_path, inputfile )
	tar_file = os.path.join( file_path, outputfile )
	with open(ori_file, 'r') as inf:
		in_lines = inf.readlines()

	with open(tar_file, 'w') as ouf:
		temp_holder = ' '
		for line in in_lines:
			(holder, idx, label) = line.strip().split(' ')
			if holder == temp_holder:
				continue
			temp_holder = holder
			
			name_fms = os.listdir( holder )
			total = len( name_fms )
			# pdb.set_trace()
			# if total >= 100:
			# 	count = 100
			# else:
			# 	count = total
			for name in name_fms:
				w_content = os.path.join( holder,name ) + ' ' + label
				print "write", w_content, "to ", tar_file
				ouf.write(w_content+'\n')

if __name__ == '__main__':
    create_iofile( opts.database, opts.input, opts.output )