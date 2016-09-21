import sys, os, optparse, array
import numpy as np 
import scipy.io as sio
import struct
import pdb
from sklearn.preprocessing import normalize
from random import shuffle

optparser = optparse.OptionParser()
optparser.add_option("--d", "--database", dest="database", default=None, help="Input database name")
optparser.add_option("--i", "--input", dest="input", default=None, help="Input file name")
optparser.add_option("--o", "--output", dest="output", default=None, help="Output file name")

(opts, _) = optparser.parse_args()

data_root     = '/cs/vml2/xla193/cluster_video/output'

def shuffle_lines(database, inputfile, outputfile):
	file_path = os.path.join( data_root, database )
	assert os.path.isdir(file_path), "File holder dose not exist"
	ori_file = os.path.join( file_path, inputfile )
	tar_file = os.path.join( file_path, outputfile )
	with open(ori_file, 'r') as inf:
		in_lines = inf.readlines()

	shuffle(in_lines)

	with open(tar_file, 'w') as ouf:
		for line in in_lines:
			print "writing", line
			ouf.write(line)

if __name__ == '__main__':
    shuffle_lines( opts.database, opts.input, opts.output )