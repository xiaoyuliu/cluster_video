import sys, os, optparse, array
import numpy as np 
import scipy.io as sio
import struct
import pdb
from sklearn.preprocessing import normalize

optparser = optparse.OptionParser()
optparser.add_option("--d", "--database", dest="database", default=None, help="Input database name")
# optparser.add_option("--i", "--input", dest="input", default=None, help="Input video cluster name")

(opts, _) = optparser.parse_args()

data_root     = '/cs/vml2/xla193/cluster_video/datasets'

def create_iofile(database):
	database_path = os.path.join( data_root, database )
	outdata_path  = os.path.join( data_root, database+'-feats' )
	assert os.path.isdir(database_path), "Database file dose not exist"
	cnames = os.listdir(database_path)
	cnames_required = cnames[:50]
	with open(os.path.join(data_root, 'input_ucf101_list_frm-nolabel.txt'), 'w') as inputf:
	# with open(os.path.join(data_root, 'input_ucf101_list_frm-nolabel.txt'), 'w') as inputf, \
		 # open(os.path.join(data_root, 'output_ucf101_list_prefix.txt'), 'w') as outputf:
		for inputname in cnames_required:
			video_path = os.path.join( data_root, database, inputname )
			vnames = os.listdir(video_path)
			vnames_required = vnames[len(vnames)/2:]
			for vname in vnames_required:
				# pdb.set_trace()
				if not os.path.exists(os.path.join(outdata_path, vname)):
					os.mkdir(os.path.join(outdata_path, vname))
				print "create inputfile from: ", vname
				fms_path = os.path.join( video_path, vname )
				fnames = os.listdir(fms_path)
				fnums  = len(fnames)
				count  = 1
				while count+16 <= fnums:
					content = fms_path+' '+str(count)+'\n'
					# content2= os.path.join(outdata_path, vname, fnames[count-1][:-4])

					print "write to inputfile without label:", content 
					# print "write to outputfile:", content2
					inputf.write(content)
					# outputf.write(content2+'\n')
					count += 16

if __name__ == '__main__':
    create_iofile( opts.database )