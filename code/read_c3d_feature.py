import sys, os, optparse, array
import numpy as np 
import scipy.io as sio
import struct
import pdb
from sklearn.preprocessing import normalize

optparser = optparse.OptionParser()
optparser.add_option("--d", "--database", dest="database", default=None, help="Input dataset name")
# optparser.add_option("--i", "--input", dest="input", default=None, help="Input feature file name")# now enter the name of the video
# optparser.add_option("--o", "--output",dest="output",default="out_video.mat", help="Input output file name")
# optparser.add_option("--p", "--precision", dest="precision", default="f", help="Input required precision")
# optparser.add_option("--d", "--database",  dest="database",  default=None, help="Input vedio name")

(opts, _) = optparser.parse_args()

data_root     = '/cs/vml2/xla193/cluster_video/datasets'
# model_root    = os.path.join( data_root, 'examples/c3d_feature_extraction/output/c3d' )

def read_binary_blob(dname):
	dataset_path = os.path.join(data_root,dname+'-feats1')
	assert os.path.isdir(dataset_path),    "Input feature fileholder dose not exists."
	# fout_path_total = os.path.join(data_root, dname+'.mat')
	vnames = os.listdir(dataset_path)
	out_list = []
	for vname in vnames:
		out_list_onevideo = []
		frame_path = os.path.join( dataset_path,vname )
		fnames = os.listdir(frame_path)
		# pdb.set_trace()
		for full_name in fnames:
			(name, format) = full_name.strip().split('.')
			if(format != 'fc6-1'):
				continue
			fout_path = os.path.join(frame_path, name+'.mat')
			read_status = 1
			out_mat = dict()
			feats = dict()
			with open(os.path.join( frame_path, full_name ), 'rb') as fin:
				# ss = struct.unpack("i", fin.read(4))
				s = array.array("i") # int32
				s.fromfile(fin, 5)
				if (len(s) == 5):
					m = reduce(lambda x,y: x*y, s)
					# data = struct.unpack(precision, fin.read(m))
					data_aux = array.array("f")
					data_aux.fromfile(fin, m)
					# pdb.set_trace()
					data = np.array(data_aux.tolist())
					if (len(data) != m):
						read_status = 0
				else:
					read_status = 0

			if read_status:
				blob = np.zeros((s[0], s[1], s[2], s[3], s[4]), np.float32)
				off = 0
				image_size = s[3]*s[4]
				for n in range(0, s[0]):
					for c in range(0, s[1]):
						for l in range(0, s[2]):
							tmp = data[off:off+image_size]
							blob[n][c][l][:][:] = np.reshape( tmp, (s[4], s[3]), order='F' ).T 
							off = off + image_size
				
				out_mat['s'] = s
				out_mat['f'+name] = blob
				out_mat['read_status'] = read_status
				print( 'Write feature to file: {0}.\n'.format( fout_path ))
				sio.savemat( fout_path, out_mat )
			else:
				s = []
				blob = []
				out_mat['s'] = s
				out_mat['f'+name] = blob
				out_mat['read_status'] = read_status
				print( 'Write feature to file: {0}.\n'.format( fout_path ))
				sio.savemat( fout_path, out_mat )

			out_list_onevideo.append(np.reshape(blob, ( blob.shape[0], blob.shape[1]), order='F'))
		out_arrays = np.array(out_list_onevideo)
		average = np.mean(out_arrays, axis=0)
		feature = normalize(average)
		feats['data'] = feature
		print( 'Write final feature to file: {0}.\n'.format( os.path.join(frame_path, 'out_video.mat') ))
		sio.savemat( os.path.join(frame_path, 'out_video.mat'), feats)
		# pdb.set_trace()
		feature2list = feature[0].tolist()
		out_list.append(feature2list)
		# pdb.set_trace()


	# reshaped_out_list = np.reshape( out_list,  )
	out_mats = dict()
	out_mats['data'] = out_list
	sio.savemat( os.path.join( data_root, dname+'_out_feats20_ft1.mat'), out_mats )

if __name__ == '__main__':
    read_binary_blob( opts.database )