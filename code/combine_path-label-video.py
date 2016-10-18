import optparse, os, sys, pdb
import scipy.io as sio
import pdb

optparser = optparse.OptionParser()
optparser.add_option("--li", "--label", dest="label", default='UCF-101-gtlabel-10.mat', help="Input label file name")
optparser.add_option("--pi", "--path", dest="path", default='list_video-20with0ftlabel.txt', help="Input path file name")
optparser.add_option("-o",  "--output",dest="output",default='list_video-10withgtlabel.txt', help="Input output file name")
optparser.add_option("-d", "--database",dest="database",default='UCF-101', help="Input dataset name")

(opts, _) = optparser.parse_args()
data_root = '/cs/vml2/xla193/cluster_video'
output_path= os.path.join( data_root, 'output' )

def combine_pl( labelin, pathin, pathout, database):
	label_file = os.path.join( output_path, database, labelin )
	path_file  = os.path.join( output_path, database, pathin )
	output_file= os.path.join( output_path, database, pathout)
	
	with open(path_file, 'r') as pathf:
		paths = pathf.readlines()[:1374]
	# pdb.set_trace()

	with open(output_file, 'w') as outf:
		label_dic = sio.loadmat( label_file )
		labels    = label_dic['label']
	
		for idx, path in enumerate(paths):
			(image_path, label_indicator) = path.strip().split(' ')
			content = image_path + ' ' + str(int(labels[idx])) + '\n'
			outf.write(content)
			print idx, "writing: ", content


if __name__ == '__main__':
	combine_pl( opts.label, opts.path, opts.output, opts.database )
