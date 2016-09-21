import optparse, os, sys, pdb
import scipy.io as sio

optparser = optparse.OptionParser()
optparser.add_option("--d", "--database", dest="database", default=None, help="Input dataset name")

(opts, _) = optparser.parse_args()

data_root = "/cs/vml2/xla193/cluster_video/datasets"

def produce_label(dataset):
	dataset_path = os.path.join( data_root, dataset )
	# pdb.set_trace()
	assert os.path.isdir( dataset_path ), "Input dataset does not exist"
	cnames = os.listdir( dataset_path )
	count = 1
	labels = dict()
	label_list = []
	for cname in cnames[:50]:
		vnames = os.listdir( os.path.join(dataset_path,cname) )
		lenv   = len(vnames)/2
		for i in range(lenv):
			tmp = [count]
			label_list.append(tmp)
		count += 1
	labels['label'] = label_list
	sio.savemat(os.path.join(data_root, dataset+"-label-50.mat"), labels)

if __name__ == '__main__':
	produce_label( opts.database )