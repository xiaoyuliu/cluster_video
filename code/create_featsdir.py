import optparse, os, sys, pdb
import scipy.io as sio

optparser = optparse.OptionParser()
optparser.add_option("--fn", "--ftnum", dest="ftnum", default=None, help="Input the number of fine-tuning")
optparser.add_option("--d", "--database",dest="database",default=None, help="Input dataset name")

(opts, _) = optparser.parse_args()
data_root = '/cs/vml2/xla193/cluster_video/datasets'

def create_fdir(ftnum, dataset):
	target_dir = os.path.join(data_root,dataset+'-feats')
	dirnames   = os.listdir(target_dir)
	dnames     = dirnames[:2694]
	# pdb.set_trace()
	new_dir	   = os.path.join(target_dir+str(ftnum))
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)
	for dname in dnames:
		vpath  = os.path.join( new_dir, dname )
		if not os.path.exists(vpath):
			os.makedirs(vpath)
			print "create file holder:", vpath

if __name__ == '__main__':
	create_fdir( opts.ftnum, opts.database ) 