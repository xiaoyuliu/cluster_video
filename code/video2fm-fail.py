import cv2, optparse, os, sys, pdb

optparser = optparse.OptionParser()
optparser.add_option("--d", "--database", dest="database", default=None, help="Input database name")
optparser.add_option("--i", "--input", dest="input", default=None, help="Input feature file name")

(opts, _) = optparser.parse_args()

data_root = '/cs/vml2/xla193/cluster_video/datasets'

def video_to_frame(database, inputfile):
	input_path = os.path.join( data_root, database, inputfile)
	assert os.path.isdir(input_path), "Input videos do not exist"
	out_base   = os.path.join( data_root, database+'-fms', inputfile)
	vnames = os.listdir(input_path)
	for idx, vname in enumerate(vnames):
		if (idx < len(vnames)/2):
			fn = os.path.join(input_path, vname)
			(name, format) = vname.strip().split('.')
			vidcap = cv2.VideoCapture(fn)
			success,image = vidcap.read()
			count = 0
			success = True
			pdb.set_trace()
			while success:
			  success,image = vidcap.read()
			  print 'Read a new frame: ', success
			  out_name = "%06d.jpg" % count
			  cv2.imwrite(os.path.join(out_base, out_name), image)    # save frame as JPEG file
			  count += 1


if __name__ == '__main__':
    video_to_frame( opts.database, opts.input )
