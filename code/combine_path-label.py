import optparse, os, sys, pdb
import scipy.io as sio
import pdb

optparser = optparse.OptionParser()
optparser.add_option("--pn", "--num", dest="num", default='20', help="Input number of clusters")
optparser.add_option("--li", "--label", dest="label", default='UCF-101-label-20-0ft.mat', help="Input label file name")
optparser.add_option("--o",  "--output",dest="output",default=None, help="Input output file name")
optparser.add_option("--d", "--database",dest="database",default=None, help="Input dataset name")

(opts, _) = optparser.parse_args()
data_root = '/cs/vml2/xla193/cluster_video'
output_path= os.path.join( data_root, 'output' )

def combine_pl(pnum, labelin, pathout, database):
	label_file = os.path.join( output_path, database, labelin )
	output_file= os.path.join( output_path, database, pathout)
	database_path = os.path.join( data_root, 'datasets', database )
	assert os.path.isdir(database_path), "Database file dose not exist"
	cnames = os.listdir(database_path)
	cnames_required = cnames[:int(pnum)]
	with open(output_file, 'w') as inputf:
		label_dic = sio.loadmat( label_file )
		labels    = label_dic['labels']
		vcount = 0
		for inputname in cnames_required:
			video_path = os.path.join( data_root, 'datasets', database, inputname )
			vnames = os.listdir(video_path)
			vlen   = len(vnames)/2
			vnames_required = vnames[vlen:]
			for vname in vnames_required:
				# pdb.set_trace()
				print "write inputfile from: ", vname
				fms_path = os.path.join( video_path, vname )
				fnames = os.listdir(fms_path)
				fnums  = len(fnames)
				count  = 1
				while count+16 <= fnums:
					content = fms_path+' '+str(count)+' '+str(labels[vcount][0])+'\n'
					# content2= os.path.join(outdata_path, vname, fnames[count-1][:-4])
					# pdb.set_trace()
					print "write to inputfile without label:", content 
					# print "write to outputfile:", content2
					inputf.write(content)
					# outputf.write(content2+'\n')
					count += 16
				vcount += 1


if __name__ == '__main__':
	combine_pl( opts.num, opts.label, opts.output, opts.database )