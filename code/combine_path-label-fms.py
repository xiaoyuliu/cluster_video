import optparse, os, sys, pdb
import scipy.io as sio
import pdb

optparser = optparse.OptionParser()
optparser.add_option("--li", "--label", dest="label", default='outputlabel-UCF-101-10-0ft.mat', help="Input label file name")
optparser.add_option("--pi", "--path", dest="path", default='list_frm-10with0label.txt', help="Input path file name")
optparser.add_option("--ci", "--count", dest="count", default='input-UCF-101-20-fmcount.txt', help="Input count file name")
optparser.add_option("-o",  "--output",dest="output",default='list_frm-10with0ftlabel.txt', help="Input output file name")
optparser.add_option("-d", "--database",dest="database",default='UCF-101', help="Input dataset name")

(opts, _) = optparser.parse_args()
data_root = '/cs/vml2/xla193/cluster_video'
output_path= os.path.join( data_root, 'output' )

def combine_pl( labelin, pathin, countin, pathout, database):
	label_file = os.path.join( output_path, database, labelin )
	path_file  = os.path.join( output_path, database, pathin )
	output_file= os.path.join( output_path, database, pathout)
	count_file = os.path.join( output_path, database, countin )
	
	with open(path_file, 'r') as pathf:
		paths = pathf.readlines()
	with open(count_file,'r') as countf:
		counts= countf.readlines()

	# pdb.set_trace()

	with open(output_file, 'w') as outf:
		label_dic = sio.loadmat( label_file )
		# pdb.set_trace()
		labels    = label_dic['pdlabels'][0]
		video_id  = 0
		frame_id  = 0
		num_fms   = int( counts[video_id] )
		for path in paths:
			(image_path, label_indicator) = path.strip().split(' ')
			frame_id += 1
			if (frame_id < num_fms):
				# pdb.set_trace()
				content = image_path + ' ' + str(int(labels[video_id])) + '\n'
				outf.write(content)
				print "writing: ", content
				# continue
			if (frame_id == num_fms):
				content = image_path + ' ' + str(int(labels[video_id])) + '\n'
				outf.write(content)
				print "writing: ", content
				video_id += 1
				frame_id = 0
				if video_id == 1374:
					print "Done"
					break
				num_fms  = int( counts[video_id] )


if __name__ == '__main__':
	combine_pl( opts.label, opts.path, opts.count, opts.output, opts.database )