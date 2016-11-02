import optparse, os, sys
import pdb
import scipy.io as sio

optparser = optparse.OptionParser()
optparser.add_option("-v", "--vpath", dest ="vpath", default='list_video-10withgtlabel.txt', help="video path file")
optparser.add_option("-l", "--lfile", dest ="lfile", default='label_vect0.txt', help="label file")
optparser.add_option("-o", "--ofile", dest ="ofile", default=None, help="order file")
optparser.add_option("-f",  "--ffile",dest ="ffile", default=None, help="output frame path & label file")

(opts, _) = optparser.parse_args()
data_root = '/local-scratch/xla193/cluster_video_/output/UCF-101'

def combine_usfb(video_path, labelfile, orderfile, framefile):
	with open(os.path.join(data_root, video_path), 'r') as vinf:
		videos = vinf.readlines()
	with open(os.path.join(data_root, labelfile), 'r') as linf:
		labels = linf.readlines()
	userfb_dict = sio.loadmat(os.path.join(data_root, orderfile))
	userfb_array= userfb_dict['ufids'][0]
	userfb_list = list(userfb_array)
	
	with open(os.path.join(data_root, framefile), 'w') as foutf:
		count = 0
		for id in userfb_list:
			video = videos[id]
			vpath_, nouse = video.strip().split(' ')
			vpath = os.path.join('/local-scratch/xla193/cluster_video_/datasets/UCF-101/', vpath_)
			fms = os.listdir(vpath)
			for fm in fms:
				fpath = os.path.join(vpath, fm)
				content = fpath+' '+labels[count]
				foutf.write(content)
			count += 1

if __name__ == '__main__':
	combine_usfb(opts.vpath, opts.lfile, opts.ofile, opts.ffile)