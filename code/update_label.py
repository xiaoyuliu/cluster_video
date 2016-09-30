
input_file = '/cs/vml2/xla193/cluster_video/output/UCF-101/input_UCF-101_20_0ft_lrcn.txt'
output_file = '/cs/vml2/xla193/cluster_video/output/UCF-101/input_UCF-101_20_0ft_lrcn_new.txt'
with open(input_file, 'r') as inf:
  lines = inf.readlines()

with open(output_file, 'w') as ouf:
  for line in lines:
    ( path, label ) = line.strip().split(' ')
    content = path + ' ' + str(int(label)+1) + '\n'
    print content
    ouf.write( content )
