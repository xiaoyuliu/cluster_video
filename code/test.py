import pdb


ml = dict()
ml[0] = [5]
num_data = 11
const = [[0]*num_data]*num_data
i=0
pdb.set_trace()
for j in range(1,num_data):
	if j== 4:
		pdb.set_trace()
	# if j in ml[0]:
		const[0][j] = 1
		const[j][0] = 1
		print ("i: %d, j: %d, const[0][0]: %d" % (i,j,const[0][0]))
	# print const[0][10]
	# print ("i: %d, j: %d, const[0][0]: %d" % (i,j,const[0][0]))