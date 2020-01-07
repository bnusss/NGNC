import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GGN_IO(nn.Module):
	def __init__(self,node_num,dim,hid):
		super(GGN_IO, self).__init__()
		self.node_num = node_num
		self.dim = dim
		self.hid = hid
		self.n2e = nn.Linear(2*dim,hid)
		self.e2e = nn.Linear(hid,hid)
		self.e2n = nn.Linear(hid,hid)
		self.n2e = nn.Linear(2*dim,hid)
		self.output = nn.Linear(dim+hid,dim)
	def forward(self, x, adj_col, i):
		# x : features of all nodes at time t,[n*d]
		# adj_col : i th column of adj mat,[n*1]
		# i : just i
		starter = x
		ender = x[i].repeat(self.node_num).view(self.node_num,-1)
		x = torch.cat((starter,ender),1)
		x = F.relu(self.n2e(x))
		x = F.relu(self.e2e(x))
		x = x * adj_col.unsqueeze(1).expand(self.node_num,self.hid)
		x = torch.sum(x,0)
		x = self.e2n(x)
		x = torch.cat((starter[i],x),dim=-1)
		x = self.output(x)

		# skip connection
		x = starter[i]+x
		return x

class GGN_IO_B(nn.Module):
	"""docstring for GGN_IO_B"""
	def __init__(self, node_num,dim,hid):
		super(GGN_IO_B, self).__init__()
		self.node_num = node_num
		self.dim = dim
		self.hid = hid
		self.n2e = nn.Linear(2*dim,hid)
		self.e2e = nn.Linear(hid,hid)
		self.e2n = nn.Linear(hid,hid)
		self.n2n = nn.Linear(hid,hid)
		self.output = nn.Linear(dim+hid,dim)
	def forward(self, x, adj_col, i):
		# print(adj_col)
		# d()
		# x : features of all nodes at time t,[b*n*d]
		# adj_col : i th column of adj mat,[n*1]
		# i : just i
		starter = x # 128,10,4
		ender = x[:,i,:] # 128,4
		ender = ender.unsqueeze(1) #128,1,4
		
		ender = ender.expand(starter.size(0),starter.size(1),starter.size(2)) #128,10,4
		x = torch.cat((starter,ender),2) #128,10,8
		x = F.relu(self.n2e(x))#128,10,256

		x = F.relu(self.e2e(x))#128,10,256
		x = x * adj_col.unsqueeze(1).expand(self.node_num,self.hid)#128,10,256

		x = torch.sum(x,1)#128,256
		x = F.relu(self.e2n(x))#128,256
		x = F.relu(self.n2n(x))#128,256

		x = torch.cat((starter[:,i,:],x),dim=-1)#128,256+4
		x = self.output(x)#128,4

		# skip connection
		# x = starter[:,i,:]+x # dont want in CML
		return x



class GGN_SEP_B(nn.Module):
	"""分开自己和邻居"""
	def __init__(self, node_num,dim,hid):
		super(GGN_SEP_B, self).__init__()
		self.node_num = node_num
		self.dim = dim
		self.hid = hid
		self.n2e = nn.Linear(2*dim,hid)
		self.e2e = nn.Linear(hid,hid)
		self.e2n = nn.Linear(hid,hid)
		self.n2n = nn.Linear(hid,hid)
		self.selflin = nn.Linear(dim,hid)
		self.output = nn.Linear(2*hid,dim)
	def forward(self, x, adj_col, i):
		# print(adj_col)
		# d()
		# x : features of all nodes at time t,[b*n*d]
		# adj_col : i th column of adj mat,[n*1]
		# i : just i
		starter = x # 128,10,4
		ender = x[:,i,:] # 128,4
		# print('ender')
		# print(ender.size())
		ender = ender.unsqueeze(1) # 128,1,4
		# print('ender unsqueeze')
		# print(ender.size())
		
		ender = ender.expand(starter.size(0),starter.size(1),starter.size(2)) #128,10,4
		# print('ender expand')
		# print(ender.size())
		
		x = torch.cat((starter,ender),2) #128,10,8
		# print('cat')
		# print(x.size())
		
		x = F.relu(self.n2e(x)) #128,10,256
		# print('n2e')
		# print(x.size())

		x = F.relu(self.e2e(x)) #128,10,256
		# print('e2e')
		# print(x.size())
		x = x * adj_col.unsqueeze(1).expand(self.node_num,self.hid) #128,10,256
		# print('times the col of adj mat')
		# print(x.size())

		x = torch.sum(x,1)
		# print('reduced sum')
		# print(x.size())
		x = self.e2n(x)
		# print('e2n')
		# print(x.size())
		x = self.n2n(x)
		# print('n2n')
		# print(x.size())

		# self information transformation
		starter = F.relu(self.selflin(starter[:,i,:]))
		# print('linear transformation for self node')
		# print(starter.size())

		# cat them together
		x = torch.cat((starter,x),dim=-1)
		# print('cat self and neighbor')
		# print(x.size())
		x = self.output(x)
		# print('output')
		# print(x.size())
		# d()

		# skip connection
		# x = starter[:,i,:]+x # dont want in CML
		return x




class MrnaLayer(nn.Module):
	def __init__(self, node_num,dim,hid):
		super(MrnaLayer, self).__init__()
		self.lamda = Parameter(torch.rand(node_num)).to(device)
		self.mi = Parameter(torch.rand(node_num)).to(device)

		self.node_num = node_num
		self.dim = dim
		self.hid = hid
		self.n2e = nn.Linear(2*dim,hid)
		self.e2e = nn.Linear(hid,hid)
		self.e2n = nn.Linear(hid,hid)
		self.n2n = nn.Linear(hid,dim)

	def regulation(self,x, adj_col,i):
		starter = x # 128,10,4
		ender = x[:,i,:] # 128,4
		ender = ender.unsqueeze(1) #128,1,4
		
		ender = ender.expand(starter.size(0),starter.size(1),starter.size(2)) #128,10,4
		x = torch.cat((starter,ender),2) #128,10,8
		x = F.relu(self.n2e(x))#128,10,256
		x = F.relu(self.e2e(x))#128,10,256
		x = x * adj_col.unsqueeze(1).expand(self.node_num,self.hid)#128,10,256
		# x = x * attr_col.unsqueeze(1).expand(self.node_num,self.hid)#128,10,256

		x = torch.sum(x,1)#128,256
		x = F.relu(self.e2n(x))#128,256
		x = F.relu(self.n2n(x))#128,256
		return x

	def forward(self,x,adj_col,i):
		x_degrade = x[:,i,:] * self.lamda[None, i, None]
		x_regulation = self.regulation(x, adj_col,i)
		x = x_regulation - x_degrade
		return x



class GGN_Gene(nn.Module):
	"""docstring for GGN_Gene"""
	def __init__(self, node_num,dim,hid):
		super(GGN_Gene, self).__init__()
		self.meta_mrna = MrnaLayer(node_num,dim,hid)

	def forward(self,x,adj_col,i):
		dx = self.meta_mrna(x, adj_col,i)
		x = x[:,i,:] + dx
		# x = F.relu6(x) / 6.
		return x

class GGN_Mlp(nn.Module):
	"""docstring for GGN_Mlp"""
	def __init__(self,node_num,dim, hid):
		super(GGN_Mlp, self).__init__()
		self.nn = node_num #10
		self.hid = hid #256
		self.dim = dim #1
		self.layer = 5
		self.mlp0 = nn.Linear(self.nn,self.hid)
		self.mlp1 = nn.Linear(self.hid,self.hid)
		self.mlp2 = nn.Linear(self.hid,self.hid)
		self.mlp3 = nn.Linear(self.hid,self.hid)
		self.mlp4 = nn.Linear(self.hid,self.hid)
		self.mlp5 = nn.Linear(self.hid,self.dim)
	def forward(self,x, adj_col, i):
		# x : features of all nodes at time t,[b*n*d](d=1)
		# adj_col : i th column of adj mat,[n*1]
		# i : just i
		x = x.squeeze()# 128*10
		

		# filter
		x = x * adj_col[None,:]

		# mlp
		x = F.relu(self.mlp0(x))
		x = F.relu(self.mlp1(x))
		x = F.relu(self.mlp2(x))
		x = F.relu(self.mlp3(x))
		x = F.relu(self.mlp4(x))
		x = self.mlp5(x)
		return x
		
class G_MLP(nn.Module):
	"""docstring for G_MLP"""
	def __init__(self, node_num, layers, units):
		super(G_MLP, self).__init__()
		# layer
		self.l = layers
		# units in each layer
		self.u = units
		# node num
		self.nn = node_num
		self.mlps = []
		for i in range(int(self.l)):
			# first layer
			if i == 0:
				self.mlps.append(nn.Linear(self.nn,self.u))
			# last layer
			elif i == self.l-1:
				self.mlps.append(nn.Linear(self.u,1))
			# normal layer
			else:
				self.mlps.append(nn.Linear(self.u,self.u))
		self.mlps = nn.ModuleList(self.mlps)

		# init with xavier
		for i in range(self.l):
			nn.init.xavier_normal_(self.mlps[i].weight, gain=nn.init.calculate_gain('relu'))

	def forward(self,x,adj_col,i):
		# ori_x = x
		x = x.squeeze()# 128*10
		# filter
		x = x * adj_col[None,:]
		# through mlp
		for i in range(self.l-1):
			x = F.relu(self.mlps[i](x))
		# last layer
		x = self.mlps[self.l-1](x)
		# return ori_x + dx
		return x



		

# 此类为一个利用Gumbel softmax生成离散网络的类
class Gumbel_Generator(nn.Module):
	def __init__(self, sz = 10, temp = 1, temp_drop_frac = 0.9999):
		super(Gumbel_Generator, self).__init__()
		self.sz = sz
		self.tau = temp
		self.drop_fra = temp_drop_frac
		self.gen_matrix = Parameter(torch.rand(sz, sz, 2))
		self.temperature = temp
		self.temp_drop_frac = temp_drop_frac
	def sample_adj_i(self,i,hard=True,sample_time=1):
		# mat = torch.zeros(sample_time,self.sz)
		# for m in range(sample_time):
		# 	mat[m] = F.gumbel_softmax(self.gen_matrix[:,i], tau=self.tau, hard=hard)[:,0]
		# res = torch.sum(mat,0) / sample_time
		# return res
		return F.gumbel_softmax(self.gen_matrix[:,i]+1e-8, tau=self.tau, hard=hard)[:,0]
	def ana_one_para(self):
		print(self.gen_matrix[0][0])
		return 1


	def sample_all(self,hard=True,sample_time=1):
		adj = torch.zeros(self.sz,self.sz)
		for i in range(adj.size(0)):
			temp = self.sample_adj_i(i,hard=hard,sample_time=sample_time)
			adj[:,i] = temp
		return adj
	def drop_temp(self):
		self.tau = self.tau * self.drop_fra





#############
# Functions #
#############
def gumbel_sample(shape, eps=1e-20):
	u = torch.rand(shape)
	gumbel = - np.log(- np.log(u + eps) + eps)
	if use_cuda:
		gumbel = gumbel.cuda()
	return gumbel
def gumbel_softmax_sample(logits, temperature): 
	""" Draw a sample from the Gumbel-Softmax distribution"""
	y = logits + gumbel_sample(logits.size())
	return torch.nn.functional.softmax( y / temperature, dim = 1)

def gumbel_softmax(logits, temperature, hard=False):
	"""Sample from the Gumbel-Softmax distribution and optionally discretize.
	Args:
	logits: [batch_size, n_class] unnormalized log-probs
	temperature: non-negative scalar
	hard: if True, take argmax, but differentiate w.r.t. soft sample y
	Returns:
	[batch_size, n_class] sample from the Gumbel-Softmax distribution.
	If hard=True, then the returned sample will be one-hot, otherwise it will
	be a probabilitiy distribution that sums to 1 across classes
	"""
	y = gumbel_softmax_sample(logits, temperature)
	if hard:
		k = logits.size()[-1]
		y_hard = torch.max(y.data, 1)[1]
		y = y_hard
	return y
def get_offdiag(sz):
	## 返回一个大小为sz的下对角线矩阵
	offdiag = torch.ones(sz, sz)
	for i in range(sz):
		offdiag[i, i] = 0
	if use_cuda:
		offdiag = offdiag.cuda()
	return offdiag       


#####################
# Network Generator #
#####################

# 此类为一个利用Gumbel softmax生成离散网络的类
class Gumbel_Generator_Old(nn.Module):
	def __init__(self, sz = 10, temp = 10, temp_drop_frac = 0.9999):
		super(Gumbel_Generator_Old, self).__init__()
		self.gen_matrix = Parameter(torch.rand(sz, sz, 2))
		#gen_matrix 为邻接矩阵的概率
		self.temperature = temp
		self.temp_drop_frac = temp_drop_frac
	def drop_temp(self):
		# 降温过程
		self.temperature = self.temperature * self.temp_drop_frac
	def sample_all(self, hard=False):
		# 采样——得到一个临近矩阵
		self.logp = self.gen_matrix.view(-1, 2)
		out = gumbel_softmax(self.logp, self.temperature, hard)
		if hard:
			hh = torch.zeros(self.gen_matrix.size()[0] ** 2, 2)
			for i in range(out.size()[0]):
				hh[i, out[i]] = 1
			out = hh
		if use_cuda:
			out = out.cuda()
		out_matrix = out[:,0].view(self.gen_matrix.size()[0], self.gen_matrix.size()[0])
		return out_matrix
	def sample_adj_i(self,i,hard=False,sample_time=1):
		self.logp = self.gen_matrix[:,i]
		out = gumbel_softmax(self.logp, self.temperature, hard=hard)
		if use_cuda:
			out = out.cuda()
		out_matrix = out[:,0]
		return out_matrix


	def get_temperature(self):
		return self.temperature
	def get_cross_entropy(self, obj_matrix):
		# 计算与目标矩阵的距离
		logps = F.softmax(self.gen_matrix, 2)
		logps = torch.log(logps[:,:,0] + 1e-10) * obj_matrix + torch.log(logps[:,:,1] + 1e-10) * (1 - obj_matrix)
		result = - torch.sum(logps)
		result = result.cpu() if use_cuda else result
		return result.data.numpy()
	def get_entropy(self):
		logps = F.softmax(self.gen_matrix, 2)
		result = torch.mean(torch.sum(logps * torch.log(logps + 1e-10), 1))
		result = result.cpu() if use_cuda else result
		return(- result.data.numpy())
	def randomization(self, fraction):
		# 将gen_matrix重新随机初始化，fraction为重置比特的比例
		sz = self.gen_matrix.size()[0]
		numbers = int(fraction * sz * sz)
		original = self.gen_matrix.cpu().data.numpy()
	    
		for i in range(numbers):
			ii = np.random.choice(range(sz), (2, 1))
			z = torch.rand(2).cuda() if use_cuda else torch.rand(2)
			self.gen_matrix.data[ii[0], ii[1], :] = z

# 此类为一个利用Gumbel softmax生成离散网络的类
class Gumbel_Generator_Att(nn.Module):
	def __init__(self, sz = 10, temp = 10, temp_drop_frac = 0.9999):
		super(Gumbel_Generator_Att, self).__init__()
		self.gen_matrix = Parameter(torch.rand(sz, sz, 2))
		#gen_matrix 为邻接矩阵的概率
		self.temperature = temp
		self.temp_drop_frac = temp_drop_frac
	def drop_temp(self):
		# 降温过程
		self.temperature = self.temperature * self.temp_drop_frac
	def sample_all(self, hard=False):
		# 采样——得到一个临近矩阵
		self.logp = self.gen_matrix.view(-1, 2)
		out = gumbel_softmax(self.logp, self.temperature, hard)
		if hard:
			hh = torch.zeros(self.gen_matrix.size()[0] ** 2, 2)
			for i in range(out.size()[0]):
				hh[i, out[i]] = 1
			out = hh
		if use_cuda:
			out = out.cuda()
		out_matrix = out[:,0].view(self.gen_matrix.size()[0], self.gen_matrix.size()[0])
		return out_matrix
	def sample_adj_i(self,i,hard=False,sample_time=1):
		self.logp = self.gen_matrix[:,i]
		out = gumbel_softmax(self.logp, self.temperature, hard=hard)
		out = 2*out-1
		if use_cuda:
			out = out.cuda()
		out_matrix = out[:,0]
		return out_matrix


	def get_temperature(self):
		return self.temperature
	def get_cross_entropy(self, obj_matrix):
		# 计算与目标矩阵的距离
		logps = F.softmax(self.gen_matrix, 2)
		logps = torch.log(logps[:,:,0] + 1e-10) * obj_matrix + torch.log(logps[:,:,1] + 1e-10) * (1 - obj_matrix)
		result = - torch.sum(logps)
		result = result.cpu() if use_cuda else result
		return result.data.numpy()
	def get_entropy(self):
		logps = F.softmax(self.gen_matrix, 2)
		result = torch.mean(torch.sum(logps * torch.log(logps + 1e-10), 1))
		result = result.cpu() if use_cuda else result
		return(- result.data.numpy())
	def randomization(self, fraction):
		# 将gen_matrix重新随机初始化，fraction为重置比特的比例
		sz = self.gen_matrix.size()[0]
		numbers = int(fraction * sz * sz)
		original = self.gen_matrix.cpu().data.numpy()
	    
		for i in range(numbers):
			ii = np.random.choice(range(sz), (2, 1))
			z = torch.rand(2).cuda() if use_cuda else torch.rand(2)
			self.gen_matrix.data[ii[0], ii[1], :] = z



class GumbelGraphSign(nn.Module):
	def __init__(self, sz=10, temp=10, temp_drop_frac=0.9999):
		super(GumbelGraphSign, self).__init__()
		self.gumbel_generator_1 = Gumbel_Generator_Old(sz=sz, temp=temp, temp_drop_frac=temp_drop_frac).to(device)
		self.gumbel_generator_2 = Gumbel_Generator_Old(sz=sz, temp=temp, temp_drop_frac=temp_drop_frac).to(device)

	def sample_all(self, hard=False):
		matrix1 = self.gumbel_generator_1.sample_all(hard=hard)
		matrix2 = self.gumbel_generator_2.sample_all(hard=hard)
		return matrix1 - matrix2
	def sample_adj_i(self,i,hard=False,sample_time=1):
		col1 = self.gumbel_generator_1.sample_adj_i(i,hard,sample_time)
		col2 = self.gumbel_generator_2.sample_adj_i(i,hard,sample_time)
		return col1-col2


