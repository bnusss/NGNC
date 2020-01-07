# dyn same & loss seperate & no snsd
from model import *
from utils import *
import torch
import time
import torch.nn.functional as F
import torch.nn.utils as U
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse


# configuration
parser = argparse.ArgumentParser(description='reconstruct the gene regulatory network')
# system config
parser.add_argument('--desc', type=str, default='penalize for structure sparcity * 0.0001', help='description')
parser.add_argument('--node_num', type=int, default=100, help='node num')
parser.add_argument('--dim', type=int, default=1, help='information diminsion of each node')
parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
parser.add_argument('--epoch', type=int, default=5000, help='epoch')
parser.add_argument('--save', type=bool, default=True, help='weather to save the result')
parser.add_argument('--sam_num', type=int, default=1562, help='sample number')
parser.add_argument('--time_lag', type=int, default=1, help='time lag')
parser.add_argument('--cuda', type=int, default=0, help='which gpu to use')

# generator config
parser.add_argument('--lrnet', type=float, default=0.004, help='lr for net generator')
parser.add_argument('--hard_sample', type=bool, default=False, help='iweather to use hard mode in gumbel')
parser.add_argument('--sample_time', type=int, default=1, help='sample time while training')
parser.add_argument('--temp', type=int, default=1,help='temperature')
parser.add_argument('--drop_frac', type=float, default=0.9999, help='temperature drop frac')

# dyn learner config
parser.add_argument('--lrdyn', type=float, default=0.001,help='lr for dyn learner')
parser.add_argument('--layers', type=int, default=5,help='layer num')
parser.add_argument('--units', type=int, default=256,help='units in each layer')

# gloabal config
parser.add_argument('--seed', type=int, default=2050,help='seed')
parser.add_argument('--start', type=str, default='0',help='start time')
parser.add_argument('--use_gumbel', type=int, default=1,help='weather to use gumbel(or answer) to reconstruct')
args = parser.parse_args()




torch.cuda.set_device(args.cuda)


print('hyper parameters configuration:')
print(str(args))

torch.manual_seed(args.seed)
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
print('start_time:',start_time)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# generator
generator = Gumbel_Generator_Old(sz=args.node_num,temp=args.temp,temp_drop_frac=args.drop_frac).to(device)
# optimizer
op_net = optim.Adam(generator.parameters(), lr=args.lrnet)


# dyn learner
dyn_learners = []
for i in range(args.node_num):
	dyn_learners.append(G_MLP(args.node_num,args.layers,args.units).to(device))
print(dyn_learners[0])

# optimizer
op_dyns = []
for i in range(args.node_num):
	op_dyns.append(optim.Adam(dyn_learners[i].parameters(),lr=args.lrdyn))



# data
loaders,object_matrix = load_gene(batch_size=args.batch_size,node=args.node_num,load_had=0,sam_num = args.sam_num,time_lag=args.time_lag,seed=args.seed)

object_matrix = object_matrix.cpu().numpy()



# indices to draw the curve
NUM_OF_1 = []

# mxs = []
# mns = []
def train_dyn_gen():
	loss_batch = []
	mse_batch = []
	
	print('current temp:',generator.temperature)
	NUM_OF_1.append(torch.sum(generator.sample_all()))


	for j in range(args.node_num):
		print('training node:'+str(j))
		for idx,data in enumerate(loaders[j]):
			data = data.to(device)

			x = data[:,:,0,:]
			y = data[:,:,1,:]

			# zero grad
			op_net.zero_grad()
			op_dyns[j].zero_grad()

			# predict and caculate the loss
			if args.use_gumbel:
				adj_col = generator.sample_adj_i(j,hard=args.hard_sample,sample_time=args.sample_time).to(device)
			else:
				adj_col = torch.from_numpy(object_matrix[:,j]).to(device)

			y_hat = dyn_learners[j](x,adj_col,j)
			loss = torch.mean(torch.abs(y_hat - y[:,j,:]))
	

			# backward and optimize
			loss.backward()
			op_net.step()
			op_dyns[j].step()
			loss_batch.append(loss.item())

	# used for more than 10 nodes
	op_net.zero_grad()
	loss = (torch.sum(generator.sample_all()))*0.0001 # 极大似然
	loss.backward()
	op_net.step()

	err,tp,tn,fp,fn = constructor_evaluator(generator,20,np.float32(object_matrix))

	print('loss:'+str(np.mean(loss_batch))+' err:'+str(round(err,2))+' tp:'+str(tp)+' fp:'+str(fp)+' tn:'+str(tn)+ ' fn:'+str(fn))

	# save the model
	torch.save(generator.cpu(),'./model/'+start_time+'_gen.pkl')
	# torch.save(dyn_isom.cpu(),'./model/'+start_time+'_dyn.pkl')
	generator.cuda()
	# dyn_isom.cuda()
	torch.save(object_matrix,'./model/'+start_time+'_mat.pkl')


	cpus = []
	for i in range(args.node_num):
		cpus.append(dyn_learners[i].cpu())
	torch.save(cpus,'./model/'+start_time+'_dyns.pkl')
	for i in range(args.node_num):
		dyn_learners[i].cuda()
	

	return np.mean(loss_batch),err,tp,tn,fp,fn


 
# start training
loss_totalbyepoch = []
err_totalbyepoch = []
tpr_totalbyepoch = []
fpr_totalbyepoch = []

# each training epoch
for e in range(args.epoch):
	print('\nepoch',e)

	# train both dyn learner and generator together
	loss,err,tp,tn,fp,fn = train_dyn_gen()
	
	# record result for each epoch
	loss_totalbyepoch.append(loss)
	err_totalbyepoch.append(err)


# save the result or not
if args.save == False:
	print('dont save and break')
	d()

last_num_of_1 = NUM_OF_1[-1]
end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
res = {'loss':loss,'err_net':err,'tp':tp,'fp':fp,'tn:':tn,'fn':fn,'numof1':last_num_of_1,'end_time':end_time}
save_exp_res(args,res,start_time,file_addr='./log/exp_log_snsd.txt')


plt.figure(figsize=(8,16))

write_info = ''
s = str(args)
for i in range(len(s)):
	write_info += s[i:i+1]
	if i % 50 == 0:
		write_info += '\n'


write_info += '\n'
write_info += 'loss:'+str(round(loss,6))+' err:'+str(round(err,2))+' tp:'+str(round(tp,2))+'fp:'+str(round(fp,2))+'tn:'+str(round(tn,2))+'fn:'+str(round(fn,2))+' end_time:'+end_time




plt.subplot(511)
plt.xticks(())
plt.yticks(())
plt.text(0.1,0.1,write_info,fontsize=8)

plt.subplot(5,1,2)
plt.ylabel('mse')
# plt.xticks(())
plt.plot(loss_totalbyepoch)

plt.subplot(5,1,3)
# plt.xticks(())
plt.ylabel('err_net')
plt.plot(err_totalbyepoch)

plt.subplot(5,1,4)
plt.ylabel('tpr and fpr')
# plt.xticks(())
plt.plot(tpr_totalbyepoch,color='red', label='tpr')
plt.plot(fpr_totalbyepoch,color='blue', label='fpr')
plt.legend(loc='best',frameon=True)

plt.subplot(5,1,5)
plt.ylabel('num of 1(abs)')
plt.plot(NUM_OF_1)

plt.show()

plt.savefig('./log/ex_log'+start_time+'.png')