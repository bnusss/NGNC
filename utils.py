import torch
import os
import pickle
import math
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix


# if use cuda
use_cuda = torch.cuda.is_available()



# ----------------- stop control ------------------------

# record one experiment
def record_stop(start_time,HYP):
    # if already exist
    if os.path.exists('stop_control.pickle'):
        with open('stop_control.pickle','rb') as f:
            stop_cont_obj = pickle.load(f)
        stop_cont_obj[start_time] = {'stop':False,'HYP':HYP}
    # if file dont exist
    else:
        stop_cont_obj = {start_time:{'stop':False,'HYP':HYP}}
    # save the result
    with open('stop_control.pickle','wb') as f:
        pickle.dump(stop_cont_obj,f)
    return True

# weather to stop early
def if_stop(start_time):
    with open('stop_control.pickle','rb') as f:
        stop_control = pickle.load(f)
    if stop_control[start_time]['stop'] == True:
        return True
    else:
        return False

# ----------------- stop control ------------------------





# ------------------ load data --------------------



# extend the data for a pertular node
# parameter:
    # data:[21,node num]
# return: [21-time_lag,2,node num]
def ts_extend(data,time_lag):
    # 0.65=>0.6
    # data = torch.tensor(data)
    # data = (data*10).int().float()/10
    # data = data.tolist()

    # 去掉序号的一列
    target = []
    data = torch.tensor(data)[:,1:].tolist()# 【20，node_num】

    for i in range(len(data)-time_lag):
        target.append([data[i],data[i+time_lag]])
    return torch.tensor(target)



def add_edge(adj,g0,g1,value):
	if int(value) == 1:
		adj[int(g0[1:])-1][int(g1[1:])-1] = 1
	return adj

# load gene data
# parameter:
	# batch_size:
	# node
	# step: length in each sample
# return train_loader,val_loader,test_loader
def load_gene(batch_size=4096,node=10,step=21,load_had = 0,sam_num=10000,time_lag=1,seed=2050):

    random.seed(seed)
    # global config
    global root_path
    global time_series_path
    global p_path
    global adj_path

    if node == 10:
        # root file folder
        root_path = '/data/zzhang/data/gene_wlf/DREAM4_insilico_size10_1/'
        time_series_path = 'insilico_size10_1_dream4_timeseries.tsv' # with noise
        # time_series_path = 'insilico_size10_1_nonoise_dream4_timeseries.tsv' # no noise
        # pertubation file path
        p_path = 'insilico_size10_1_dream4_timeseries_perturbations.tsv'
        # adj file path
        adj_path = 'insilico_size10_1_goldstandard.tsv'
    elif node == 100:
        root_path = '/data/zzhang/data/gene_100/100_2/' 
        time_series_path = 'insilico_size100_2_dream4_timeseries.tsv' # with noise
        # time_series_path = 'insilico_size100_2_nonoise_dream4_timeseries.tsv' # no noise
        p_path = 'insilico_size100_2_dream4_timeseries_perturbations.tsv'
        adj_path = 'insilico_size100_2_goldstandard.tsv'


    # where save after loading
    data_pkl_add = './data/data_100_1_s1k_nono.pkl'
    adj_pkl_add = './data/adj_100_s1k_nono.pkl'

    # 已经load过
    if load_had == 1:
        with open(data_pkl_add,'rb') as f:
            data_no_p = pickle.load(f)
        with open(adj_pkl_add,'rb') as f:
            adj = pickle.load(f)
        return data_no_p,adj



    ts_data=pd.read_csv(root_path+time_series_path, sep='\t')
    ts_list = list(ts_data.values)
    data = []
    all_num = int(len(list(ts_data['G1'])) / step)
    sample_num = all_num
    sample_num = sam_num
    p_list = load_pertubations()

    

    # how many room do we need
    # 计算每个节点需要的数据存储空间，提前分配好内存，以免临时分配导致后面load数据越来越慢
    print('space caculating ...')
    start_pos = int(random.random()*int(all_num-sample_num))
    print(start_pos)
    rooms = torch.zeros(node)
    for i in range(start_pos,start_pos+int(sample_num)):
        # print(i)
        if i % 1000 == 0:
            print(i)
        for j in range(node):
            if p_list[i][j] == 0:
                # 每次采样占用的空间
                rooms[j] += 21 - time_lag
    print('space caculating finished')
    data_no_p = []
    for i in range(node):
        print('distributing space for node '+str(i)+'...')
        print(int(rooms[i].tolist()),2,node)
        data_no_p.append(torch.zeros(int(rooms[i].tolist()),2,node))


    # 分配好空间了，直接把数据填充到对应的位置即可
    print('time series data gathering...')
    marks = torch.zeros(node)
    for i in range(start_pos,start_pos+int(sample_num)):
        if i % 100 == 0:
            print(i)
        for j in range(node):
            if p_list[i][j] == 0:
                data_no_p[j][int(marks[j]):int(marks[j]+21-time_lag)] = ts_extend(ts_list[i*step:i*step+step],time_lag)
                marks[j] += 21-time_lag

    # generate train loaders
    for i in range(node):
        # [sample,2,node] => [sample,node,2,1]
        data_no_p[i] = data_no_p[i].transpose(1,2).unsqueeze(3)
        data_no_p[i] = DataLoader(data_no_p[i], batch_size=batch_size, shuffle=True)

    # adj
    adj_data = pd.read_csv(root_path+adj_path,sep='\t')
    # initialize it to be zero
    adj = torch.zeros(node,node)
    # add first line
    adj = add_edge(adj,adj_data.columns[0],adj_data.columns[1],adj_data.columns[2])
    # add other lines
    list_value = list(adj_data.values)
    for i in range(int(len(list(adj_data.values)))):
        adj = add_edge(adj,list_value[i][0],list_value[i][1],list_value[i][2])


    return data_no_p,adj

def load_pertubations():
    # 100 nodes
    # p_path = 'insilico_size100_1_dream4_timeseries_perturbations.tsv'
    p_data=pd.read_csv(root_path+p_path, sep='\t')
    list_value = list(p_data.values)
    return list_value

def save_exp_res(conf,res,time,file_addr='./log/log_exp.txt'):
    # check existance
    k_str = '----------Configuration----------\n'
    # for k in conf:
    #     k_str += k+':'+str(conf[k])+'\n'
    # namespcace
    k_str += str(conf)
    k_str += '\n'

    v_str = '----------Results----------\n'
    for v in res:
        v_str += v+':'+str(res[v])+'\n'
    # write
    f = open(file_addr,'a')
    f.write('\n\n***********************************n')
    f.write(time+'\n')
    f.write(k_str)
    f.write(v_str+'\n')

# load_gene()
# load_pertubations()

# ------------------ load data --------------------





# ------------------------- construct evaluator -----------------------


def tpr_fpr(out,adj):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(out.shape[0]):
        for j in range(out.shape[0]):
            if adj[i][j] == 1:
                # positive
                if out[i][j] == 1:
                    # true positive
                    tp += 1
                else:
                    # false positive
                    fn += 1
            else:
                # negative
                if out[i][j] == 1:
                    # true negative
                    fp += 1
                else:
                    # false negative
                    tn += 1
    # tpr = tp /  (tp + fp)

    # try:
    #     tpr = float(tp) / (tp + fn)
    # except ZeroDivisionError:
    #     tpr=0
    # try:
    #     fpr = float(fp) / (fp + tn)
    # except ZeroDivisionError:
    #     fpr = 0
    return tp,tn,fp,fn

def calc_tpr_fpr(matrix, matrix_pred):
    matrix = matrix.to('cpu').data.numpy()
    matrix_pred = matrix_pred.to('cpu').data.numpy()

    # 去对角元素
    # matrix = skip_diag_strided(matrix)
    # matrix_pred = skip_diag_strided(matrix_pred)

    # 计算指标
    # print(matrix)
    # print(matrix_pred)
    # tn, fp, fn, tp = confusion_matrix(matrix.astype(int).reshape(-1),
    #                                   matrix_pred.astype(int).reshape(-1)).ravel()

    # print(tn, fp, fn, tp)

    tp,tn,fp,fn = tpr_fpr(matrix_pred,matrix)

    # return tpr, fpr
    return tp,tn,fp,fn

def constructor_evaluator(gumbel_generator, tests, obj_matrix):
    # obj_matrix = obj_matrix.cuda()
    tps = []
    tns = []
    fps = []
    fns = []
    errs = []
    obj_matrix = torch.abs(torch.from_numpy(obj_matrix))
    for t in range(tests):
        out_matrix = torch.abs(gumbel_generator.sample_all(hard=True).cpu())
        err = torch.sum(torch.abs(out_matrix-obj_matrix))
        # out_matrix_c = 1.0*(torch.sign(out_matrix-1/2)+1)/2
        # err = torch.sum(torch.abs(out_matrix_c * get_offdiag(sz) - obj_matrix * get_offdiag(sz)))
        err = err.cpu() if use_cuda else err
        # if we got nan in err
        if math.isnan(err):
            print('problem cocured')
            # torch.save(gumbel_generator,'problem_generator_genchange.model')
            d()
            t=t-1
            continue
        errs.append(err.data.numpy())
        # tpr,fpr = calc_tpr_fpr(out_matrix,obj_matrix)
        tp,tn,fp,fn = calc_tpr_fpr(out_matrix,obj_matrix)
        # print(tpr)
        tps.append(tp)
        fps.append(fp)
        tns.append(tn)
        fns.append(fn)
    # print(out_matrix)
        
    err_net = np.mean(errs)
    tp = np.mean(tps)
    tn = np.mean(tns)
    fp = np.mean(fps)
    fn = np.mean(fns)
    return err_net,tp,tn,fp,fn


# ------------------------- construct evaluator -----------------------


