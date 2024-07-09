import argparse
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import torch
import time
import yaml
import random
from plot import plot_output 
import os

from model import utils, WFCG
from loadData import data_reader, split_data
from createGraph import rdSLIC, create_graph
import sys

parser = argparse.ArgumentParser(description='FDGC')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--path-config', type=str, default='config/config.yaml') # here
parser.add_argument('-pc', '--print-config', action='store_true', default=False)
parser.add_argument('-pdi','--print-data-info', action='store_true', default=False)
parser.add_argument('-sr','--show-results', action='store_true', default=False)
parser.add_argument('--save-results', action='store_true', default=True)
args = parser.parse_args()  # running in command line


config = yaml.load(open(args.path_config, "r"), Loader=yaml.FullLoader)
dataset_name = config["data_input"]["dataset_name"]
traind_on = config["data_input"]["traind_on"]
class_count = config["data_input"]["classes"]
samples_type = config["data_split"]["samples_type"]
split_size = config["data_split"]["split_size"]
train_num = config["data_split"]["train_num"]
val_num = config["data_split"]["val_num"]
train_ratio = config["data_split"]["train_ratio"]
val_ratio = config["data_split"]["val_ratio"]
superpixel_scale = config["data_split"]["superpixel_scale"]
max_epoch = config["network_config"]["max_epoch"]
learning_rate = config["network_config"]["learning_rate"]
weight_decay = config["network_config"]["weight_decay"]
lb_smooth = config["network_config"]["lb_smooth"]
path_weight = config["result_output"]["path_weight"]
path_result = config["result_output"]["path_result"]
path = path_result + dataset_name + "/WFCG"


def load_data(dataset_name, traind_on):

    dr = data_reader.DataReader(dataset_name, traind_on)

    data = dr.normal_cube
    data_train_gt = dr.train
    data_test_gt = dr.test
    

    return data, data_train_gt, data_test_gt



data, data_train_gt, data_test_gt = load_data(dataset_name, traind_on)

class_num = np.max(data_test_gt)
print("class_num: ", class_num)
height, width, bands = data.shape
data_train_gt_reshape = np.reshape(data_train_gt, [-1])
data_test_gt_reshape = np.reshape(data_test_gt, [-1])
# load config

if args.print_config:
    print(config)





train_num = (int)(data_train_gt.shape[0] * data_train_gt.shape[1] * split_size // class_count)
print("data:    ", data_train_gt.shape[0] * data_train_gt.shape[1])
print("split_size:  ", split_size)
print("class_count: ",class_count)
train_index, val_index, test_index = split_data.split_data(data_train_gt_reshape, class_num, train_ratio, val_ratio, train_num, val_num, samples_type)

# create graph
train_samples_gt,test_samples_gt, val_samples_gt = create_graph.get_label(data_train_gt_reshape,data_test_gt_reshape,
                                                 train_index, val_index, test_index)

# create_graph.get_label_mask use data_train_gt shape only
train_label_mask, test_label_mask, val_label_mask = create_graph.get_label_mask(train_samples_gt, test_samples_gt, val_samples_gt, data_train_gt, class_num)



# label transfer to one-hot encode
train_gt = np.reshape(train_samples_gt,[height,width])
test_gt = np.reshape(test_samples_gt,[height,width])
val_gt = np.reshape(val_samples_gt,[height,width])

if args.print_data_info:
    data_reader.data_info(train_gt, val_gt, test_gt)

train_gt_onehot = create_graph.label_to_one_hot(train_gt, class_num)
test_gt_onehot = create_graph.label_to_one_hot(test_gt, class_num)
val_gt_onehot = create_graph.label_to_one_hot(val_gt, class_num)

# superpixels
ls = rdSLIC.LDA_SLIC(data, train_gt, class_num-1)
tic0=time.time()
Q, S ,A, Seg= ls.simple_superpixel(scale=superpixel_scale)
toc0 = time.time()
LDA_SLIC_Time=toc0-tic0

Q=torch.from_numpy(Q).to(args.device)
A=torch.from_numpy(A).to(args.device)

train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(args.device)
test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(args.device)
val_samples_gt=torch.from_numpy(val_samples_gt.astype(np.float32)).to(args.device)

train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).to(args.device)
test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(args.device)
val_gt_onehot = torch.from_numpy(val_gt_onehot.astype(np.float32)).to(args.device)

train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(args.device)
test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(args.device)
val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(args.device)


net_input=np.array(data, np.float32)
net_input=torch.from_numpy(net_input.astype(np.float32)).to(args.device)

# model
net = WFCG.WFCG(height, width, bands, class_num, Q, A).to(args.device)

# train
print("\n\n==================== train ====================\n")
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate, weight_decay=weight_decay) #, weight_decay=0.0001
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
zeros = torch.zeros([height * width]).to(args.device).float()
best_loss = 99999
net.train()
tic1 = time.time()
for i in range(max_epoch+1):
    optimizer.zero_grad()  # zero the gradient buffers
    output= net(net_input) # harc? 
    loss = utils.compute_loss(output, train_gt_onehot, train_label_mask)
    loss.backward(retain_graph=False)
    optimizer.step()  # Does the update
    
    # if i%10==0:
    with torch.no_grad():
        net.eval()
        output= net(net_input)
        trainloss = utils.compute_loss(output, train_gt_onehot, train_label_mask)
        trainOA = utils.evaluate_performance(output, train_samples_gt, train_gt_onehot, zeros)
        valloss = utils.compute_loss(output, val_gt_onehot, val_label_mask)
        valOA = utils.evaluate_performance(output, val_samples_gt, val_gt_onehot, zeros)
        # print("{}\ttrain loss={:.4f}\t train OA={:.4f} val loss={:.4f}\t val OA={:.4f}".format(str(i + 1), trainloss, trainOA, valloss, valOA))

        if valloss < best_loss :
            best_loss = valloss
            path_model = path + "/weights" 
            os.makedirs(path_model, exist_ok=True)
            torch.save(net.state_dict(), path_model + "/" r"model.pt")
            # print('save model...')
    # scheduler.step(valloss)
    torch.cuda.empty_cache()
    net.train()

    if i%10==0:
        print("{}\ttrain loss={:.4f}\t train OA={:.4f} val loss={:.4f}\t val OA={:.4f}".format(str(i + 1), trainloss, trainOA, valloss, valOA))
toc1 = time.time()

print("\n\n====================training done. starting evaluation...========================\n")



# test
torch.cuda.empty_cache()
with torch.no_grad():
    net.load_state_dict(torch.load(path + "/weights/" + r"model.pt"))
    net.eval()
    tic2 = time.time()
    output = net(net_input)
    toc2 = time.time()
    testloss = utils.compute_loss(output, test_gt_onehot, test_label_mask)
    testOA = utils.evaluate_performance(output, test_samples_gt, test_gt_onehot, zeros)
    print("{}\ttest loss={:.4f}\t test OA={:.4f}".format(str(i + 1), testloss, testOA))


torch.cuda.empty_cache()
del net

LDA_SLIC_Time=toc0-tic0
# print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))
training_time = toc1 - tic1 + LDA_SLIC_Time
testing_time = toc2 - tic2 + LDA_SLIC_Time
training_time, testing_time

# classification report
test_label_mask_cpu = test_label_mask.cpu().numpy()[:,0].astype('bool')
test_samples_gt_cpu = test_samples_gt.cpu().numpy().astype('int64')
predict = torch.argmax(output, 1).cpu().numpy()

# saving predic
path_pred = path 
os.makedirs(path_pred , exist_ok=True)
np.save(path_pred + "/" + dataset_name + "_" + traind_on + "_WFCG.npy", predict)
plot_output.plot_output(predict, height, width, path_pred + "/" + dataset_name + "_" + traind_on + "_WFCG")

classification = classification_report(test_samples_gt_cpu[test_label_mask_cpu], 
                                    predict[test_label_mask_cpu]+1, digits=4)
kappa = cohen_kappa_score(test_samples_gt_cpu[test_label_mask_cpu], predict[test_label_mask_cpu]+1)

if args.show_results:
    print(classification, kappa)

# store results
if args.save_results:
    print("save results")
    #os.makedirs(path, exist_ok=True)
    run_date = time.strftime('%Y%m%d-%H%M-',time.localtime(time.time()))
    f = open(path + "/"  + dataset_name + "_"+ traind_on + '.txt', 'a+')
    str_results = '\n ======================' \
                + '\nrun data = ' + run_date \
                + "\nlearning rate = " + str(learning_rate) \
                + "\nepochs = " + str(max_epoch) \
                + "\nsamples_type = " + str(samples_type) \
                + "\nsplit_size = " + str(split_size) \
                + "\ntrain num = " + str(train_num) \
                + '\ntrain time = ' + str(training_time) \
                + '\ntest time = ' + str(testing_time) \
                + '\n' + classification \
                + "kappa = " + str(kappa) \
                + '\n'
    f.write(str_results)
    f.close()