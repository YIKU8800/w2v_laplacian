from PIL import Image
from chainer import Variable, FunctionSet, optimizers,serializers,cuda
import chainer.functions  as F
import numpy as np
#np.set_printoptions(threshold=np.inf)
np.random.seed(0)
import cupy
cupy.random.seed(0)
import scipy.io
import time
import cPickle as pickle
import chainer
import gc
import csv
import argparse

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description='process net parameters.')
parser.add_argument("--alpha", help="the alpha parameter", type=float)
parser.add_argument("--beta", help="the beta parameter", type=float, default=1.0)
args = parser.parse_args()

PICKLE_PATH="alex_net.pkl"
original_model=pickle.load(open(PICKLE_PATH))

rate = args.alpha   #alpha
weight_p = args.beta    #beta

print('begin training.........alpha = %.7f,' % args.alpha)

dim=1000

n_epoch = 500
batchsize = 100

#L=np.load('./L/L_%.2f/WL_%d.npy'%(weight_p,dim))
L=np.load('./L/L_B/WL_%d.npy'%(dim))
L=L[np.newaxis,:].astype(np.float32)
L_origin=L

for i in xrange(batchsize-1):
    L=np.concatenate((L,L_origin),axis=0)

#build and read the model
model = FunctionSet(conv1=F.Convolution2D(3,  96, 11, stride=4),
                    bn1=F.BatchNormalization(96),
                    conv2=F.Convolution2D(96, 256,  5, pad=2),
                    bn2=F.BatchNormalization(256),
                    conv3=F.Convolution2D(256, 384,  3, pad=1),
                    conv4=F.Convolution2D(384, 384,  3, pad=1),
                    conv5=F.Convolution2D(384, 256,  3, pad=1),
                    fc6=F.Linear(2304,1024),
                    fc7=F.Linear(1024, 260))
## copy parameter
model.conv1.W.data = original_model.conv1.W.data
model.conv1.b.data = original_model.conv1.b.data
model.conv2.W.data = original_model.conv2.W.data
model.conv2.b.data = original_model.conv2.b.data
model.conv3.W.data = original_model.conv3.W.data
model.conv3.b.data = original_model.conv3.b.data
model.conv4.W.data = original_model.conv4.W.data
model.conv4.b.data = original_model.conv4.b.data
model.conv5.W.data = original_model.conv5.W.data
model.conv5.b.data = original_model.conv5.b.data
#model.fc6.W.data = original_model.fc6.W.data
#model.fc6.b.data = original_model.fc6.b.data
#model.fc7.W.data = original_model.fc7.W.data
#model.fc7.b.data = original_model.fc7.b.data

del original_model
gc.collect()

model.to_gpu()

#make train dataset
im_size=127
num_label=260
x_train=[]

f=open("corel5k_train_list.txt")
line=f.read()
f.close()
line=line.split('\n')
for i in xrange(len(line)):
    line[i]=line[i]+'.jpeg'

for i in xrange(4500):
    filepath=line[i]
    if i==0:
        x_train=[np.array(Image.open(filepath).resize((im_size,im_size)))]
    else:
        if np.array(Image.open(filepath).resize((im_size,im_size))).shape ==(im_size, im_size, 3):    
            x_train.append(np.array(Image.open(filepath).resize((im_size,im_size)))) 
x_train=np.array(x_train)
x_train=x_train.astype(np.int32)/255.0
x_train=np.transpose(x_train,(0,3,1,2))
x_train=x_train.astype(np.float32)
print x_train.shape

y_train=np.load('y_train.npy')
y_train=y_train.astype(np.int32)

N=len(x_train)

#make test dataset
x_test=[]

f=open("corel5k_test_list.txt")
line=f.read()
f.close()
line=line.split('\n')
for i in xrange(len(line)-1):
    line[i]=line[i]+'.jpeg'

for i in xrange(len(line)-1):
    filepath=line[i]
    if i==0:
        x_test=[np.array(Image.open(filepath).resize((im_size,im_size)))]
    else:
        if np.array(Image.open(filepath).resize((im_size,im_size))).shape ==(im_size, im_size, 3):
            x_test.append(np.array(Image.open(filepath).resize((im_size,im_size))))
x_test=np.array(x_test)
x_test=x_test.astype(np.int32)/255.0
x_test=np.transpose(x_test,(0,3,1,2))
x_test=x_test.astype(np.float32)
print x_test.shape

y_test=np.load('y_test.npy')
y_test=y_test.astype(np.int32)
print y_test.shape

N_test=len(x_test)

#define forward process
def forward(x_data, y_data,L=L,batchsize=batchsize):
    L=Variable(cuda.to_gpu(L))
    x, t = Variable(cuda.to_gpu(x_data)), Variable(cuda.to_gpu(y_data))
#    x, t = Variable(x_data), Variable(y_data)
    h=F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv1(x))),3,stride=2)
    h=F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv2(h))),3,stride=2)
    h=F.relu(model.conv3(h))
    h=F.relu(model.conv4(h))
    h=F.max_pooling_2d(F.relu(model.conv5(h)),3,stride=2)
    h=F.relu(model.fc6(h))
    y = model.fc7(h)

    #y_f=F.sigmoid(y)
    #y_f=F.softmax(y)
    y_f = y
    y_ft=F.expand_dims(y_f,2)
#    print F.batch_matmul(y_f,L,transa=True).data.shape
#    print F.batch_matmul(F.batch_matmul(y_f,L,transa=True),y_ft).data.shape
    term=(F.sum(F.batch_matmul(F.batch_matmul(y_f,L,transa=True),y_ft)))/batchsize

    sce= F.sigmoid_cross_entropy(y, t)
    E = sce + (rate * term)   
    #E=sce   
    return E,sce,term,y_f

def cul_acc(x_data,y_data,threshold=0.500):
    output=np.array(x_data).astype(np.float32)

    output=output.reshape(len(y_data)*260,)
    target=y_data.reshape(len(y_data)*260,)
    output[np.where(output>threshold)[0]]=1
    output[np.where(output<threshold)[0]]=0
    correct=np.count_nonzero(output==target)
    return float(correct)/(len(y_data)*260)

#setup optimizer
optimizer = optimizers.Adam(alpha=0.0001)
#optimizer = optimizers.SGD(lr=0.1)
#optimizer.setup(model.collect_parameters())
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.001))

for epoch in xrange(1, n_epoch+1):
    print 'epoch', epoch

    #training
    perm = np.random.permutation(N)
    sum_loss = 0
    sum_sce=0
    sum_term=0
    loss_val=[]
    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
    	y_batch = y_train[perm[i:i+batchsize]]

        optimizer.zero_grads()
        loss,sce,GFHF,pred = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()
        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
        sum_sce     += float(cuda.to_cpu(sce.data)) * batchsize
        sum_term     += float(cuda.to_cpu(GFHF.data)) * batchsize
        num_train = i + batchsize
	if (num_train%500)==0:
	    print 'num_train=',num_train, 'loss=', sum_loss/num_train
    
    print 'train loss={},sce={},term={}'.format(sum_loss / N,sum_sce/N,sum_term/N)

    #test for training dataset
    sum_loss = 0
    sum_sce=0
    sum_term=0
    sum_acc=0
    loss_val=[]
    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
    	y_batch = y_train[perm[i:i+batchsize]]

        loss,sce,GFHF,pred = forward(x_batch, y_batch)
        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
        sum_sce     += float(cuda.to_cpu(sce.data)) * batchsize
        sum_term     += float(cuda.to_cpu(GFHF.data)) * batchsize
        pred=cuda.to_cpu(pred.data)
#        print np.max(pred)
	acc=cul_acc(pred,y_batch)
	sum_acc+=acc*batchsize
    print 'test train loss={},sce={},term={},acc={}'.format(sum_loss / N,sum_sce/N,sum_term/N,sum_acc/N)
#    loss_val.append((sum_sce+sum_term)/N)
    loss_val.append(sum_loss/N)
    loss_val.append(sum_sce/N)
    loss_val.append(sum_term/N)
    loss_val.append((1.0-(sum_acc/N)))
    with open('./loss_data/adam/eigenvector/w2v_train_%.10f_%.2f_%d.csv'%(rate,weight_p,dim),'ab') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(loss_val)
    #test for training dataset
    sum_loss = 0
    sum_sce=0
    sum_term=0
    sum_acc=0
    loss_val=[]
    for i in xrange(0, N_test):
        x_batch = x_test[i:i+1]
    	y_batch = y_test[i:i+1]

        loss,sce,GFHF,pred = forward(x_batch, y_batch,L=L_origin,batchsize=1)
        sum_loss     += float(cuda.to_cpu(loss.data)) 
        sum_sce     += float(cuda.to_cpu(sce.data)) 
        sum_term     += float(cuda.to_cpu(GFHF.data))
	pred=cuda.to_cpu(pred.data)
#        print np.max(pred)
	acc=cul_acc(pred,y_batch)
	sum_acc+=acc
    print 'test test loss={},sce={},term={},acc={}'.format(sum_loss / N_test,sum_sce/N_test,sum_term/N_test,sum_acc/N_test)
#    loss_val.append((sum_sce+sum_term)/N)
    loss_val.append(sum_loss/N)
    loss_val.append(sum_sce/N_test)
    loss_val.append(sum_term/N_test)
    loss_val.append(1.0-(sum_acc/N_test))
    with open('./loss_data/adam/eigenvector/w2v_test_%.10f_%.2f_%d.csv'%(rate,weight_p,dim),'ab') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(loss_val)
model.to_cpu()

#serializers.save_npz('./model/sim_w2v_model_%.10f_%.2f_%d.model'%(rate,weight_p,dim),model)
serializers.save_npz('./model/hw2v_model_%.10f_%.2f_%d.model'%(rate,weight_p,dim),model)
