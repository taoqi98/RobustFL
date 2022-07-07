import os
import numpy as np
import struct
from keras.utils.np_utils import to_categorical


def load_data(mode,data_root_path):
    if mode == 'train':
        file_path = os.path.join(data_root_path,'train-images.idx3-ubyte')
        label_path = os.path.join(data_root_path,'train-labels.idx1-ubyte')
    else:
        file_path = os.path.join(data_root_path,'t10k-images.idx3-ubyte')
        label_path = os.path.join(data_root_path,'t10k-labels.idx1-ubyte')
        
    binfile = open(file_path, 'rb') 
    buffers = binfile.read()
    magic,num,rows,cols = struct.unpack_from('>IIII',buffers, 0)
    bits = num * rows * cols
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    images = np.reshape(images, [num, rows * cols])
    
    images = images.reshape((len(images),28,28,1))
    
    binfile = open(label_path, 'rb')
    buffers = binfile.read()
    magic,num = struct.unpack_from('>II', buffers, 0) 
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    binfile.close()
    labels = np.reshape(labels, [num])
    
    images = images/255
    labels = to_categorical(labels)
    
    if  mode == 'train':
        #random_index = np.random.permutation(len(images))
        random_index = np.load('random_index.npy')
        N = int(0.2*len(images))
        unlabeled_images = images[random_index][:N]
        images = images[random_index][N:]
        labels = labels[random_index][N:]
    else:
        unlabeled_images = None
    
    return unlabeled_images,images,labels


  
def local_data_partition(train_labels,alpha=5,local_size=400):
    range_length = local_size
    CLASS_NUM = train_labels.shape[1]

    category_dict = {}
    category_used_data = {}
    train_users_dict = {}

    for cid in range(CLASS_NUM):
        category_used_data[cid] = 0
        flag = np.where(train_labels.argmax(axis=-1) == cid)[0]
        perb = np.random.permutation(len(flag))
        flag = flag[perb]
        category_dict[cid] = flag

    user_num = int(np.ceil(len(train_labels)//range_length))
    for uid in range(user_num-1):
        p = np.random.dirichlet([alpha]*CLASS_NUM, size=1)*range_length
        p = np.array(np.round(p),dtype='int32')[0]
        ix = p.argmax()
        p[ix] = p[ix] - (p.sum()-range_length)
        assert p.sum() == range_length and (p>=0).mean() == 1.0

        data = []
        for cid in range(CLASS_NUM):
            s = category_used_data[cid]
            ed = s + p[cid]
            category_used_data[cid] += p[cid]

            data.append(category_dict[cid][s:ed])
        
        data = np.concatenate(data).tolist()
        if len(data)<local_size:
            for cid in range(len(category_used_data)):
                left = local_size-len(data)
                if category_used_data[cid] < category_dict[cid].shape[0]:
                    s = category_used_data[cid]
                    ed = s + left
                    category_used_data[cid] += left
                    data += category_dict[cid][s:ed].tolist()
                left = local_size-len(data) 
                if left ==0:
                    break
                
        data = np.array(data)
        train_users_dict[uid] = data

    data = []
    for cid in range(CLASS_NUM):
        s = category_used_data[cid]        
        data.append(category_dict[cid][s:])
    data = np.concatenate(data)
    train_users_dict[user_num-1] = data

    train_users = train_users_dict
    
    return train_users


class TriggerGenerator:
    def __init__(self,size,x,y,pix,label,C):
        self.trigger = np.zeros((size,size))+pix
        self.trigger = self.trigger/255
        
        self.size = size
        self.x = x
        self.y = y
        self.pix = pix
        self.label = to_categorical(label,num_classes=C)
        
    def insert_trigger(self,data):
        data2 = np.copy(data)
        data2[self.x:self.x+self.size,self.y:self.y+self.size,0] = self.trigger
        return data2,self.label

def generate_malicious_clients(train_users,train_images,train_labels,trigger):
    user_index = np.random.permutation(len(train_users))
    
    r = 0.3
    N = int(r*len(user_index))
    
    all_trigger_images = []
    all_trigger_labels = []
    start = len(train_labels)

    malicious_clients = set(user_index[:N].tolist())

    for uix in malicious_clients:
        data_index = train_users[uix]

        local_trigger_images = []
        local_trigger_labels = []

        for j in range(len(data_index)):
            inx = data_index[j]
            data = trigger.insert_trigger(train_images[inx])
            local_trigger_images.append(data[0])
            local_trigger_labels.append(data[1])

        local_trigger_images = np.array(local_trigger_images)
        local_trigger_labels = np.array(local_trigger_labels)

        all_trigger_images.append(local_trigger_images)
        all_trigger_labels.append(local_trigger_labels)


        ed = start + len(local_trigger_images)
        local_trigger_index = np.array([i for i in range(start,ed)])
        start = ed
        data_index = np.concatenate([data_index,local_trigger_index],axis=0)
        shuffled_data_index = np.random.permutation(len(data_index))
        data_index = data_index[shuffled_data_index]
        train_users[uix] = data_index

    all_trigger_images = np.concatenate(all_trigger_images,axis=0)
    all_trigger_labels = np.concatenate(all_trigger_labels,axis=0)

    train_images = np.concatenate([train_images,all_trigger_images],axis=0)
    train_labels = np.concatenate([train_labels,all_trigger_labels],axis=0)
    
    return train_images,train_labels,malicious_clients

def generate_poisoned_test_data(test_images,test_labels,trigger):
    
    index = np.random.permutation(len(test_images))

    poisoned_test_images = []
    poisoned_test_labels = []

    for i in range(len(test_images)):
        data = trigger.insert_trigger(test_images[i])
        if test_labels[i].argmax() == trigger.label.argmax():
            continue
        poisoned_test_images.append(data[0])
        poisoned_test_labels.append(data[1])

    poisoned_test_images = np.array(poisoned_test_images)
    poisoned_test_labels = np.array(poisoned_test_labels)
    
    return poisoned_test_images,poisoned_test_labels