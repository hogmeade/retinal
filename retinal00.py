'''
Update:  train randomly from beginning, calculate accuracy vs batches
Author: Huong N. Pham
Classification problem: Dog vs Cat (25000 samples, 4k for valuation, 4k for testing)

'''
from pyLib.sendEmail import send_email
from pyLib.stopInstance import stop_instance
import os
import time
import pickle
import shutil
import argparse
import numpy as np
from random import randrange

import copy
import torch
import torchvision


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets.folder import ImageFolder

#################################################################
# Default parameters
'''

'''
#################################################################
class ImageFolderWithIDs(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithIDs, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (index,))
        return tuple_with_path
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ , _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    print('Mean: %3f'%(mean))
    print('STD: %3f'%(std))
    return mean, std
def tensorRemoveElementByIndices(tensorR, indices):
    print
    for idx, val in enumerate(sorted(indices)):
      tensorR = torch.cat([tensorR[:(val-idx)], tensorR[(val-idx)+1:]])
    return tensorR
def findIndicesByIDs(tensorIDs, IDs):
    indices = []
    for idx in IDs:
      indices.append((tensorIDs == idx).nonzero(as_tuple=True)[0])
    return np.unique(np.array(torch.cat(indices).tolist()))
def imageStat(directory):
    # directory = "/content/drive/My Drive/Colab Notebooks/Kaggle/DME/train/"
    data = ImageFolderWithIDs(root=directory, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset=data, batch_size=16)
    mean, std = get_mean_and_std(dataloader)
    return mean, std
#"/content/drive/My Drive/Colab Notebooks/Kaggle/DME/val/"
def load_images(directory):
    from torchvision import transforms
    transforms = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )])
    data = ImageFolderWithIDs(root=directory, transform=transforms)
#    test_data_path = "/content/drive/My Drive/Colab Notebooks/Kaggle/DogCat/sample/test/"
#    test_data = ImageFolderWithIDs(root=test_data_path, transform=transforms)
    return data
def listTensor(data_loader):
    listImages = []
    listLabels = []
    listIDs  = []
    start = time.time()
    for idx, data in enumerate(data_loader):
        images, labels, ids = data
        listImages.append(images)
        listLabels.append(labels)
        listIDs.append(ids)
    listImagesFlatten = torch.cat(listImages)
    listLabelsFlatten = torch.cat(listLabels)
    listIDsFlatten    = torch.cat(listIDs)
    done = time.time()
    print("Loading time: {} seconds".format(done - start))
    return listImagesFlatten, listLabelsFlatten, listIDsFlatten

def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='DME classification')
 
    parser.add_argument('-probsThresthold', type=float, default=0.8, help="probability threshold min to remove out of training ")
    parser.add_argument('-train_data_path', type=str, default='/home/hpham/Data/DME/train', help='train directory')
    parser.add_argument('-val_data_path', type=str, default='/home/hpham/Data/DME/val', help='valuation directory')
    parser.add_argument('-test_data_path', type=str, default='/home/hpham/Data/DME/test', help='valuation directory')
    parser.add_argument('-batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-batchNumber', type=int, default=20, help='number of batches to keep training')
    parser.add_argument('-epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('-version', type=int, default=0, help='version')
    parser.add_argument('-terminate','-t', action='count', default=0, help="stop instance")
    return parser
        
def extract_data(args):
    '''
    Translate Image data structure into a data set for training/evaluating a single model
    
    @param args Argparse object, which contains key information, including train_datapath, 
            val_data_path, test_data_path
            
    @return Tensor for training set input/output, 
            validation set input/output and testing set input/output; and a
            dictionary containing the lists of data paths that have been chosen
    '''
    from torchvision import transforms
    # Load data from tensors or images and transforms data to tensor with IDs atttached
    if args.train_data_path[-3:] == '.pl':
      train_dataset = torch.load(args.train_data_path)
    else:
      train_dataset = load_images(args.train_data_path)
      
    if args.val_data_path[-3:] == '.pl':
      val_dataset   = torch.load(args.val_data_path)
    else:
      val_dataset   = load_images(args.val_data_path)

    if args.test_data_path[-3:] == '.pl':
      test_dataset   = torch.load(args.test_data_path)
    else:
      test_dataset   = load_images(args.test_data_path)

    """test_data_path = "/content/drive/My Drive/Colab Notebooks/Kaggle/DogCat/sample/test/"
    test_dataset = ImageFolderWithIDs(root=test_data_path, transform=transforms)"""

    return train_dataset, val_dataset, test_dataset

def generate_fname(args):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.
    '''
    if args.probsThresthold is None:
        probsThresthold_str = ''
    else:
        probsThresthold_str = 'probsThresthold_%0.2f_'%(args.probsThresthold)

    if args.batch_size is None:
        batch_size_str = ''
    else:
        batch_size_str = 'batch_size_%d_'%(args.batch_size)

    if args.batchNumber is None:
        batchNumber_str = ''
    else:
        batchNumber_str = 'batchNumber_%d_'%(args.batchNumber)

    if args.epochs is None:
        epochs_str = ''
    else:
        epochs_str = 'epochs_%d_'%(args.epochs)



    if args.version is None:
        version_str = ''
    else:
        version_str = 'ver_%d_'%(args.version)

    # Put it all together, including #of training folds and the experiment rotation
    return "%s%s%s%s%s"%(
                      probsThresthold_str, 
                      
                      batch_size_str, batchNumber_str, epochs_str, version_str)
    
def train(args, model, optimizer, loss_fn, train_data, val_data, test_data):
    # create file name
    fbase = generate_fname(args)
    ranID = randrange(100000)
    
    #check if gpu is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # weight to GPU
    if torch.cuda.is_available():
        model.cuda()
    #weight sampler
    weights = np.ones(len(train_data))

    #train_loader = DataLoader(train_data, batch_size=args.batch_size)
    val_loader   = DataLoader(val_data, batch_size=args.batch_size)
    test_loader  = DataLoader(test_data, batch_size=args.batch_size)

    listRemove = []
    
    training_loss_log = []
    validation_loss_log = []
    accuracy_log = []

    best_acc1 = 0
    accuracy = 0
    
    results = {}
    
    for epoch in range(int(args.epochs)):
        batchNumber = 0
        for i in range(0,args.batchNumber):
            training_loss = 0.0
            valid_loss = 0.0
            model.train()
            
            sampler = WeightedRandomSampler(weights, args.batch_size)
            train_loader = DataLoader(train_data, shuffle=(sampler is None),sampler=sampler, batch_size = args.batch_size)
            
            batch = iter(train_loader).next()
            batchNumber += 1

            optimizer.zero_grad()
            inputs, targets, ids = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs) 
            print("######################## Epoch {} - Batch {} ########################".format(epoch, batchNumber))
            print("IDs in batch {}: {}".format(batchNumber,ids))
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
        training_loss /= batchNumber
        # Validation
        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets, ids = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(outputs,targets)
            valid_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(outputs, dim = 1), dim=1)[1],
            targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]

            #writer.add_scalar('accuracy', num_correct / num_examples, epoch)
            
        valid_loss /= len(val_loader)
        accuracy = num_correct / num_examples

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss, valid_loss, accuracy))

        if accuracy > best_acc1 and accuracy >= 0.85:
            best_acc1 = accuracy
            best_model_state1 = copy.deepcopy(model.state_dict())
            print("Save best Model_1 @ epoch {} acc: {}".format(epoch, best_acc1))
            
            epochState1 = epoch
            best_valid_loss = valid_loss
            best_optimizer = optimizer.state_dict()
            # Save model when achieve the best
            torch.save({
                'model_state1_dict': best_model_state1,
                'optimizer'        : best_optimizer,
                'valid_loss_min'   : best_valid_loss,
                'accuracy'         : best_acc1,
                'epoch'            : epochState1
                }, "%smodel"%(fbase))
        if epoch == 2998:
            model_2998 = copy.deepcopy(model.state_dict())
            torch.save({'model_2998': model_2998}, "model_2998")
        # Performance log data
        training_loss_log.append(round(training_loss,2))
        validation_loss_log.append(round(valid_loss,2))
        accuracy_log.append(round(accuracy,2))

        if accuracy >= args.probsThresthold:
          break

    # Generate log data
    
    """results['args'] = args"""

    results['training_loss_log'] = training_loss_log
    results['validation_loss_log'] = validation_loss_log
    results['accuracy_log'] = accuracy_log

    
    results['best_acc'] = [best_acc1]
    results['epochState'] = [epochState1]
    
    # Save log files
    
    print("Saved file as %s_%s_%s.pkl"%(os.path.basename(__file__)[:-3],fbase,ranID))
    results['fname_base'] = fbase
    fp = open("%s_%s_%s.pkl"%(os.path.basename(__file__)[:-3],fbase,ranID), "wb")
    pickle.dump(results, fp)
    fp.close()
    
    # Show results
    print("Validation accuracy state 1: {} @ epoch {} ".format(best_acc1, epochState1))
    print(accuracy_log)
   

    send_email("{}\nMax validation accuracy: {} @ epoch {} pytorch GPU".format(fbase,best_acc1,epochState1))
    if (args.terminate>=2):
        stop_instance('inlaid-fuze-338203','us-central1-a','pytorch-gpu')
def execute_exp(args=None):
    '''
    Perform the training and evaluation for a single model
    
    @args Argparse arguments
    '''
    # Check the arguments
    if args is None:
      # Case where no args are given (usually, because we are calling from within Jupyter)
      #  In this situation, we just use the default arguments
      parser = create_parser()
      args = parser.parse_args([])
    
    # Extract the data sets
    train_data, val_data, test_data = extract_data(args)
    
    # Load model
    transfer_model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
    optimizer = optim.Adam(transfer_model.parameters(), lr=0.001)

    # Freeze parameters
    for name, param in transfer_model.named_parameters():
      if("bn" not in name):
        param.requires_grad = False

    # Replace last layer
    transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features,500), nn.ReLU(), nn.Dropout(), nn.Linear(500,3))
    
    # Tune the model
    train(args, transfer_model, optimizer, torch.nn.CrossEntropyLoss(), train_data, val_data, test_data)

    # Report if verbosity is turned on
    """if args.verbose >= 1:
        print(model.summary())"""

#################################################################
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    execute_exp(args)
# (nohup python3 retinal00.py -probsThresthold 0.98 -batchNumber 1 -batch_size 16 -epochs 4000 -train_data_path '/home/huong_n_pham01/data/2t_retinal/train1' -val_data_path '/home/huong_n_pham01/data/2t_retinal/val1' -test_data_path '/home/huong_n_pham01/data/2t_retinal/test' > retinalNoweightTrain1.log ; nohup python3 retinal00.py -probsThresthold 0.99 -batchNumber 1 -batch_size 16 -epochs 4000 -train_data_path '/home/huong_n_pham01/data/2t_retinal/train2' -val_data_path '/home/huong_n_pham01/data/2t_retinal/val2' -test_data_path '/home/huong_n_pham01/data/2t_retinal/test' > retinalNoweightTrain2.log) &