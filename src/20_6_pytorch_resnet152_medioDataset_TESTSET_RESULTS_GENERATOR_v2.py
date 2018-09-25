#  # Developer: Vajira Thambawita
#  # Last modified date: 18/07/2018
#  # ##################################

#  # Description ##################
#  # pythroch resnet18 training






###########################################

from __future__ import print_function, division

import datetime

start = datetime.datetime.now()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
#  import numpy as np
#  import torchvision
from torchvision import datasets, models, transforms, utils

import pickle
from pandas_ml import ConfusionMatrix

import matplotlib as mpl
#  ######################################################
#  #### Matplotlib X display error - removing for server#
#  ######################################################
mpl.use('Agg')  # This has to run before pyplot import

import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import pandas as pd
import numpy as np

import sklearn.metrics as mtc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools


plt.ion()   # interactive mode

###################################################################
#  Getting main data directory
###############################################################

arg_main_data_dir = sys.argv[1]  # Main data directory to be handled
arg_model_name = sys.argv[2] # model name to be saves
arg_check_point_name_format = sys.argv[3]
arg_mode = sys.argv[4]  # Mode of the runing - Train, Test or Retrain from the best weight file

my_file_name = arg_model_name #"8_1_pytorch_resnet18_v1"  # model name to be saved

#####################################################################

# This is an additional test - Not necessary one

if not arg_mode in ["train", "test", "re-train"]:
    print("Invalid mode type")
    #exit(0)

print("Mode of runing: ", arg_mode)
#exit()

###############################################################

#  Set parameters here

data_dir = arg_main_data_dir
model_dir = data_dir + '/pytorch_models'
plot_dir  = data_dir + '/pytorch_plots'
history_dir = data_dir + '/pytorch_history'
submission_dir = data_dir + '/pytorch_submission_medico'



model_name = my_file_name # take my file name as the model name
checkpoint_name_format = arg_check_point_name_format # "13_1_weights-improvement-{epoch:02d}-{val_acc:.4f}.hdf5"
# checkpoint_name = ""

acc_loss_plot_name = 'acc_loss_plot_' + model_name
accuracy_plot_name = 'accuracy_plot_' + model_name
loss_plot_name = 'loss_plot_' + model_name
cm_plot_name = 'cm_'+model_name



batch_size = 1

########################################################################
#  Managin Directory structure
########################################################################
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

if not os.path.exists(history_dir):
    os.mkdir(history_dir)

if not os.path.exists(submission_dir):
    os.mkdir(submission_dir)

print("1 - Folder structure created")
#####################################################################

#  Preparing Data - Training and Validation + testing
if arg_mode in ["train", "re-train"]:

    number_of_epochs = int(input("Number of epochs to run:")) # number of epochs to train or re-train

    print("Preparing train data loaders")
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'validation']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=8)
                   for x in ['train', 'validation']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    print("Finished train data preparing..")

    # testing purpose only
    weights = torch.tensor([[1/123, 1/186, 1/319, 1/291,
               1/310, 1/50, 1/291, 1/307,
               1/305, 1/70, 1/429, 1/165,
               1/278, 1/91, 1/256, 1/319]])

   # print(image_datasets["train"].classes)
   # exit()


##################################################
#  Preparing test data
#################################################

elif arg_mode == "test":

    print('Preparing test data...')

    test_data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_datasets = datasets.ImageFolder(os.path.join(data_dir, 'test'), test_data_transforms)

    # This is only for taking class names to get correct class names for the output
    validation_datasets_to_get_classes = datasets.ImageFolder(os.path.join(data_dir, 'validation'), test_data_transforms)

    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=8)

    print('Preparing test data finised')

    test_image_names = [path[0][-12:] for path in test_datasets.imgs]

    print(test_image_names)

#exit() # temporary exit

###########################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("2 - Device set to :", device)
#########################################################################

#########################################################################
#  Printing images just for testing
#########################################################################
'''
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(dataloaders['train'])
sample_images, sample_labels = dataiter.next()



npimg = sample_images[0].numpy()

npimg = np.transpose(npimg,(1,2,0))



plt.imshow(npimg[:,:, 0])
plt.show()
print(npimg[:, :, 0])
#imshow(utils.make_grid(sample_images))
input()
exit()
'''

#######################################################################


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    if arg_mode == "re-train":
        best_acc = float(best_weight_file_name[-11:-5])
    elif arg_mode == "train":
        best_acc = 0.0

    history_tensor = torch.empty((num_epochs, 4), device=device)  # 4- trai_acc, train_loss, val_acc, val_loss
    checkpoint_name = None
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)



        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            indicator = 0  # just for print batch processing status (no of batches)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:



                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                  #  print("outputs=", outputs) # only for testing - vajira
                  #  print("labels = ", labels) # only for testing - vajira
                    print(indicator, sep='-', end='=', flush=True)
                    indicator += 1

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Collecting data for making plots
            if phase == 'train':
                history_tensor[epoch, 0] = epoch_acc
                history_tensor[epoch, 1] = epoch_loss
            if phase == 'validation':
                history_tensor[epoch, 2] = epoch_acc
                history_tensor[epoch, 3] = epoch_loss

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                checkpoint_name = checkpoint_name_format.format(epoch=epoch, val_acc=best_acc)
                print("Found a best model:", checkpoint_name)
            elif phase== 'validation':
                print("No improvement from the previous best model ")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history_tensor, checkpoint_name

##########################################################
#  Model testing method
##########################################################


def test_model(test_model,test_dataloader):
    print("Testing started..")
    test_model.eval()
    correct = 0
    total = 0
    all_labels_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_probabilities_d = torch.tensor([], dtype=torch.float).to(device)

    if batch_size == 1:
        all_timePerFrame_host = []

    else:
        print("Please set batch size to 1....")
        exit(0)

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            frame_time_start = datetime.datetime.now() # frame start time

            outputs = test_model(inputs)
            outputs = F.softmax(outputs, 1)
            print(outputs)
            predicted_probability, predicted = torch.max(outputs.data, 1)

            frame_time_end = datetime.datetime.now() # frame end time

            time_per_image = (frame_time_end - frame_time_start).total_seconds()
            #print((predicted == labels).sum())
            total += labels.size(0)
            correct += (predicted == labels).sum()
            all_labels_d = torch.cat((all_labels_d, labels), 0)
            all_predictions_d = torch.cat((all_predictions_d, predicted), 0)
            all_predictions_probabilities_d = torch.cat((all_predictions_probabilities_d, predicted_probability), 0)
            all_timePerFrame_host = all_timePerFrame_host + [time_per_image]

    print('copying some data back to cpu for generating confusion matrix...')
    y_true = all_labels_d.cpu()
    y_predicted = all_predictions_d.cpu()  # to('cpu')
    testset_predicted_probabilites = all_predictions_probabilities_d.cpu()  # to('cpu')


    return y_predicted, testset_predicted_probabilites, all_timePerFrame_host

'''
    cm = confusion_matrix(y_true, y_predicted)  # confusion matrix



    print('Accuracy of the network on the %d test images: %f %%' % (total, (
            100.0 * correct / total)))

    print(cm)

    print("taking class names to plot CM")

    class_names = test_datasets.classes  # taking class names for plotting confusion matrix

    print("Generating confution matrix")

    plot_confusion_matrix(cm, classes=class_names, title='my confusion matrix')

    print('confusion matrix saved to ', plot_dir)

    ##################################################################
    # classification report
    #################################################################
    print(classification_report(y_true, y_predicted, target_names=class_names))

    ##################################################################
    # Standard metrics for medico Task
    #################################################################
    print("Printing standard metric for medico task")

    print("Accuracy =",mtc.accuracy_score(y_true, y_predicted))
    print("Precision score =", mtc.precision_score(y_true,y_predicted, average="weighted"))
    print("Recall score =", mtc.recall_score(y_true, y_predicted, average="weighted"))
    print("F1 score =", mtc.f1_score(y_true, y_predicted, average="weighted"))
    print("Specificity =")
    print("MCC =", mtc.matthews_corrcoef(y_true, y_predicted))

    ##################################################################
    # Standard metrics for medico Task
    #################################################################
    print("Printing standard metric for medico task")

    weights = [1 / 53, 1 / 81, 1 / 138, 1 / 125,
               1 / 134, 1 / 11, 1 / 125, 1 / 132,
               1 / 132, 1 / 4, 1 / 184, 1 / 72,
               1 / 120, 1 / 39, 1 / 110, 1 / 138]

    print("1. Recall score (REC) =", mtc.recall_score(y_true, y_predicted, average="weighted"))
    print("2. Precision score (PREC) =",
          mtc.precision_score(y_true, y_predicted, average="weighted"))
    print("3. Specificity (SPEC) =")
    print("4. Accuracy (ACC) =", mtc.accuracy_score(y_true, y_predicted, weights))
    print("5. Matthews correlation coefficient(MCC) =", mtc.matthews_corrcoef(y_true, y_predicted))

    print("6. F1 score (F1) =", mtc.f1_score(y_true, y_predicted, average="weighted"))

    panda_cm_data = ConfusionMatrix(y_true, y_predicted)
    panda_cm_data.print_stats()
    cm_dictionary = panda_cm_data.stats()
    print("cm _ dictionary saving")
    f = open(os.path.join(history_dir, "20_5_cm_dictionary.pkl"), "wb")
    pickle.dump(cm_dictionary['class'], f)
    f.close()

    print('Finished.. ')

    print('Finished.. ')
'''




##########################################################
# Prepare submission file:
##########################################################

def prepare_submission_file(image_names, predicted_labels, max_probability, time_per_image, submit_dir, data_classes):

    predicted_label_names = []

    for i in predicted_labels:
        predicted_label_names = predicted_label_names + [data_classes[i]]

    print(predicted_label_names)

    submission_dataframe = pd.DataFrame(np.column_stack([image_names,
                                                         predicted_label_names,
                                                         max_probability,
                                                         time_per_image]),
                                   columns=['images', 'labels', 'PROB', 'time'])
    #print("image names:{0}".format(image_names))

    submission_dataframe.to_csv(os.path.join(submit_dir, "test1_PROB_softmax_time"), index=False)

    print(submission_dataframe)
    print("successfully created submission file")
###########################################################

###########################################################
#  Ploting history and save plots to plots directory
###########################################################
def plot_and_save_training_history(history_tensor, mode):
    history_data = history_tensor.cpu().numpy()
    df = pd.DataFrame(history_data, columns=['train_acc', 'train_loss', 'val_acc', 'val_loss'])

    # writing plotting data to csv file
    df.to_csv(os.path.join(history_dir, acc_loss_plot_name), sep='\t', encoding='utf-8')

    pie = df.plot()
    fig = pie.get_figure()
    if mode == "train":
        fig.savefig(os.path.join(plot_dir, "_training_" + acc_loss_plot_name))
    elif mode == "re-train":
        fig.savefig(os.path.join(plot_dir, "__re-train__" + acc_loss_plot_name))


############################################################
# Plot confusion matrix - method
############################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          plt_size=[10,10]):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.rcParams['figure.figsize'] = plt_size
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(plot_dir, cm_plot_name))
    print("Finished confusion matrix drawing...")


#############################################################
#  Loading a pretraind model and modifing the last layers

model_ft = models.resnet152(pretrained=True)

#print(model_ft)
#exit()
# num_ftrs = model_ft.fc.in_features
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 16)

#print(model_ft)
#exit() # Just for testing

print("3 - Model created")


## #######################################################
# If multiple GPUS are there, run on multiple GPUS
##########################################################
#  Setting model in multiple GPUs
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model_ft = nn.DataParallel(model_ft)
elif torch.cuda.device_count() == 1:
    print("Found only one GPU")
else:
    print("No GPU.. Runing on CPU")


print("4 - Setup data parallel devices")

##########################################################
# If the mode == re-training - load the best weight file
# If the mode == test - load the best weight file
##########################################################
if arg_mode in ["re-train", "test"]:
    best_weight_file_name = input('Please, enter the best weights value file name:')
    model_ft.load_state_dict(torch.load(os.path.join(model_dir, best_weight_file_name)))
    print('4-1 - Model loaded with the best weight file')



##############################################################
#  Loading model to GPUs
##############################################################


# Moving model to the GPu has to be done before setting parameters
# to the model. parameters of the model has different object type
# after moving the model to the GPU (According to the pytorch document)

model_ft = model_ft.to(device)

print("5 - Model loadded to device")


############################################################
#  Testing method will be called when mode == test
############################################################
if arg_mode == "test":
    predicted_labels, predicted_prb, time_per_image = test_model(model_ft, test_dataloader)  # this exit from the main program
    print("Testing completed..")
    end = datetime.datetime.now()
    total_time = end - start
    print("Elapsed Time=", total_time.total_seconds())
    print("Number of images in Test test=", len(test_datasets))
    print("Number of frames per second =", len(test_datasets)/total_time.total_seconds())

    print("preparing SUMISSION FILE")
    prepare_submission_file(test_image_names,
                            predicted_labels, predicted_prb , time_per_image = time_per_image,
                            data_classes= validation_datasets_to_get_classes.classes,
                            submit_dir=submission_dir )
    exit(0)



############################################################
# Setting model parameters
############################################################

weights = weights.to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# optimizer_ft = optim.RMSprop(model_ft.parameters(), lr= 0.1, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
mulStep_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft,
                                                milestones=[5, 10, 15, 20], gamma=0.5)

print("6 - Setup parameters, This has to be done after loading model to the device")
#############################################################
# Start Training
############################################################

model_ft, history_tensor, check_point_name = train_model(model_ft, criterion,
                                                         optimizer_ft, mulStep_lr_scheduler,
                                                         num_epochs=number_of_epochs)


print("7 - Training complered")
############################################################
# Save the model to the directory
############################################################

if not os.path.exists(model_dir):
    os.mkdir(model_dir)  # to save plots

if not check_point_name==None:
    print(check_point_name)
    model_ft.to("cpu")
    torch.save(model_ft.state_dict(), os.path.join(model_dir, check_point_name))
    print("8 -Model saved")
########################################################

plot_and_save_training_history(history_tensor, arg_mode)

print("9 - Plots saved to", plot_dir)


