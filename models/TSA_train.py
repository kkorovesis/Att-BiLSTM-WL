import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd
import numpy as np
from load_embeddings import load_word_vectors
from datasets import SentimentDataset
from models import RNN
from tools import sort_batch, split_train_set, eval_dataset, progress, class_weigths, loss_curve, acc_curve, f1_curve, \
    save_model
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def train_epoch(_epoch, dataloader, model, loss_function):
    # switch to train mode -> enable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0

    for i_batch, sample_batched in enumerate(dataloader, 1):

        # get the inputs (batch)
        inputs, topics, labels, lengths, topic_lengths, weights, indices = sample_batched

        # sort batch (for handling inputs of variable length)
        lengths, (inputs, labels, topics) = sort_batch(lengths, (inputs, labels, topics))

        # convert to CUDA Variables
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            topics = Variable(topics.cuda())
            labels = Variable(labels.cuda())
            lengths = Variable(lengths.cuda())
            topic_lengths = Variable(topic_lengths.cuda())
        else:
            inputs = Variable(inputs)
            topics = Variable(topics)
            labels = Variable(labels)
            topic_lengths = Variable(topic_lengths)

        # 1 - zero the gradients
        optimizer.zero_grad()

        # 2 - forward pass: compute predicted y by passing x to the model
        outputs = model(inputs, topics, lengths, topic_lengths)

        # 3 - compute loss
        loss = loss_function(outputs, labels)

        # 4 - backward pass: compute gradient wrt model parameters
        loss.backward()

        # 5 - update weights
        optimizer.step()

        running_loss += loss.data[0]

        # print statistics
        progress(loss=loss.data[0],
                 epoch=_epoch,
                 batch=i_batch,
                 batch_size=BATCH_SIZE,
                 dataset_size=len(loader_train.sampler.indices))

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        try:
            for p in model.parameters():
                if p.grad is not None:
                    p.data.add_(-lr, p.grad.data)

        except ValueError:
            print("'NoneType' object has no attribute 'data'")

    return running_loss / i_batch


# download from http://nlp.stanford.edu/data/
WORD_VECTORS = "../embeddings/glove.twitter.27B.200d.txt"

TRAIN_DATA = "datasets/test"

WORD_VECTORS_DIMS = 200
BATCH_SIZE = 128
EPOCHS = 30
seed = 1111
lr = 1e-3
clip = 1

_hparams = {
    "rnn_hidden_size": 150,
    "num_rnn_layers": 2,
    "bidirectional": True,
    "noise": 0.5,
    "dropout_embeds": 0.5,
    "dropout_rnn": 0.5,
}

# keep the same seed for random variables
np.random.seed(seed)
torch.manual_seed(seed)
# torch.backends.cudnn.enabled = False
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

########################################################
# PREPARE FOR DATA
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(WORD_VECTORS,
                                                   WORD_VECTORS_DIMS)

tword2idx, idx2tword, topic_embeddings = load_word_vectors(WORD_VECTORS,
                                                           WORD_VECTORS_DIMS)

train_set = SentimentDataset(file=TRAIN_DATA, word2idx=word2idx, tword2idx=tword2idx,
                             max_length=0, max_topic_length=0, topic_bs=True)

print("Batching...")

train_sampler, validation_sampler = split_train_set(train_set, contiguous=True, split_rate=0.1)

loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler,
                          shuffle=False, num_workers=4)

loader_val = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=validation_sampler,
                        shuffle=False, num_workers=4)

print("Train Size: ", len(loader_train.sampler.indices))
print("Validation Size: ", len(loader_val.sampler.indices))

num_classes = len(train_set.label_encoder.classes_)
model = RNN(embeddings, num_classes=num_classes, **_hparams)

weights = class_weigths(train_set.labels)

if torch.cuda.is_available():
    model.cuda()
    weights = weights.cuda()

criterion = torch.nn.CrossEntropyLoss(weight=weights)

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=lr)

#############################################################
# Train
#############################################################

best_val_loss = None

colms_l = ['Train_Loss', 'Val_Loss']
colms_acc = ['Train_Acc', 'Val_Acc']
colms_f1 = ['Train_F1', 'Val_F1']

df_l = pd.DataFrame(columns=colms_l, index=range(1, EPOCHS + 1))
df_acc = pd.DataFrame(columns=colms_acc, index=range(1, EPOCHS + 1))
df_f1 = pd.DataFrame(columns=colms_f1, index=range(1, EPOCHS + 1))

print("Number of classes: " + str(num_classes))
print("Training...")
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_epoch(epoch, loader_train, model, criterion)

    # evaluate the performance of the model, on both data sets
    avg_train_loss, (y, y_pred) = eval_dataset(loader_train, model, criterion)
    print("\tTrain: loss={:.4f}, acc={:.4f}, f1={:.4f}, p={:.4f}, r={:.4f}".

          format(avg_train_loss,
                 accuracy_score(y, y_pred),
                 f1_score(y, y_pred, average="macro"),
                 precision_score(y, y_pred, average="micro"),
                 recall_score(y, y_pred, average='macro')))

    df_l.loc[epoch].Train_Loss = avg_train_loss
    df_acc.loc[epoch].Train_Acc = accuracy_score(y, y_pred)
    df_f1.loc[epoch].Train_F1 = f1_score(y, y_pred, average="macro")

    avg_val_loss, (y, y_pred) = eval_dataset(loader_val, model, criterion)
    print("\tEval:  loss={:.4f}, acc={:.4f}, f1={:.4f}, p={:.4f}, r={:.4f}".

          format(avg_val_loss,
                 accuracy_score(y, y_pred),
                 f1_score(y, y_pred, average="macro"),
                 precision_score(y, y_pred, average="micro"),
                 recall_score(y, y_pred, average='macro')))

    df_l.loc[epoch].Val_Loss = avg_val_loss
    df_acc.loc[epoch].Val_Acc = accuracy_score(y, y_pred)
    df_f1.loc[epoch].Val_F1 = f1_score(y, y_pred, average="macro")


# Print curves
acc_curve(df_acc, EPOCHS)
loss_curve(df_l, EPOCHS)
f1_curve(df_f1, EPOCHS)

#############################################################
# Save Model
#############################################################

save_model(model, 'TSA_model.pt')
