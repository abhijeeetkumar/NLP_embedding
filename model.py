# -*- coding:utf-8 -*-
"""
This py file is a skeleton for the main project components.  
Places that you need to add or modify code are marked as TODO
"""

import torch
import torch.nn as nn
import torch.optim
import numpy as np


def word2index(word, vocab):
    """
    Convert a word token to a dictionary index
    """
    if word in vocab:
        value = vocab[word][0]
    else:
        value = -1
    return value


def index2word(index, vocab):
    """
    Convert a dictionary index to a word token
    """
    for w, v in vocab.items():
        if v[0] == index:
            return w
    return 0


class Model(object):
    def __init__(self, args, vocab, pos_data, neg_data):
        """The Text Classification constructor """
        self.embeddings_dict = {}
        self.algo = args.algo
        if self.algo == "GLOVE":
            print("Now we are using the GloVe embeddings")
            self.load_glove(args.emb_file)
        else:
            print("Now we are using the BOW representation")
        self.vocab = vocab
        self.pos_sentences = pos_data
        self.neg_sentences = neg_data
        self.lr = args.lr
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.dataset = []
        self.labels = []
        self.sentences = []

        self.train_data = []
        self.train_label = []

        self.valid_data = []
        self.valid_label = []

        '''
            # TODO
            You should modify the code for the baseline classifiers for self.algo
            shown below, which is a three layer model with an input layer, a hidden layer,
            and an output layer. You will need at least to define the dimensions for
            the size of the input layer (ISIZE; see where this is passed in by argparse),
            and the hidden layer (e.g., HSIZE).  Do not change the size of the output
            layer, which is currently 2.  This corresponds to the number of classes.
            You need to choose an activation function. Once you get this working
            by uncommenting these lines, adding the activation function, and replacing
            ISIZE and HSIZE, then see if you can achieve the minimum classification
            accuracy of 0.80 for either model.  You are free to modify this
            code, e.g., to add more hidden layers, or make other changes, to raise
            the accuracy.
        '''
        
        if self.algo == "GLOVE":
            self.model = nn.Sequential(
            nn.Linear(self.embed_size, 2*self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(2*self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, 2),
            nn.LogSoftmax(),)
        else:
            self.model = nn.Sequential(
            nn.Linear(len(self.vocab), 2*self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(2*self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(0.9),
            nn.Linear(self.hidden_size, 2),
            nn.LogSoftmax(), )


    def load_dataset(self):
        """
        Load the training and testing dataset
        """
        for sentence in self.pos_sentences:
            new_sentence = []
            for l in sentence:
                if l in self.vocab:
                    if self.algo == "GLOVE":
                        new_sentence.append(l)
                    else:
                        new_sentence.append(word2index(l, self.vocab))
            self.dataset.append(self.sentence2vec(new_sentence, self.vocab))
            self.labels.append(0)
            self.sentences.append(sentence)

        for sentence in self.neg_sentences:
            new_sentence = []
            for l in sentence:
                if l in self.vocab:
                    if self.algo == "GLOVE":
                        new_sentence.append(l)
                    else:
                        new_sentence.append(word2index(l, self.vocab))
            self.dataset.append(self.sentence2vec(new_sentence, self.vocab))
            self.labels.append(1)
            self.sentences.append(sentence)

        indices = np.random.permutation(len(self.dataset))

        self.dataset = [self.dataset[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.sentences = [self.sentences[i] for i in indices]

        # split dataset
        test_size = len(self.dataset) // 10
        self.train_data = self.dataset[2 * test_size:]
        self.train_label = self.labels[2 * test_size:]

        self.valid_data = self.dataset[:2 * test_size]
        self.valid_label = self.labels[:2 * test_size]

    def rightness(self, predictions, labels):
        """ 
        Prediction of the error rate
        """
        pred = torch.max(predictions.data, 1)[1]
        rights = pred.eq(labels.data.view_as(pred)).sum()
        return rights, len(labels)

    def sentence2vec(self, sentence, dictionary):
        """ 
        Convert sentence text to vector representation 
        """
        """
        #TODO 
        You should modify the code to define two methods to convert the review text to a vector:
        one is for Glove and another is for BOW. The first step is to set the size of the vectors, 
        which will be different for GLOVE and BOW. The next step is to create the vectors for your input sentences.
        Hint: Use numpy to init the vector. Retrieve the BOW vector from self.vocab defined as part of the init for the 
        class, and write a function to create the vector values. Retrieve the GLOVE word vectors from the 
        embeddings_dict created by the load_glove(path) function
        """
        # Code:
        if self.algo == "GLOVE":
          sentence_vector = [0] * self.embed_size
          countWord = 0

          for word in sentence:
            if word in self.embeddings_dict:
              sentence_vector += self.embeddings_dict[word]
              countWord += 1
          sentence_vector = sentence_vector // countWord
  
          return sentence_vector
        else:
            sentence_vector = [0] *len(dictionary)
            for i in sentence:
              sentence_vector[i] += 1
            return sentence_vector

    def load_glove(self, path):
        """
        Load Glove embeddings dictionary
        """
        """
        You should load the Glove embeddings from the local glove files like﻿"glove.8B.50d", 
        Then use "self.embeddings_dict" to store this words dict.
        """
        with open(path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
        return 0

    def training(self):
        """
        The whole training and testing process.
        """
        losses = []
        
        # TODO
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
        self.model.parameters(),
        self.lr)

        """
        Note that the learning rate (lr) for the optimizer is a command line parameter.

        Decide on the number of training epochs; it can be the same for both representations, or different
        """
        # TODO
        if self.algo == "GLOVE":
            tr_epochs = 100
        else:
            tr_epochs = 10

        for epoch in range(tr_epochs):
            print(epoch)
            for i, data in enumerate(zip(self.train_data, self.train_label)):
                x, y = data
                x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                y = torch.tensor(np.array([y]), dtype=torch.long)
                optimizer.zero_grad()
                # predict
                predict = self.model(x)
                # calculate loss
                loss = loss_function(predict, y)
                losses.append(loss.data.numpy())
                loss.backward()
                optimizer.step()
                # test every 1000 data
                if i % 1000 == 0:
                    val_losses = []
                    rights = []
                    for j, val in enumerate(zip(self.valid_data, self.valid_label)):
                        x, y = val
                        x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                        y = torch.tensor(np.array([y]), dtype=torch.long)
                        predict = self.model(x)
                        right = self.rightness(predict, y)
                        rights.append(right)
                        loss = loss_function(predict, y)
                        val_losses.append(loss.data.numpy())

                    right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
                    print('At the {} epoch，Training loss：{:.2f}, Testing loss：{:.2f}, Testing Acc: {:.2f}'.format(epoch, np.mean(losses),
                                                                                np.mean(val_losses), right_ratio))
        print("Training End")




