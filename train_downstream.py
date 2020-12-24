""" This module prepares midi file data and feeds it to the neural
    network for training """
import multiprocessing
import math
from collections import defaultdict
from music21 import converter, instrument, note, chord
import torch.nn as nn
import torch
from fractions import Fraction
from torch import optim
import random
import os
import pickle
import numpy as np
import pandas as pd
import torch.utils
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import Subset
from matplotlib import pyplot as plt


def parse_single_file(file):
    # print("Parsing: {}, current core: {}".format(file, multiprocessing.current_process()))
    midi = converter.parse(file)
    print(midi)
    notes_to_parse = None

    try:  # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()
    except:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
    notes = []
    for element in notes_to_parse:
        # print(element.duration)
        if isinstance(element, note.Note):
            # print(str(element.pitch))
            notes.append((str(element.pitch), element.duration.quarterLength, element.offset))
        elif isinstance(element, chord.Chord):
            # print('.'.join(str(n) for n in element.normalOrder))
            notes.append(('.'.join(str(n) for n in element.normalOrder), float(element.duration.quarterLength),
                          float(element.offset)))
    return notes


def get_notes(path):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    input_paths = os.listdir(path)
    for i in range(len(input_paths)):
        input_paths[i] = path + "/" + input_paths[i]
    pool = multiprocessing.Pool(8)
    try:
        res_list = pool.map(parse_single_file, input_paths)
    finally:
        pool.close()
        pool.join()
    print("parsing finished")
    final_res_list = []
    for list in res_list:
        final_res_list.extend(list)

    return final_res_list

class Tokenizer():

    def __init__(self):
        self.notes = {"<sos>": 0}
        self.duration = {"<sos>": 0}

    def indexData (self, data):
        for (pitch, duration, offset) in data:
            if not pitch in self.notes:
                self.notes[pitch] = len(self.notes)
            if (isinstance(duration, Fraction)):
                duration = float(duration)
            temp = round(duration, 5)
            if not temp in self.duration:
                self.duration[temp] = len(self.duration)

    def tokenizeData(self, data):
        result = defaultdict(list)
        for (pitch, duration, offset) in data:
            if (isinstance(duration, Fraction)):
                duration = float(duration)
            duration = round(duration, 5)
            try:
                result["pitch"].append(self.notes[pitch])
                result["duration"].append(self.duration[duration])
                result["offset"].append(float(offset))
            except KeyError:
                continue
        return result


class RNNOverNotes(nn.Module):
    def __init__(self, total_num_pitches, total_num_durations,
                 emb_size, hidden_size, dropout, outputs, rnn_type='lstm'):
        super(RNNOverNotes, self).__init__()
        self.pitch_embedding = nn.Embedding(total_num_pitches, emb_size)
        self.duration_embedding = nn.Embedding(total_num_durations, emb_size)
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers=1, dropout=dropout, batch_first=True)
        self.W1 = nn.Linear(hidden_size, outputs)

    def forward(self, pitch_input, duration_input):

        embedded_pitch_input = self.pitch_embedding(pitch_input)
        embedded_duration_input = self.duration_embedding(duration_input)

        embedded_input = embedded_pitch_input + embedded_duration_input
        output, (hidden_state, cell_state) = self.rnn(embedded_input)
        probs_timeperiod = self.log_softmax(self.W1(output))
        return probs_timeperiod


class RNNLanguageModel:
    def __init__(self, model):
        self.model = model

    def get_log_prob_single(self, next_pitches, next_durations, pitch_input, duration_input):
        probs_pitches, probs_durations = self.model.forward(torch.tensor(pitch_input).unsqueeze(0), torch.tensor(duration_input).uunsqueeze(0)).detach().numpy()
        prob_next_pitches = probs_pitches[next_pitches]
        prob_next_durations = probs_durations[next_durations]
        return prob_next_pitches, prob_next_durations

    def get_log_prob_sequence(self, next_pitches, next_durations, pitch_input, duration_input):
        TOTAL_pitch_prob = 0
        TOTAL_duration_prob = 0
        for i in range(len(next_pitches)):
            pitch_new = pitch_input[i:] + next_pitches[:i]
            duration_nex = duration_input[i:] + next_durations[:i]
            pitch_prob, duration_prob = self.get_log_prob_single(next_pitches[i], next_durations[i], pitch_new, duration_nex)
            TOTAL_pitch_prob += pitch_prob
            TOTAL_duration_prob += TOTAL_duration_prob
        return TOTAL_pitch_prob, TOTAL_duration_prob


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def convertToClass(year):
    year = int(year)
    if (year < 1500):
        return 0
    if (year > 1500 and year < 1700):
        return 1
    if (year > 1700 and year < 1750):
        return 2
    if (year > 1750 and year < 1800):
        return 3
    if (year > 1800 and year < 1830):
        return 4
    if (year > 1830 and year < 1870):
        return 5
    if (year > 1870 and year < 1900):
        return 6
    if (year > 1900):
        return 7

def train_downstream():


    """ train the neural network """
    train_record_path = "./train_record.pk"
    Train_PATH = "../test"
    Dev_PATH = "../test"

    d = pd.read_csv("timePeriodMapping.csv", sep="\t")
    d["combined_name"] = d['surname'].map(str) + ', ' + d['firstname'].map(str)
    result = d.set_index('combined_name').T.to_dict()

    Train_PATH = "../train_downstream"
    input_paths = os.listdir(Train_PATH)
    i = 0

    inputs = []
    tot_notes = []
    classes = []
    for path in input_paths:
        try :
            person = path.split(",")[0] + "," + path.split(",")[1]
            print(i, person)
            if person in result:
                if (result[person]['birth'] != "unknown" and i < 50):
                    notes = parse_single_file(Train_PATH+"/"+path)
                    tot_notes.extend(notes)
                    inputs.append((notes, convertToClass(result[person]['birth'])))
                    classes.append(convertToClass(result[person]['birth']))
                    i = i + 1
        except :
            print("person" + path)


    dev_record_path = "./train_record_lm.pk"
    dev_seq = pickle.load(open(dev_record_path, 'rb'))

    data = []
    for i in range(len(dev_seq)):
        data.extend(dev_seq[i]['notes'])

    t = Tokenizer()
    t.indexData(data)

    tokenized_inputs = []

    classes = []

    for data,label in inputs:
        tokenized = t.tokenizeData(data)
        new_pitch_data, new_labels = slidingWindow(tokenized['pitch'], 20, 20, label)
        new_duration_data, new_labels = slidingWindow(tokenized['duration'], 20, 20, label)
        for i in range(len(new_duration_data)):
            try:
                tokenized_inputs.append((np.array([new_pitch_data[i],new_duration_data[i]], dtype=np.uint8), np.array(new_labels[i], dtype=np.uint8)))
                classes.append(label)
            except:
                print(new_pitch_data)

    plt.hist(classes, bins=8)
    plt.show()

    modified_data = train_val_dataset(tokenized_inputs)

    train_loader = DataLoader(modified_data["train"], batch_size =64, shuffle = True)
    dev_loader = DataLoader(modified_data["val"], batch_size =64, shuffle = False)

    model = RNNOverNotes(len(t.notes), len(t.duration), 50, 50, dropout=0.1, outputs=8)
    # model = torch.load("pretrained_lm_best.pt")

    initial_learning_rate = 0.001
    nll_loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

    num_epochs = 64

    for epoch in range(0, num_epochs):
        print(epoch)
        train_total_loss = 0.0
        test_total_loss = 0.0
        sum_f1 = 0.0
        sum_accuracy = 0.0
        for data, label in iter(train_loader):
            transposed = torch.transpose(data, 0, 1)
            pitches = transposed[0].long()
            durations = transposed[1].long()
            label = label.long()
            probs = model.forward(pitches, durations)
            classification_loss = nll_loss(probs.transpose(1, 2), label)
            loss = classification_loss
            train_total_loss += classification_loss
            loss.backward()
            optimizer.step()

        for data, label in iter(dev_loader):
            transposed = torch.transpose(data, 0, 1)
            pitches = transposed[0].long()
            durations = transposed[1].long()
            label = label.long()
            probs = model.forward(pitches, durations)
            predictions = torch.argmax(probs, dim = -1)
            for b in range(len(predictions)):
                sample_f1 = f1_score(predictions.numpy()[b], label.numpy()[b], average='weighted')
                sum_f1 += sample_f1
                sum_accuracy += accuracy_score(predictions.numpy()[b], label.numpy()[b])
            classification_loss = nll_loss(probs.transpose(1, 2), label)
            loss = classification_loss
            test_total_loss += classification_loss

        print(train_total_loss, test_total_loss, sum_f1/len(modified_data["val"]), sum_accuracy/len(modified_data["val"]))

    return RNNLanguageModel(model)


def batch(bs, dataset):
    for i in range(0, len(dataset), bs):
        yield torch.LongTensor(dataset[i:i + bs])


def slidingWindow(data, stride, windowSize, l):
    input = []
    label = []
    for i in range(0, math.ceil((len(data) - windowSize - 1) / stride)):
        input_element = data[i*stride:i*stride+windowSize]
        input_element.insert(0, 0)
        input.append(input_element)
        label.append([l]* (windowSize+1))
    return input, label



if __name__ == '__main__':

    train_downstream()

    # pretrained = [269.7546, 245.2379, 234.5252, 228.1013, 222.9419, 215.3953, 209.2990, 201.8422, 193.5248, 184.7912, 178.2723,
    # 168.1528, 158.1804, 151.9943, 143.1302, 138.6015, 136.6225, 136.3991, 133.0374, 129.5001, 127.9119, 118.6662, 112.8633,113.4936
    # ,103.3885, 101.4649, 95.4878,93.1707]
    #
    # from_scratch = [277.5245, 265.1591, 259.2801, 254.0755, 246.5464, 240.9550, 237.7403, 234.3348, 230.4709, 228.9903,
    # 225.9451, 223.8794, 221.0402, 218.6401, 215.8376, 214.7515, 213.2289, 208.4751, 208.7539, 204.6742, 202.8121, 202.1627,
    # 197.5896, 197.6708, 193.0410, 192.0672, 189.3810, 186.4594]
    #
    #
    # plt.plot(np.arange(len(pretrained)), pretrained, label = "Pretrained")
    # plt.plot(np.arange(len(pretrained)), from_scratch, label = "No Pretraining")
    # plt.title("Epoch vs Training Loss")
    # plt.ylabel("Training Loss (entropy)")
    # plt.xlabel("Epochs")
    # plt.legend()
    # plt.show()

    # for path in input_paths:

    # print(i)
