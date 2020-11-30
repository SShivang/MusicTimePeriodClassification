""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
import math
from collections import defaultdict
from music21 import converter, instrument, note, chord
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from fractions import Fraction
from torch import optim
import random
import timeit
import collections

MIDI_DIRECTORY_PATH = "/Users/gengmengning/Downloads/midis/*"

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    list = ["/Users/gengmengning/Downloads/midis/Kötzschke, Hanns, Piano Sonata, BY8n7VRjhg4.mid",
            "/Users/gengmengning/Downloads/midis/Bennett, William Sterndale, 3 Musical Sketches, Op.10, HjT2pciYq6M.mid",
            "/Users/gengmengning/Downloads/midis/Mendelssohn, Felix, Clarinet Sonata in E-flat major, MWV Q 15, zhotPVqosa0.mid"]

    for file in list:
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes


        for element in notes_to_parse:
            # print(element.duration)
            if isinstance(element, note.Note):
                # print(str(element.pitch))
                notes.append((str(element.pitch), element.duration.quarterLength , element.offset))
            elif isinstance(element, chord.Chord):
                # print('.'.join(str(n) for n in element.normalOrder))
                notes.append(('.'.join(str(n) for n in element.normalOrder), float(element.duration.quarterLength) , float(element.offset)))

    return notes

    # with open('data/notes', 'wb') as filepath:
    #     pickle.dump(notes, filepath)


class Tokenizer():

    def __init__(self):
        self.notes = {"<sos>": 0}
        self.duration = {"<sos>": 0}
        self.numNotes = 1
        self.numDurations = 1

    def indexData (self, data):
        for (pitch, duration, offset) in data:
            if not pitch in self.notes:
                self.notes[pitch] = len(self.notes)
                self.numNotes += 1
            if not duration in self.duration:
                if (isinstance(duration, Fraction)):
                    duration = float(duration)
                duration = round(duration, 5)
                self.duration[duration] = len(self.duration)
                self.duration += 1

    def tokenizeData(self, data):
        result = defaultdict(list)
        for (pitch, duration, offset) in data:
            if (isinstance(duration, Fraction)):
                duration = float(duration)
            duration = round(duration, 5)
            result["pitch"].append(self.notes[pitch])
            result["duration"].append(self.duration[duration])
            result["offset"].append(float(offset))
        return result



class RNNLanguageModel(nn.Module):
    def __init__(self, total_num_pitches, total_num_durations,
                 emb_size, hidden_size, dropout, rnn_type='lstm'):
        super(RNNLanguageModel, self).__init__()
        self.pitch_embedding = nn.Embedding(total_num_pitches, emb_size)
        self.duration_embedding = nn.Embedding(total_num_durations, emb_size)
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers=1, dropout=dropout, batch_first=True)
        self.W1 = nn.Linear(hidden_size, total_num_pitches)
        self.W2 = nn.Linear(hidden_size, total_num_durations)

    def forward(self, pitch_input, duration_input):
        # pitch_input: [batch_size, input_length]
        # duration_input: [batch_size, input_length]
        embedded_pitch_input = self.pitch_embedding(pitch_input)
        embedded_duration_input = self.duration_embedding(duration_input)
        # embedded_pitch_input: [batch_size, input_length, emb_size]
        # embedded_duration_input: [batch_size, input_length, emb_size]
        embedded_input = embedded_pitch_input + embedded_duration_input
        output, (hidden_state, cell_state) = self.rnn(embedded_input)
        # output: [batch_size, input_length, hidden_size]
        # hidden_state: [batch_size, num_of_layers, hidden_size]
        # cell_state: [batch_size, num_of_layers, hidden_size]
        probs_pitch = self.log_softmax(self.W1(hidden_state.squeeze()))
        probs_duration = self.log_softmax(self.W2(hidden_state.squeeze()))
        return probs_pitch, probs_duration


def train():
    """ train the neural network """
    seq = get_notes()
    token = Tokenizer()
    token.indexData(seq)
    tokenizedData = token.tokenizeData(seq)
    dataset_pitch, label_pitch = slidingWindow(tokenizedData["pitch"], 20, 20)
    dataset_duration, label_duration = slidingWindow(tokenizedData["duration"], 20, 20)
    dataset_offset, label_offset = slidingWindow(tokenizedData["offset"], 20, 20)
    num_epochs = 5
    total_num_pitches = token.numNotes
    total_num_durations = token.numDurations
    model = RNNLanguageModel(total_num_pitches, total_num_durations, 128, 128, dropout=0.1)
    initial_learning_rate = 0.01
    nll_loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    # training loop
    for epoch in range(0, num_epochs):
        total_loss = 0.0
        batch_pitch_list, batch_pitch_list_label = torch.tensor(batch(32, dataset_pitch)), torch.tensor(batch(32, label_pitch))
        batch_duration_list, batch_duration_list_label = torch.tensor(batch(32, dataset_duration)), torch.tensor(batch(32, label_duration))
        for i in range(len(batch_pitch_list)):
            pitch_label = batch_pitch_list_label[i]
            duration_label = batch_duration_list_label[i]
            model.zero_grad()
            probs = model.forward(batch_pitch_list[i], batch_duration_list[i])
            loss = nll_loss(probs, y)
            total_loss += loss
            loss.backward()
            optimizer.step()
        model.eval()
    return model


def batch(bs, dataset):
    for i in range(0, len(dataset), bs):
        yield torch.tensor(dataset[i:i + bs])


def slidingWindow(data, stride, windowSize):
    input = []
    label = []
    for i in range(0, math.ceil((len(data) - windowSize - 1) / stride)):
        input_element = data[i*stride:i*stride+windowSize]
        input_element.insert(0, 0)
        input.append(torch.tensor(input_element))
        label.append(torch.tensor(data[i*stride:i*stride+windowSize+1]))
    return input, label


if __name__ == '__main__':
    train()