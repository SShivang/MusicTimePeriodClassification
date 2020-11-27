""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from fractions import Fraction

MIDI_DIRECTORY_PATH = "/Users/shivangsingh/Desktop/midis/*"

def train_network():
    """ Train a Neural Network to generate music """
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    list = ["/Users/shivangsingh/Desktop/Neural Network Project/midis/Kötzschke, Hanns, Piano Sonata, BY8n7VRjhg4.mid", "/Users/shivangsingh/Desktop/Neural Network Project/midis/Bennett, William Sterndale, 3 Musical Sketches, Op.10, HjT2pciYq6M.mid", "/Users/shivangsingh/Desktop/Neural Network Project/midis/Mendelssohn, Felix, Clarinet Sonata in E-flat major, MWV Q 15, zhotPVqosa0.mid"]

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
        self.notes = {}
        self.duration = {}
        self.numNotes =0
        self.numDurations = 0

    def indexData (self, data):
        for (pitch, duration, offset) in data:
            if not pitch in self.notes:
                self.notes[pitch] = len(self.notes)
            if not duration in self.duration:
                if (isinstance(duration, Fraction)):
                    duration = float(duration)
                duration = round(duration, 5)
                self.duration[duration] = len(self.duration)

    def tokenizeData(self, data):
        result = []
        for (pitch, duration, offset) in data:
            if (isinstance(duration, Fraction)):
                duration = float(duration)
            duration = round(duration, 5)
            result.append([self.notes[pitch], self.duration[duration]])

        return result

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train():
    """ train the neural network """
    seq = get_notes()
    token = Tokenizer()
    token.indexData(seq)
    tokenizedData = token.tokenizeData(seq)

    dataset = slidingWindow(tokenizedData)

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    for i, batch in enumerate(dataloader):
        print(i, batch)


def slidingWindow(data, windowSize = 20):
    result = []

    for i in range(0, len(data) - windowSize - 1):
        result.append((data[i:i+windowSize], data[i+windowSize]))

    return result


if __name__ == '__main__':
    train()
