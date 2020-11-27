""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from torch.utils.data import Dataset, DataLoader
from fractions import Fraction

MIDI_DIRECTORY_PATH = "/Users/shivangsingh/Desktop/midis/*"

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
            result.append([self.notes[pitch], self.duration[duration], float(offset)])

        return result


def train():
    """ train the neural network """
    seq = get_notes()
    token = Tokenizer()
    token.indexData(seq)
    tokenizedData = token.tokenizeData(seq)

    dataset = slidingWindow(tokenizedData)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # training loop
    for i, batch in enumerate(dataloader):
        print(i, batch)


def slidingWindow(data, windowSize = 20):
    result = []

    for i in range(0, len(data) - windowSize - 1):
        result.append((data[i:i+windowSize], data[i+windowSize]))

    return result


if __name__ == '__main__':
    train()
