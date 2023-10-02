#!/usr/bin/env python3

"""
Tool for learning DPAs using alergia (including evaluation).

Copyright (C) 2020  Vojtech Havlena, <ihavlena@fit.vutbr.cz>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License.
If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import getopt
import os
import csv
import math
from enum import Enum
from dataclasses import dataclass

from typing import Tuple, FrozenSet

import learning.fpt as fpt
import learning.alergia as alergia
import parser.IEC104_parser as con_par
import parser.IEC104_conv_parser as iec_prep_par


import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import learning.Network_LSTM as LSTM

#configuration
BATCH_SIZE = 32
INPUT_DIM = 2
HIDDEN_DIM = 5
STACKED_LAYERS = 1
LEARNING_RATE = 0.001
NUM_EPOCH = 10
SEQUENCE_LEN = 10

ComPairType = FrozenSet[Tuple[str,str]]
rows_filter = ["asduType", "cot"]
TRAINING = 0.33

"""
Program parameters
"""
class Algorithms(Enum):
    PA = 0
    PTA = 1


"""
Program parameters
"""
class InputFormat(Enum):
    IPFIX = 0
    CONV = 1


"""
Program parameters
"""
@dataclass
class Params:
    alg : Algorithms
    file : str
    file_format : InputFormat


"""
Abstraction on messages
"""
def abstraction(item):
    return tuple([item[k] for k in rows_filter])



"""
Print help message
"""
def print_help():
    print("./pa_learning <csv file> [OPT]")
    print("OPT are from the following: ")
    print("\t--atype=pa/pta\t\tlearning based on PAs/PTAs (default PA)")
    print("\t--format=conv/ipfix\tformat of input file: conversations/IPFIX (default IPFIX)")
    print("\t--help\t\t\tprint this message")


"""
Function for learning based on Alergia (PA)
"""
def learn_pa(training):
    if len(training) == 0:
        raise Exception("training set is empty")

    tree = fpt.FPT()
    for line in training:
        tree.add_string(line)

    alpha = 0.05
    t0 = int(math.log(len(training), 2))

    aut = alergia.alergia(tree, alpha, t0)
    aut.rename_states()
    return aut.normalize(), alpha, t0


"""
Communication entity string format
"""
def ent_format(k: ComPairType) -> str:
    [(fip, fp), (sip, sp)] = list(k)
    return "{0}v{1}--{2}v{3}".format(fip, fp, sip, sp)


"""
Function for learning based on prefix trees (PTA)
"""
def learn_pta(training):
    if len(training) == 0:
        raise Exception("training set is empty")

    tree = fpt.FPT()
    tree.add_string_list(training)
    tree.rename_states()
    return tree.normalize(), None, None


"""
Store automaton into file
"""
def store_automata(csv_file, fa, alpha, t0, par=""):
    store_filename = os.path.splitext(os.path.basename(csv_file))[0]
    if (alpha is not None) and (t0 is not None):
        store_filename = "{0}a{1}t{2}{3}".format(store_filename, alpha, t0, par)
    else:
        store_filename = "{0}{1}-pta".format(store_filename,par)

    fa_fd = open("{0}.fa".format(store_filename), "w")
    fa_fd.write(fa.to_fa_format(True))
    fa_fd.close()

    if (alpha is not None) and (t0 is not None):
        legend = "File: {0}, alpha: {1}, t0: {2}, {3}".format(csv_file, alpha, t0, par)
    else:
        legend = "File: {0}, {1}".format(csv_file, par)
    dot_fd = open("{0}.dot".format(store_filename), "w")
    dot_fd.write(fa.to_dot(aggregate=False, legend=legend))
    dot_fd.close()

"""
Transform list of tuples to pytorch tensors
"""
def list_tensor(x):
    x = numpy.array(x).astype(float)
    x = torch.tensor(x, dtype=torch.float32)
    return x

"""
Training function
"""
def train_epoch(model, training_loader, epoch, loss_function, optimizer):
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(training_loader):
        x_batch, y_batch = batch[0].to(LSTM.DEVICE), batch[1].to(LSTM.DEVICE)
        
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_index % 100 == 99:
            avg_loss = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index + 1, avg_loss))
            running_loss = 0.0
    print()


"""
validation function
"""
def validate_epoch(model, testing_loader, loss_function):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(testing_loader):
        x_batch, y_batch = batch[0].to(LSTM.DEVICE), batch[1].to(LSTM.DEVICE)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss

    avg_loss = running_loss / len(testing_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss))
    print('***********************************************')
    print()

"""
Main
"""
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ha:f:", ["help", "atype=", "format="])
        if len(args) > 0:
            opts, _ = getopt.getopt(args[1:], "ha:f:", ["help", "atype=", "format="])
    except getopt.GetoptError as err:
        sys.stderr.write("Error: bad parameters (try --help)\n")
        sys.exit(1)

    params = Params(Algorithms.PA, None, InputFormat.IPFIX)
    learn_fnc = learn_pa

    for o, a in opts:
        if o in ("-a", "--atype"):
            if a == "pa":
                params.alg = Algorithms.PA
                learn_fnc = learn_pa
            elif a == "pta":
                params.alg = Algorithms.PTA
                learn_fnc = learn_pta
        elif o in ("-h", "--help"):
            print_help()
            sys.exit()
        elif o in ("-f", "--format"):
            if a == "conv":
                params.file_format = InputFormat.CONV
            elif a == "ipfix":
                params.file_format = InputFormat.IPFIX
        else:
            sys.stderr.write("Error: unrecognized parameters (try --help)\n")
            sys.exit(1)

    if len(args) == 0:
        sys.stderr.write("Missing input file (try --help)\n")
        sys.exit(1)
    params.file = args[0]

    try:
        csv_fd = open(params.file, "r")
    except FileNotFoundError:
        sys.stderr.write("Cannot open file: {0}\n".format(params.file))
        sys.exit(1)
    csv_file = os.path.basename(params.file)

    ############################################################################
    # Preparing the learning data
    ############################################################################
    normal_msgs = con_par.get_messages(csv_fd)
    csv_fd.close()

    parser = None
    try:
        if params.file_format == InputFormat.IPFIX:
            parser = con_par.IEC104Parser(normal_msgs)
        elif params.file_format == InputFormat.CONV:
            parser = iec_prep_par.IEC104ConvParser(normal_msgs)
    except KeyError as e:
        sys.stderr.write("Missing column in the input csv: {0}\n".format(e))
        sys.exit(1)

    for compr_parser in parser.split_communication_pairs():
        compr_parser.parse_conversations()

        lines = compr_parser.get_all_conversations(abstraction)
        index = int(len(lines)*TRAINING)
        training, testing = lines[:index], lines[index:]

        try:
            fa, alpha, t0 = learn_fnc(training)
        except Exception as e:
            sys.stderr.write("Learning error {0}: {1}\n".format(csv_file, e))
            sys.exit(1)

        par = ent_format(compr_parser.compair)
        store_automata(csv_file, fa, alpha, t0, par)

        miss = 0
        for line in testing:
            prob = fa.string_prob_deterministic(line)
            if prob is None:
                miss += 1

        print("File: {0} {1}".format(csv_file, ent_format(compr_parser.compair)))
        if (alpha is not None) and (t0 is not None):
            print("alpha: {0}, t0: {1}".format(alpha, t0))
        print("States {0}".format(len(fa.get_states())))
        print("Testing: {0}/{1} (missclassified/all)".format(miss, len(testing)))
        if len(testing) > 0:
            print("Accuracy: {0}".format((len(testing)-miss)/float(len(testing))))

        for pair in testing:
            print(pair)

        # Preparing data for NN
        x_training = list_tensor(training)
        x_testing = list_tensor(testing)

        # tensor of outputs for training
        y_training = torch.ones(x_training.shape[0], 1)
        y_testing = torch.ones(x_testing.shape[0], 1)

        training_dataset = LSTM.NetworkTrainingDataset(x_training, y_training)
        testing_dataset = LSTM.NetworkTrainingDataset(x_testing, y_testing)
        #testing_dataset = LSTM.NetworkTestingDataset(x_testing)

        training_loader = DataLoader(training_dataset, BATCH_SIZE, shuffle=True)
        testing_loader = DataLoader(testing_dataset, BATCH_SIZE, shuffle=False)
        
        model = LSTM.NetworkLSTM(INPUT_DIM, HIDDEN_DIM, STACKED_LAYERS)
        model.to(LSTM.DEVICE)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCH):
            train_epoch(model, training_loader, epoch, loss_function, optimizer)
            validate_epoch(model, testing_loader, loss_function)

        #TODO add the nn training and detection here

if __name__ == "__main__":
    main()
