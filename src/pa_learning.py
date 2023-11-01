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


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import learning.Network_LSTM as LSTM

# configuration
BATCH_COUNT = 75
INPUT_DIM = 2
MAX_HIDDEN_DIM = 10
MAX_STACKED_LAYERS = 2
LEARNING_RATE = 0.0001
MAX_EPOCH = 50
STOP_EARLY_THRESHOLD = 5
LEARNING = 0.85


ComPairType = FrozenSet[Tuple[str,str]]
rows_filter = ["asduType", "cot"]
TRAINING = 0.25

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
Transform list of conversations to pytorch tensors
"""
def list_tensor(x, conv_len):
    ret = torch.tensor(())
    tmp = torch.tensor((), dtype=torch.float32)

    for conv in range(len(x)):
        tmp = tmp.new_zeros(1, conv_len, INPUT_DIM)

        for msg in range(len(x[conv])):
            for input in range(INPUT_DIM):
                tmp[0][msg][input] = float(x[conv][msg][input])

        ret = torch.cat((ret, tmp))
    return ret

"""
Training function
"""
def train_epoch(model, learn_loader, epoch, loss_function, optimizer):
    model.train(True)

    for batch_index, batch in enumerate(learn_loader):
        x_batch, y_batch = batch[0].to(LSTM.DEVICE), batch[1].to(LSTM.DEVICE)
        
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

"""
validation function
"""
def validate_epoch(model, validate_loader, loss_function):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(validate_loader):
        x_batch, y_batch = batch[0].to(LSTM.DEVICE), batch[1].to(LSTM.DEVICE)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss

    avg_loss = running_loss / len(validate_loader)

    return avg_loss

"""
NN checkpoint creation
"""
def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


"""
NN checkpoint load
"""
def resume(model, filename):
    model.load_state_dict(torch.load(filename))


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
        training = lines[:index]

        # file name for best model for comunication pair
        par = ent_format(compr_parser.compair)
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        file_name = "{0}{1}.pth".format(file_name, par)

        # split training data to learning and validation data
        index = int(len(training)*LEARNING)
        learn, validate = training[:index], training[index:]

        # Preparing data for NN
        conv_len = max(len(row) for row in learn)
        x_learn = list_tensor(learn, conv_len)

        conv_len = max(len(row) for row in validate)
        x_validate = list_tensor(validate, conv_len)

        # tensors of outputs for training
        y_learn = torch.ones(x_learn.shape[0], 1)
        y_validate = torch.ones(x_validate.shape[0], 1)

        # datasets and dataloaders
        learn_dataset = LSTM.NetworkDataset(x_learn, y_learn)
        validate_dataset = LSTM.NetworkDataset(x_validate, y_validate)

        batch_size = (int)(index/BATCH_COUNT)
        learn_loader = DataLoader(learn_dataset, batch_size, shuffle=True)
        validate_loader = DataLoader(validate_dataset, batch_size, shuffle=False)

        # defining used loss function
        loss_function = nn.L1Loss()

        # varibles for picking best model
        best_loss_overall = 100
        best_layer = 0
        best_dimension = 0

        for stacked_layers in range(MAX_STACKED_LAYERS):
            for hidden_dimension in range(MAX_HIDDEN_DIM):
                # creating NN model, and optimizer
                model = LSTM.NetworkLSTM(INPUT_DIM, (hidden_dimension + 1), (stacked_layers + 1))
                model.to(LSTM.DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

                # variable for stopping training loop
                best_loss = 100
                best_epoch = 0

                # learning loop
                for epoch in range(MAX_EPOCH):
                    train_epoch(model, learn_loader, epoch, loss_function, optimizer)
                    loss = validate_epoch(model, validate_loader, loss_function)

                    # saving best model
                    if(loss < best_loss):
                        best_loss = loss
                        best_epoch = epoch
                        checkpoint(model, "model.pth")
                        

                    # if model doesnt improve stop loop
                    if(epoch - best_epoch > STOP_EARLY_THRESHOLD):
                        print("Training ({0} stacked layers and {1} hidden dimension) stopped early at epoch: {2}".format((stacked_layers + 1), (hidden_dimension + 1), epoch))
                        break
                    
                print("Loss ({0} stacked layers and {1} hidden dimension): %e".format((stacked_layers + 1), (hidden_dimension + 1)) % best_loss)

                if(best_loss < best_loss_overall):
                    best_loss_overall = best_loss
                    best_layer = (stacked_layers + 1)
                    best_dimension = (hidden_dimension + 1)
                    resume(model, "model.pth")
                    checkpoint(model, file_name)

        os.remove("model.pth")
        print("Best model found with: {0} stacked layers, {1} hidden dimension".format(best_layer, best_dimension))
        print("Best loss: %e" % best_loss_overall)


if __name__ == "__main__":
    main()
