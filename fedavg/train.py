import argparse
import json
import os
import pdb
import pickle
import random
import string
import sys
from collections import defaultdict
from copy import deepcopy
from glob import glob
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.nn.utils import rnn
from config import FEDAVG
from aruba import MetaOpt


CACHE = FEDAVG + 'cache/'
DATA = FEDAVG + 'data/shakespeare/data/raw_data/by_play_and_character/'
CHARMAP = defaultdict(lambda: 1)
CHARMAP.update({char: i+2 for i, char in enumerate(string.printable)})
VOCAB = len(set(CHARMAP.values())) + 1
MAXLEN = 80
VALLR = [0.0, None, 'iso'] + [float(10 ** -i) for i in range(1, 4)]
COEFS = [0.0] + [float(10 ** i) for i in range(-2, 4)]
REFMAP = lambda lr, coef: '/ARUBA++'+str(coef)+'/' if coef else '/ARUBA/' if lr is None else '/isotropic/' if lr == 'iso' else '/global/' if lr == 0.0 else '/SGD-'+str(lr)+'/'


def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('logdir')
    parser.add_argument('--rounds', default=500, type=int, help='number of averagings')
    parser.add_argument('--batch', default=10, type=int, help='within-task batch size')
    parser.add_argument('--meta-batch', default=10, type=int, help='number of tasks to sample')
    parser.add_argument('--lr', default=1.0, type=float, help='within-task learning rate')
    parser.add_argument('--aruba', action='store_true', help='train using ARUBA')
    parser.add_argument('--iso', action='store_true', help='train using isotropic ARUBA')
    parser.add_argument('--eps', default=1.0, type=float)
    parser.add_argument('--zeta', default=1.0, type=float)
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--decay', default=1.0, type=float, help='decay within-task lr')
    parser.add_argument('--hidden', default=256, type=int, help='number of hidden units in LSTM')
    parser.add_argument('--layers', default=2, type=int, help='number of layers in LSTM')
    parser.add_argument('--seed', default=0, type=int, help='random task seed')
    parser.add_argument('--sweep', action='store_true', help='sweep through all optimizers at meta-test-time')
    parser.add_argument('--numval', default=10, type=int, help='number of meta-val tasks during meta-training')
    parser.add_argument('--valint', default=10, type=int, help='interval between meta-val during meta-training')
    parser.add_argument('--adaptive', action='store_true', help='meta-test ARUBA++')
    return parser.parse_args()


def write(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()
    return len(msg)


def traintest(data, frac=0.8):
    return data[:int(frac*len(data))], data[int(frac*len(data)):]


def line2data(line):
    return [[line[max(i-MAXLEN, 0):i], char] for i, char in enumerate(line)][1:]


def file2tensor(fname):
    with open(fname, 'r') as f:
        lines = [line for line in f]
    trainlines, testlines = traintest(lines)
    trainlines, testlines = [''.join(trainlines)], [''.join(testlines)]
    train, test = sum((line2data(line) for line in trainlines), []), sum((line2data(line) for line in testlines), [])
    Xtrain, Ytrain = torch.zeros((len(train), MAXLEN)).long(), torch.empty(len(train)).long()
    Xtest, Ytest = torch.zeros((len(test), MAXLEN)).long(), torch.empty(len(test)).long()
    for (X, Y, data) in [(Xtrain, Ytrain, train), (Xtest, Ytest, test)]:
        for i, (x, y) in enumerate(data):
            for j, char in enumerate(x):
                X[i,j] = CHARMAP[char]
            Y[i] = CHARMAP[y]
    return (Xtrain, Ytrain), (Xtest, Ytest)


class CharLSTM(nn.Module):

    def __init__(self, input_size=8, hidden_size=256, batch_size=10, **kwargs):

        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=VOCAB, embedding_dim=input_size, padding_idx=0)
        self.lstm= nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=False, **kwargs)
        self.linear = nn.Linear(hidden_size, VOCAB)

    def forward(self, X, lengths):

        X = self.embedding(X)
        X = rnn.pack_padded_sequence(X, lengths, batch_first=True)
        self.lstm.flatten_parameters()
        X, _ = self.lstm(X)
        X, _ = rnn.pad_packed_sequence(X, batch_first=True)
        return self.linear(X[:,-1])


def train(model, X, Y, meta_opt, batch=10, **kwargs):

    optimizer = meta_opt.optimizer(model.parameters(), **kwargs)
    criterion = nn.CrossEntropyLoss()
    m = len(Y)
    randperm = torch.randperm(m)
    X, Y = X[randperm], Y[randperm]
    for i in range(0, m, batch):
        lengths = MAXLEN - (X[i:i+batch] == 0).sum(1)
        lengths, sortperm = lengths.sort(0, descending=True)
        pred = model(X[i:i+batch][sortperm], lengths)
        loss = criterion(pred, Y[i:i+batch][sortperm])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return optimizer


def test(model, X, Y):
    lengths = MAXLEN - (X == 0).sum(1)
    lengths, sortperm = lengths.sort(0, descending=True)
    pred = model(X[sortperm], lengths).argmax(1)
    return (Y[sortperm] == pred).sum().float() / len(Y)


def eval(model, Xtrain, Ytrain, Xtest, Ytest, meta_opt, batch=10, **kwargs):

    if not meta_opt is None:
        train(model, Xtrain, Ytrain, meta_opt, batch=batch, **kwargs)
    return [sum(test(model, X[i:i+batch], Y[i:i+batch]) * min(len(Y)-i, batch) 
                for i in range(0, len(Y), batch)) / len(Y) 
            for X, Y in [(Xtrain, Ytrain), (Xtest, Ytest)]]


def meta_eval(model, tasks, meta_opt, vallr, batch=10, **kwargs):

    results = {lr: {mode: np.empty(len(tasks)) for mode in ['train', 'test']} for lr in vallr}
    num_samp = {mode: np.empty(len(tasks)) for mode in ['train', 'test']}
    for k, task in enumerate(tasks):
        write('\r\ttask '+str(k+1))
        data = torch.load(CACHE+task+'.pt')
        input = tuple(data[key].cuda() for key in ['Xtrain', 'Ytrain', 'Xtest', 'Ytest'])
        for lr in vallr:
            meta_opt.update_eta(lr=lr)
            results[lr]['train'][k], results[lr]['test'][k] = eval(deepcopy(model), *input, None if lr == 0.0 else meta_opt, batch=batch, **kwargs)
        num_samp['train'][k], num_samp['test'][k] = len(input[1]), len(input[3])
    write('\r')
    return results, num_samp


def main():

    args = parse()
    random.seed(args.seed)
    if args.sweep:
        vallr = VALLR
    elif args.aruba:
        vallr = [0.0, None]
    elif args.iso:
        vallr = [0.0, 'iso']
    else:
        vallr = [lr for lr in VALLR if type(lr) == float]
    if args.aruba and args.adaptive:
        coefs = COEFS
    else:
        coefs = [0.0]

    if not os.path.isdir(FEDAVG+'log'):
        os.makedirs(FEDAVG+'log')
    loggers = {(name, REFMAP(lr, coef)): SummaryWriter(os.path.join(FEDAVG, 'log', args.logdir, name, REFMAP(lr, coef)[1:]))
               for name in ['Fed', 'Meta'] for coef in coefs for lr in vallr for mode in ['train', 'test']}
    with open(FEDAVG+'log/'+args.logdir+'/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    fnames = glob(DATA+'*')
    tasks = [fname.split('/')[-1][:-4] for fname in fnames]
    if not os.path.isdir(CACHE):
        os.makedirs(CACHE)
    actual = []
    for i, (fname, task) in enumerate(zip(fnames, tasks)):
        if not os.path.isfile(CACHE+task+'.pt'):
            write('\rCaching task '+str(i)+' of '+str(len(tasks)))
            dump = {}
            (dump['Xtrain'], dump['Ytrain']), (dump['Xtest'], dump['Ytest']) = file2tensor(fname)
            torch.save(dump, CACHE+task+'.pt')
    write('\rCompleted cache-check'+20*' '+'\n')

    lstm = CharLSTM(hidden_size=args.hidden, num_layers=args.layers).cuda()
    metatrain, metatest = traintest(tasks)
    meta_opt = MetaOpt(zeta=args.zeta, p=args.p, lr=args.lr)
    output = {'Fed': {}, 'Meta': {}}
    for i in range(args.rounds):

        if not i % args.valint:
            write('\rRound '+str(i)+' evaluation:\n')
            for name, subset in [('Fed', metatrain), ('Meta', metatest)]:
                if args.aruba and args.adaptive:
                    output[name][i] = {}
                for coef in coefs:
                    results, num_samp = meta_eval(lstm, random.sample(subset, args.numval), meta_opt, vallr, batch=args.batch, coef=coef)
                    if args.aruba and args.adaptive:
                        output[name][i][coef] = {'results': results, 'num_samp': num_samp}
                    else:
                        output[name][i] = {'results': results, 'num_samp': num_samp}
                    for lr in vallr:
                        if coef and not lr is None:
                            continue
                        for mode in ['train', 'test']:
                            perf = (results[lr][mode] * num_samp[mode] / sum(num_samp[mode])).sum()
                            loggers[(name, REFMAP(lr, coef))].add_scalar(mode, perf, i)
                            if lr and type(lr) == float:
                                continue
                            write('\t'+name+REFMAP(lr, coef).replace('/', ' ')+mode+': \t'+str(round(perf, 4))+'\t')
                            if mode == 'test':
                                write('\n')

        meta_opt.update_eta(lr=None if args.aruba else 'iso' if args.iso else pow(args.decay, i) * args.lr)
        total = 0
        for k, task in enumerate(random.sample(metatrain, args.meta_batch)):
            write('\rRound '+str(i+1)+' task '+str(k+1)+5*' ')
            locallstm = deepcopy(lstm)
            data = torch.load(CACHE+task+'.pt')
            Xtrain, Ytrain = data['Xtrain'].cuda(), data['Ytrain'].cuda()
            optimizer = train(locallstm, Xtrain, Ytrain, meta_opt, batch=args.batch, collect=args.sweep or args.aruba or args.iso)
            meta_opt.update_state(optimizer)
            if k:
                for param, out in zip(locallstm.parameters(), outlstm.parameters()):
                    out.data += param.data * len(Ytrain)
            else:
                outlstm = locallstm
                for param in outlstm.parameters():
                    param.data *= len(Ytrain)
            total += len(Ytrain)
        for param in outlstm.parameters():
            param.data /= total
        lstm = outlstm

    torch.save(lstm, FEDAVG+'log/'+args.logdir+'/model.pt')
    torch.save(meta_opt, FEDAVG+'log/'+args.logdir+'/opt.pt')
    with open(FEDAVG+'log/'+args.logdir+'/online_eval.pkl', 'wb') as f:
        pickle.dump(output, f)

    write('\rFederated and Meta-Testing\n')
    output = {}
    pretty = {}
    for name, subset in [('Fed', metatrain), ('Meta', metatest)]:
        pretty[name] = {}
        for coef in coefs:
            results, num_samp = meta_eval(lstm, subset, meta_opt, vallr, batch=args.batch, coef=coef)
            for lr in vallr:
                if coef and not lr is None:
                    continue
                pretty[name][REFMAP(lr, coef).replace('/', '')] = {}
                for mode in ['train', 'test']:
                    perf = (results[lr][mode] * num_samp[mode] / sum(num_samp[mode])).sum()
                    write('\t'+name+REFMAP(lr, coef).replace('/', ' ')+mode+': \t'+str(round(perf, 4))+'\t')
                    if mode == 'test':
                        write('\n')
                    pretty[name][REFMAP(lr, coef).replace('/', '')][mode] = perf
        output[name] = {'results': results, 'num_samp': num_samp}

    with open(FEDAVG+'log/'+args.logdir+'/trained_eval.pkl', 'wb') as f:
        pickle.dump(output, f)
    with open(FEDAVG+'log/'+args.logdir+'/results.json', 'w') as f:
        json.dump(pretty, f, indent=4)


if __name__ == '__main__':

    main()
