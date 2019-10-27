import json
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pdb
import pickle
import random
import sys
from glob import glob
from operator import itemgetter
import tensorflow as tf
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except AttributeError:
    tf.logging.set_verbosity(tf.logging.ERROR)
from supervised_reptile.args import argument_parser, evaluate_kwargs, model_kwargs, train_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models import MiniImageNetModel, OmniglotModel
from supervised_reptile.miniimagenet import read_dataset as mini_dataset
from supervised_reptile.omniglot import read_dataset, split_dataset, augment_dataset
from supervised_reptile.train import train


DATA_DIR = {'omniglot': 'data/omniglot', 'miniimagenet': 'data/miniimagenet'}


def main():

    parser = argument_parser()
    parser.add_argument('--dataset', help='which dataset to use', default='omniglot', type=str)
    parser.add_argument('--trials', help='number of seeds', default=3, type=int)
    parser.add_argument('--val-samples', help='number of validation samples', default=100, type=int)
    parser.add_argument('--restore', help='restore from final training checkpoint', action='store_true')

    args = parser.parse_args()

    os.makedirs(args.checkpoint, exist_ok=True)
    with open(os.path.join(args.checkpoint, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    random.seed(args.seed)
    if args.dataset == 'omniglot':
        train_set, test_set = split_dataset(read_dataset(DATA_DIR[args.dataset]))
        trainval_set = list(augment_dataset(train_set))
        val_set = list(train_set[-200:])
        train_set = list(augment_dataset(train_set[:-200]))
        test_set = list(test_set)
    else:
        train_set, val_set, test_set = mini_dataset(DATA_DIR[args.dataset])
        trainval_set = train_set

    print('Training...')
    modes = ['train', 'test']
    metrics = ['acc', 'loss']
    try:
        with open(os.path.join(args.checkpoint, 'results.pkl'), 'rb') as f:
            results = pickle.load(f)
    except FileNotFoundError:
        results = {'regular': {mode: {metric: {} for metric in metrics} for mode in modes}, 'trials': set()}
        if args.transductive:
            results['transductive'] = {mode: {metric: {} for metric in metrics} for mode in modes}
        if args.adaptive:
            results['adaptive'] = {mode: {metric: {} for metric in metrics} for mode in modes}
            if args.transductive:
                results['adaptive_transductive'] = {mode: {metric: {} for metric in metrics} for mode in modes}

    for j in range(args.trials):
        if j in results['trials']:
            print('skipping trial', j)
            continue
        tf.reset_default_graph()
        model = OmniglotModel(args.classes, **model_kwargs(args)) if args.dataset == 'omniglot' else MiniImageNetModel(args.classes, **model_kwargs(args))
        with tf.Session() as sess:
            checkpoint = os.path.join(args.checkpoint, 'final'+str(j))
            if args.restore:
                print('Trial', j, 'Restoring...')
                tf.train.Saver().restore(sess, tf.train.latest_checkpoint(checkpoint))
                args.restore = False
            else:
                os.makedirs(checkpoint, exist_ok=True)
                print('Trial', j, 'Training...')
                train(sess, model, trainval_set, test_set, checkpoint, **train_kwargs(args))
            print('Trial', j, 'Evaluating...')
            for ev in results.keys():
                if ev == 'trials':
                    continue
                evkw = evaluate_kwargs(args)
                evkw['transductive'] = 'transductive' in ev
                evkw['adaptive'] = args.adaptive if 'adaptive' in ev else 0.0
                for mode, dset in zip(modes, [trainval_set, test_set]):
                    results[ev][mode]['acc'][j], results[ev][mode]['loss'][j] = evaluate(sess, model, dset, **evkw)
            results['trials'].add(j)
        with open(os.path.join(args.checkpoint, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        with open(os.path.join(args.checkpoint, 'results.json'), 'w') as f:
            tr = results.pop('trials')
            json.dump(results, f, indent=4)
            results['trials'] = tr

    print('Train Acc:', sum(results['regular']['train']['acc'].values())/args.trials)
    print('Test Acc:', sum(results['regular']['test']['acc'].values())/args.trials)
    if args.transductive:
        print('Transductive Train Acc:', sum(results['transductive']['train']['acc'].values())/args.trials)
        print('Transductive Test Acc:', sum(results['transductive']['test']['acc'].values())/args.trials)
    if args.adaptive:
        print('Adaptive Train Acc:', sum(results['adaptive']['train']['acc'].values())/args.trials)
        print('Adaptive Test Acc:', sum(results['adaptive']['test']['acc'].values())/args.trials)
        if args.transductive:
            print('Adaptive Transductive Train Acc:', sum(results['adaptive_transductive']['train']['acc'].values())/args.trials)
            print('Adaptive Transductive Test Acc:', sum(results['adaptive_transductive']['test']['acc'].values())/args.trials)


if __name__ == '__main__':

    main()
