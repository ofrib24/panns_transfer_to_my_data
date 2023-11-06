import os
# python3 utils/plot_statistics.py 1 --workspace=$WORKSPACE --select=2_cnn14
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging
import matplotlib.pyplot as plt
# from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
from config import (sample_rate, classes_num, mel_bins, fmin, fmax, window_size,
                    hop_size, window, pad_mode, center, ref, amin, top_db)
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup
from utilities import (create_folder, get_filename, create_logging, StatisticsContainer, Mixup)
from data_generator import GtzanDataset, TrainSampler, EvaluateSampler, collate_fn, CustomAudioDataset
from models import Transfer_Cnn14, Transfer_Logeml_Wavegram
from evaluate import Evaluator
from torch.utils.data import DataLoader


def train(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    resume_iteration = args.resume_iteration
    stop_iteration = args.stop_iteration
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = 8

    loss_func = get_loss_func(loss_type)
    pretrain = True if pretrained_checkpoint_path else False

    hdf5_path = os.path.join(workspace, 'features', 'waveform.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename,
                                   'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
                                   'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
                                   'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename,
                                   'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
                                   'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
                                   'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base),
                                   'statistics.pickle')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename,
                            'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
                            'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
                            'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
    create_logging(logs_dir, 'w')
    logging.info(args)

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')

    # Model
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)

    if resume_iteration:
        resume_checkpoint_path = os.path.join(checkpoints_dir, '{}_iterations.pth'.format(resume_iteration))
        logging.info('Load resume model from {}'.format(resume_checkpoint_path))
        resume_checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(resume_checkpoint['model'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = resume_checkpoint['iteration']
    else:
        iteration = 0

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    dataset = GtzanDataset()

    # Data generator
    train_sampler = TrainSampler(
        hdf5_path=hdf5_path,
        holdout_fold=holdout_fold,
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size)

    validate_sampler = EvaluateSampler(
        hdf5_path=hdf5_path,
        holdout_fold=holdout_fold,
        batch_size=batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_sampler=train_sampler, collate_fn=collate_fn,
                                               num_workers=num_workers, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_sampler=validate_sampler, collate_fn=collate_fn,
                                                  num_workers=num_workers, pin_memory=True)

    if 'cuda' in device:
        model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                           eps=1e-08, weight_decay=0., amsgrad=True)

    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)

    # Evaluator
    evaluator = Evaluator(model=model)

    train_bgn_time = time.time()

    # Train on mini batches
    for batch_data_dict in train_loader:

        # import crash
        # asdf
        # print(batch_data_dict['waveform'].dtype)
        # Evaluate
        if iteration % 200 == 0 and iteration > 0:
            if resume_iteration > 0 and iteration == resume_iteration:
                pass
            else:
                logging.info('------------------------------------')
                logging.info('Iteration: {}'.format(iteration))

                train_fin_time = time.time()

                statistics = evaluator.evaluate(validate_loader)
                logging.info('Validate accuracy: {:.3f}'.format(statistics['accuracy']))

                statistics_container.append(iteration, statistics, 'validate')
                statistics_container.dump()

                train_time = train_fin_time - train_bgn_time
                validate_time = time.time() - train_fin_time

                logging.info(
                    'Train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(train_time, validate_time))

                train_bgn_time = time.time()

        # Save model
        if iteration % 2000 == 0 and iteration > 0:
            checkpoint = {
                'iteration': iteration,
                'model': model.module.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))

            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))

        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(len(batch_data_dict['waveform']))

        # Move data to GPU
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        # Train
        model.train()
        print(batch_data_dict['waveform'].size())
        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['waveform'],
                                      batch_data_dict['mixup_lambda'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': do_mixup(batch_data_dict['target'],
                                                    batch_data_dict['mixup_lambda'])}
            """{'target': (batch_size, classes_num)}"""
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None)
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': batch_data_dict['target']}
            """{'target': (batch_size, classes_num)}"""

        # loss
        loss = loss_func(batch_output_dict, batch_target_dict)
        print(iteration, loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == stop_iteration:
            break

        iteration += 1


def train_on_decase(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    resume_iteration = args.resume_iteration
    stop_iteration = args.stop_iteration
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = 8

    loss_func = get_loss_func(loss_type)
    pretrain = True if pretrained_checkpoint_path else False
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename,
                                   'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
                                   'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
                                   'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename,
                                   'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
                                   'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
                                   'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base),
                                   'statistics.pickle')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename,
                            'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain),
                            'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
                            'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
    create_logging(logs_dir, 'w')
    logging.info(args)

    # Model
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)

    if resume_iteration:
        resume_checkpoint_path = os.path.join(checkpoints_dir, '{}_iterations.pth'.format(resume_iteration))
        logging.info('Load resume model from {}'.format(resume_checkpoint_path))
        resume_checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(resume_checkpoint['model'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = resume_checkpoint['iteration']
    else:
        iteration = 0

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    # Data loader
    train_dataset = CustomAudioDataset(dataset_dir, 'train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomAudioDataset(dataset_dir, 'val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if 'cuda' in device:
        model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                           eps=learning_rate, weight_decay=0., amsgrad=True)

    # if 'mixup' in augmentation:
    #     mixup_augmenter = Mixup(mixup_alpha=1.)

    # Evaluator
    # evaluator = Evaluator(model=model)

    # train_bgn_time = time.time()
    train_losses = []
    val_losses = []
    for epoch in range(stop_iteration):
        # Train on mini batches
        for batch_data_dict in train_loader:
            # Evaluate
            if iteration % 200 == 0 and iteration > 0:
                if resume_iteration > 0 and iteration == resume_iteration:
                    pass
                else:
                    model.eval()  # Set the model to evaluation mode
                    with torch.no_grad():  # Disable gradient calculation during evaluation
                        val_loss = 0.0
                        num_samples = 0
                        for val_data_dict in val_loader:
                            val_data_dict[0] = val_data_dict[0].to(device, dtype=torch.float)
                            val_data_dict[1] = val_data_dict[1].reshape((len(val_data_dict[1]), 1))
                            val_data_dict[1] = val_data_dict[1].to(device)

                            val_output_dict = model(val_data_dict[0])
                            val_loss += loss_func(val_output_dict['clipwise_output'], val_data_dict[1]).item()
                            num_samples += val_data_dict[0].size(0)

                        average_val_loss = val_loss / num_samples
                        # print(f"Epoch {1}, Validation Loss: {average_val_loss}")
                        val_losses.append(average_val_loss)
                        train_losses.append(loss.item())
                    logging.info('------------------------------------')
                    logging.info(
                        f"Iteration: {epoch}. Train Loss: {loss.item()}. Validation Loss: {average_val_loss}".format(
                            iteration))

                    train_fin_time = time.time()

                    # statistics = evaluator.evaluate(val_loader)
                    # logging.info('Validate accuracy: {:.3f}'.format(statistics['accuracy']))
                    #
                    # statistics_container.append(iteration, statistics, 'validate')
                    # statistics_container.dump()
                    #
                    # train_time = train_fin_time - train_bgn_time
                    # validate_time = time.time() - train_fin_time
                    #
                    # logging.info(
                    #     'Train time: {:.3f} s, validate time: {:.3f} s'
                    #     ''.format(train_time, validate_time))
                    #
                    # train_bgn_time = time.time()

            # Save model
            if iteration % 2000 == 0 and iteration > 0:
                checkpoint = {
                    'iteration': iteration,
                    'model': model.module.state_dict()}

                checkpoint_path = os.path.join(checkpoints_dir, f'{epoch}_iterations.pth'.format(iteration))

                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to {}'.format(checkpoint_path))

            # if 'mixup' in augmentation:
            #     batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(len(batch_data_dict))

            batch_data_dict[0] = batch_data_dict[0].to(device, dtype=torch.float)

            # Train
            model.train()

            # if 'mixup' in augmentation:
            #     batch_output_dict = model(batch_data_dict['waveform'],
            #                               batch_data_dict['mixup_lambda'])
            #     """{'clipwise_output': (batch_size, classes_num), ...}"""
            #
            #     batch_target_dict = {'target': do_mixup(batch_data_dict['target'],
            #                                             batch_data_dict['mixup_lambda'])}
            #     """{'target': (batch_size, classes_num)}"""
            # else:
            #     batch_output_dict = model(batch_data_dict['waveform'], None)
            #     """{'clipwise_output': (batch_size, classes_num), ...}"""
            #
            #     batch_target_dict = {'target': batch_data_dict['target']}
            #     """{'target': (batch_size, classes_num)}"""

            batch_output_dict = model(batch_data_dict[0])

            # loss
            batch_data_dict[1] = batch_data_dict[1].reshape((len(batch_output_dict['clipwise_output']), 1))
            batch_data_dict[1] = batch_data_dict[1].to('cuda')
            loss = loss_func(batch_output_dict['clipwise_output'], batch_data_dict[1])

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # Stop learning
            # if iteration == stop_iteration:
            #     break

            iteration += 1

    del train_loader
    del train_dataset
    del val_loader
    del val_dataset
    test_dataset = CustomAudioDataset(dataset_dir, 'test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_calc(model, test_loader, device, train_losses, val_losses)


def test_calc(model, test_loader, device, train_losses, val_losses):
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve
    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to store true labels and predicted probabilities
    true_labels = []
    predicted_probabilities = []

    with torch.no_grad():
        for batch_data_dict in test_loader:
            # Move data to the device
            batch_data_dict[0] = batch_data_dict[0].to(device, dtype=torch.float)

            # Forward pass
            batch_output = model(batch_data_dict[0])

            # Convert true labels to a list and store them
            true_labels.extend(batch_data_dict[1].cpu().numpy())
            predicted_probabilities.extend(batch_output['clipwise_output'].cpu().numpy())

    # Calculate accuracy
    true_labels = [int(label) for label in true_labels]  # Convert true labels to integers
    predicted_labels = [1 if prob >= 0.5 else 0 for prob in predicted_probabilities]
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy}, f1 score: {f1}")

    # Calculate precision and recall for the precision-recall curve
    precision, recall, _ = precision_recall_curve(true_labels, predicted_probabilities)

    # Plot the precision-recall curve
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True)
    plt.show()

    epochs = range(1, len(train_losses) + 1)

    # Create a plot for training losses (in blue)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')

    # Create a plot for validation losses (in red)
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')

    # Add labels and a legend
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                              required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int)
    parser_train.add_argument('--stop_iteration', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        # train(args)
        train_on_decase(args)

    else:
        raise Exception('Error argument!')
