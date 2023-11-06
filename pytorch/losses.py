import torch
import torch.nn.functional as F


def clip_nll(output_dict, target_dict):
    loss = - torch.mean(target_dict['target'] * output_dict['clipwise_output'])
    return loss


def binary_loss(output_dict, target_dict):
    loss = nn.BCEWithLogitsLoss()
    return loss(output_dict, target_dict)


def get_loss_func(loss_type):
    if loss_type == 'clip_nll':
        return clip_nll
    if loss_type == 'binary_loss':
        return binary_loss
