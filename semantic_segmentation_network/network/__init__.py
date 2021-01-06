"""
Network Initializations
"""

import logging
import importlib
import torch


def get_net_ori(args, criterion):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model_ori(network=args.arch,
                    criterion=criterion)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.1f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net

def get_net(args, criterion, criterion2):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(network=args.arch,
                    criterion=criterion, criterion2=criterion2)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.1f}M'.format(num_params / 1000000))

    net = net.cuda()
    return net


def wrap_network_in_dataparallel(net, use_apex_data_parallel=False):
    """
    Wrap the network in Dataparallel
    """
    if use_apex_data_parallel:
        import apex
        net = apex.parallel.DistributedDataParallel(net)
    else:
        net = torch.nn.DataParallel(net)
    return net

def get_model_ori(network, criterion):
    """
    Fetch Network Function Pointer
    """
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(criterion=criterion)
    return net

def get_model(network, criterion, criterion2):
    """
    Fetch Network Function Pointer
    """
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(criterion=criterion, criterion2=criterion2)
    return net
