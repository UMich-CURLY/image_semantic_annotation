"""
Loss.py
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


def get_loss(args, data_type):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    return: criterion, criterion_val
    """
#    task_weights = nn.Parameter(torch.ones(2, requires_grad=True).cuda())
#    task_weights = torch.ones(2, requires_grad=True)

    if args.img_wt_loss:
        if data_type=='semantic':
            criterion = ImageBasedCrossEntropyLoss2d_semantic(
                classes=args.dataset_cls.num_classes1, size_average=True,
                ignore_index=args.dataset_cls.ignore_label,
                upper_bound=args.wt_bound).cuda()
        elif data_type=='trav':
            criterion = ImageBasedCrossEntropyLoss2d_trav(
                classes=args.dataset_cls.num_classes2, size_average=True,
                ignore_index=args.dataset_cls.ignore_label,
                upper_bound=args.wt_bound).cuda()
        elif data_type=='trav_alone':
            criterion = ImageBasedCrossEntropyLoss2d_trav_alone(
                classes=args.dataset_cls.num_classes, size_average=True,
                ignore_index=args.dataset_cls.ignore_label,
                upper_bound=args.wt_bound).cuda()

    elif args.jointwtborder:
        criterion = ImgWtLossSoftNLL(classes=args.dataset_cls.num_classes,
                                     ignore_index=args.dataset_cls.ignore_label,
                                     upper_bound=args.wt_bound).cuda()
    else:
        criterion = CrossEntropyLoss2d(size_average=True,
                                       ignore_index=args.dataset_cls.ignore_label).cuda()

    criterion_val = CrossEntropyLoss2d(size_average=True,
                                       weight=None,
                                       ignore_index=args.dataset_cls.ignore_label).cuda()
    return criterion, criterion_val

class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING
        self.task_weights = nn.Parameter(torch.ones(1, requires_grad=True))
        #self.wght = torch.ones(21)

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):

        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculate_weights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        loss1 = []
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()

            nll = self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),targets[i].unsqueeze(0))
#            loss += loss1/torch.exp(self.task_weights) + 0.5*self.task_weights
            loss1.append(nll)            
        return torch.mean(torch.stack(loss1))/torch.exp(self.task_weights) + 0.5*self.task_weights

class ImageBasedCrossEntropyLoss2d_semantic(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d_semantic, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.kldiv_loss = nn.KLDivLoss(size_average)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING
        #self.task_weights = nn.Parameter(torch.ones(1, requires_grad=True))
        #self.wght = torch.ones(21)

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]

#        hist = ((hist != 0) * self.upper_bound * (1 - hist))
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):

        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculate_weights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        loss2 = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
#                tgt_cpu = target_cpu[i]
#                tgt_cpu[tgt_cpu>18] = 255
#                tgt_cpu[tgt_cpu<0] = 255
#                weights = self.calculate_weights(tgt_cpu)
                weights = self.calculate_weights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()


            nll = self.nll_loss(F.log_softmax(inputs[i].narrow(0,0,19).unsqueeze(0), dim=1),targets[i].unsqueeze(0))
#            loss1.append(nll)            

            trav = torch.sum(torch.cat([inputs[i].narrow(0,0,2),inputs[i][9,:,:].unsqueeze(0)], 0),0)
            untrav = torch.sum(torch.cat([inputs[i].narrow(0,2,7),inputs[i].narrow(0,10,9)], 0),0)
            softmax1 = F.softmax(torch.cat([untrav.unsqueeze(0),trav.unsqueeze(0)], 0).unsqueeze(0), dim=1)
            softmax2 = F.log_softmax(inputs[i].narrow(0,19,2).unsqueeze(0), dim=1)
            kld1 = self.kldiv_loss(softmax2,softmax1)
            softmax1 = F.log_softmax(torch.cat([untrav.unsqueeze(0),trav.unsqueeze(0)], 0).unsqueeze(0), dim=1)
            softmax2 = F.softmax(inputs[i].narrow(0,19,2).unsqueeze(0), dim=1)
            kld2 = self.kldiv_loss(softmax1,softmax2)
            loss += nll #+ kld
            loss2 += kld2

#        loss = torch.mean(torch.stack(loss1))/torch.exp(self.task_weights[0]) + 0.5*self.task_weights[0] #+ torch.mean(torch.stack(loss2))/torch.exp(self.task_weights[1]) + 0.5*self.task_weights[1]
        return loss, loss2

class ImageBasedCrossEntropyLoss2d_trav(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d_trav, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.kldiv_loss = nn.KLDivLoss(size_average)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING
        #self.task_weights = nn.Parameter(torch.ones(1, requires_grad=True))
        #self.wght = torch.ones(21)

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]

#        hist = ((hist != 0) * self.upper_bound * (1 - hist))
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):

        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculate_weights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        loss2 = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
#                tgt_cpu = target_cpu[i]
#                tgt_cpu[tgt_cpu>1] = 255
#                tgt_cpu[tgt_cpu<0] = 255
#                weights = self.calculate_weights(tgt_cpu)
                weights = self.calculate_weights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()


            nll = self.nll_loss(F.log_softmax(inputs[i].narrow(0,19,2).unsqueeze(0), dim=1),targets[i].unsqueeze(0))
#            loss1.append(nll)            

            trav = torch.sum(torch.cat([inputs[i].narrow(0,0,2),inputs[i][9,:,:].unsqueeze(0)], 0),0)
            untrav = torch.sum(torch.cat([inputs[i].narrow(0,2,7),inputs[i].narrow(0,10,9)], 0),0)
            softmax1 = F.softmax(torch.cat([untrav.unsqueeze(0),trav.unsqueeze(0)], 0).unsqueeze(0), dim=1)
            softmax2 = F.log_softmax(inputs[i].narrow(0,19,2).unsqueeze(0), dim=1)
            kld1 = self.kldiv_loss(softmax2,softmax1)
            softmax1 = F.log_softmax(torch.cat([untrav.unsqueeze(0),trav.unsqueeze(0)], 0).unsqueeze(0), dim=1)
            softmax2 = F.softmax(inputs[i].narrow(0,19,2).unsqueeze(0), dim=1)
            kld2 = self.kldiv_loss(softmax1,softmax2)
            loss += nll #+ kld
            loss2 += kld2

#        loss = torch.mean(torch.stack(loss1))/torch.exp(self.task_weights[0]) + 0.5*self.task_weights[0] #+ torch.mean(torch.stack(loss2))/torch.exp(self.task_weights[1]) + 0.5*self.task_weights[1]
        return loss, loss2

class ImageBasedCrossEntropyLoss2d_trav_alone(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d_trav_alone, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.kldiv_loss = nn.KLDivLoss(size_average)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING
        #self.task_weights = nn.Parameter(torch.ones(1, requires_grad=True))
        #self.wght = torch.ones(21)

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]

#        hist = ((hist != 0) * self.upper_bound * (1 - hist))
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):

        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculate_weights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()

            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),targets[i].unsqueeze(0))
        return loss

class ImageBasedCrossEntropyLoss2d_old(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.kldiv_loss = nn.KLDivLoss(size_average)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = cfg.BATCH_WEIGHTING
        self.task_weights = nn.Parameter(torch.zeros(2, requires_grad=True))
        #self.wght = torch.ones(21)

    def calculate_weights(self, target, n_class):
        """
        Calculate weights of classes based on the training crop
        """
        hist = np.histogram(target.flatten(), range(
            n_class + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):

        target_cpu = targets.data.cpu().numpy()
        #task_cpu = self.task_weights.data.cpu().numpy()
        #print(task_cpu)
        if self.batch_weights:
            weights = self.calculate_weights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        loss1 = []
        loss2 = []
        loss3 = []
        loss4 = []
        task_n = 0
        for i in range(0, inputs.shape[0]):
            if target_cpu[i].min()<19:
                task_n = 0
                weights = self.calculate_weights(target_cpu[i], 19)
#                print(weights)
                self.nll_loss.weight = torch.Tensor(weights).cuda()
                softmax = F.log_softmax(inputs[i].narrow(0,0,19).unsqueeze(0), dim=1)
                tgt = targets[i].unsqueeze(0)
                nll = self.nll_loss(softmax,tgt)
                loss1.append(nll)
            else:
                task_n = 1
                tgt_cpu = target_cpu[i] - 19
                tgt_cpu[tgt_cpu>1] = 255
                tgt_cpu[tgt_cpu<0] = 255
                weights = self.calculate_weights(tgt_cpu, 2)
#                print(weights)
                self.nll_loss.weight = torch.Tensor(weights).cuda()
                softmax = F.log_softmax(inputs[i].narrow(0,19,2).unsqueeze(0), dim=1)
                tgt = targets[i]-19
                tgt[tgt>10] = 255
                tgt = tgt.unsqueeze(0)
                nll = self.nll_loss(softmax,tgt)
                loss2.append(nll)

#            nll = self.nll_loss(softmax,tgt)
#            if task_n ==0:
#                loss += nll/torch.exp(self.task_weights[0]) + 0.5*self.task_weights[0] 
#            else:
#                loss += nll/torch.exp(self.task_weights[1]) + 0.5*self.task_weights[1] 

            trav = torch.sum(torch.cat([inputs[i].narrow(0,0,2),inputs[i][9,:,:].unsqueeze(0)], 0),0)
            untrav = torch.sum(torch.cat([inputs[i].narrow(0,2,7),inputs[i].narrow(0,10,9)], 0),0)
            softmax1 = F.softmax(torch.cat([untrav.unsqueeze(0),trav.unsqueeze(0)], 0).unsqueeze(0), dim=1)
            softmax2 = F.log_softmax(inputs[i].narrow(0,19,2).unsqueeze(0), dim=1)
            kld = self.kldiv_loss(softmax2,softmax1)
            loss3.append(kld)
            softmax1 = F.log_softmax(torch.cat([untrav.unsqueeze(0),trav.unsqueeze(0)], 0).unsqueeze(0), dim=1)
            softmax2 = F.softmax(inputs[i].narrow(0,19,2).unsqueeze(0), dim=1)
            kld1 = self.kldiv_loss(softmax1,softmax2)
            loss4.append(kld1)

#            if task_n ==0:
##                loss1.append(nll)
#                softmax1 = F.softmax(torch.cat([untrav.unsqueeze(0),trav.unsqueeze(0)], 0).unsqueeze(0), dim=1)
#                softmax2 = F.log_softmax(inputs[i].narrow(0,19,2).unsqueeze(0), dim=1)
#                kld2 = self.kldiv_loss(softmax2,softmax1)
#                loss3.append(kld2)
#            else:
###                loss2.append(nll)
#                softmax1 = F.log_softmax(torch.cat([untrav.unsqueeze(0),trav.unsqueeze(0)], 0).unsqueeze(0), dim=1)
#                softmax2 = F.softmax(inputs[i].narrow(0,19,2).unsqueeze(0), dim=1)
#                kld1 = self.kldiv_loss(softmax1,softmax2)
#                loss4.append(kld1)
               
        task_weight3 = torch.exp(self.task_weights[0])+torch.exp(self.task_weights[1])
        if len(loss1)>0: 
            loss += torch.mean(torch.stack(loss1))/torch.exp(self.task_weights[0]) + 0.5*self.task_weights[0] #+ torch.mean(torch.stack(loss3))/torch.exp(self.task_weights[1]) + 0.5*self.task_weights[1]
#            print('loss1:', torch.mean(torch.stack(loss3)))
        if len(loss2)>0:
            loss += torch.mean(torch.stack(loss2))/torch.exp(self.task_weights[1]) + 0.5*self.task_weights[1] #+ torch.mean(torch.stack(loss4))/torch.exp(self.task_weights[0]) + 0.5*self.task_weights[0]
#            print('loss2:',torch.mean(torch.stack(loss4)))
        loss += torch.mean(torch.stack(loss3+loss4))/task_weight3
        return loss



class CrossEntropyLoss2d(nn.Module):
    """
    Cross Entroply NLL Loss
    """

    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        #self.nll_loss1 = nn.NLLLoss2d(weight, size_average, ignore_index)
        #self.nll_loss2 = nn.NLLLoss2d(weight, size_average, ignore_index)
        # self.weight = weight

    def forward(self, inputs, targets):
#        softmax1 = F.log_softmax(inputs.narrow(1,0,19),dim=1)
#        softmax2 = F.log_softmax(inputs.narrow(1,19,2),dim=1)
#        softmax = torch.cat((softmax1, softmax2), 1)

#        target_cpu = targets.data.cpu().numpy()
#        if target_cpu[i].min()<19:
#            l1 = self.nll_loss1(softmax1, targets)
#        else:
#            l2 = self.nll_loss2(softmax2, targets)
#            print("l2:", l2)

        return self.nll_loss(F.log_softmax(inputs), targets)


def customsoftmax(inp, multihotmask):
    """
    Custom Softmax
    """
    soft = F.softmax(inp)
    # This takes the mask * softmax ( sums it up hence summing up the classes in border
    # then takes of summed up version vs no summed version
    return torch.log(
        torch.max(soft, (multihotmask * (soft * multihotmask).sum(1, keepdim=True)))
    )

class ImgWtLossSoftNLL(nn.Module):
    """
    Relax Loss
    """

    def __init__(self, classes, ignore_index=255, weights=None, upper_bound=1.0,
                 norm=False):
        super(ImgWtLossSoftNLL, self).__init__()
        self.weights = weights
        self.num_classes = classes
        self.ignore_index = ignore_index
        self.upper_bound = upper_bound
        self.norm = norm
        self.batch_weights = cfg.BATCH_WEIGHTING
        self.fp16 = False


    def calculate_weights(self, target):
        """
        Calculate weights of the classes based on training crop
        """
        if len(target.shape) == 3:
            hist = np.sum(target, axis=(1, 2)) * 1.0 / target.sum()
        else:
            hist = np.sum(target, axis=(0, 2, 3)) * 1.0 / target.sum()
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist[:-1]

    def custom_nll(self, inputs, target, class_weights, border_weights, mask):
        """
        NLL Relaxed Loss Implementation
        """
        if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
            border_weights = 1 / border_weights
            target[target > 1] = 1
        if self.fp16:
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].half() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :-1, :, :].half())).sum(1)) * \
                          (1. - mask.half())
        else:
            loss_matrix = (-1 / border_weights *
                           (target[:, :-1, :, :].float() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :-1, :, :].float())).sum(1)) * \
                          (1. - mask.float())

            # loss_matrix[border_weights > 1] = 0
        loss = loss_matrix.sum()

        # +1 to prevent division by 0
        loss = loss / (target.shape[0] * target.shape[2] * target.shape[3] - mask.sum().item() + 1)
        return loss

    def forward(self, inputs, target):
        if self.fp16:
            weights = target[:, :-1, :, :].sum(1).half()
        else:
            weights = target[:, :-1, :, :].sum(1).float()
        ignore_mask = (weights == 0)
        weights[ignore_mask] = 1

        loss = 0
        target_cpu = target.data.cpu().numpy()

        if self.batch_weights:
            class_weights = self.calculate_weights(target_cpu)

        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                class_weights = self.calculate_weights(target_cpu[i])
            # loss = loss + self.custom_nll(inputs[i].unsqueeze(0),
            #                              target[i].unsqueeze(0),
            #                              class_weights=torch.Tensor(class_weights).cuda(),
            #                              border_weights=weights, mask=ignore_mask[i])
            loss = loss + self.custom_nll(inputs[i].unsqueeze(0),
                                          target[i].unsqueeze(0),
                                          class_weights=None,
                                          border_weights=weights, mask=ignore_mask[i])

        return loss
