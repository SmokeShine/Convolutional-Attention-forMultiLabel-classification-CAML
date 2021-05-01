import os
import time
import logging
import logging.config
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# https://stackoverflow.com/questions/61524717/pytorch-how-to-find-accuracy-for-multi-label-classification
# This will not work for multi label model


def compute_batch_accuracy(output, target):
    """Computes the accuracy for a batch"""
    with torch.no_grad():

        batch_size = target.size(0)
        output_labels_len = target.size(1)
        # _, pred = output.max(1)
        outputs = torch.sigmoid(output)
        # fixed bug for Classification metrics can't handle a mix of binary and continuous targets
        # fix for F1 always giving 1.0 value
        threshold = 0.2
        outputs[outputs >= threshold] = 1
        # Adding values for 0 as well
        outputs[outputs < threshold] = 0
        # correct = pred.eq(target).sum()
        correct = torch.eq(outputs, target).sum(axis=0).float()/batch_size
        macro_accuracy = correct.sum()/target.size()[1]

        target_flattened = target.flatten()
        outputs_flattened = outputs.flatten()
        soft_outputs = torch.sigmoid(output).cpu().numpy()
        soft_outputs_flattened = torch.sigmoid(output).flatten()
        target_cpu = target.cpu().numpy()
        output_cpu = outputs.cpu().numpy()
        
        micro_accuracy = torch.eq(
            target_flattened, outputs_flattened).float().sum()/outputs_flattened.size()[0]

        # https://stackoverflow.com/questions/55984768/i-am-having-trouble-calculating-the-accuracy-recall-precision-and-f1-score-for
        micro_TP = torch.logical_and(
            target_flattened, outputs_flattened).sum().float()
        micro_TN = torch.logical_and(torch.logical_not(
            target_flattened), torch.logical_not(outputs_flattened)).sum().float()
        micro_FP = torch.logical_and(torch.logical_not(
            target_flattened), (outputs_flattened.long())).sum().float()
        micro_FN = torch.logical_and((target_flattened), torch.logical_not(
            outputs_flattened.long())).sum().float()
        micro_precision = torch.mean(
            micro_TP/(micro_TP+micro_FP+1e-12))  # avoid 0/0 calculation
        # avoid 0/0 calculation
        micro_recall = torch.mean(micro_TP/(micro_TP+micro_FN+1e-12))
        micro_f1 = (2*micro_precision*micro_recall) / \
            (micro_recall+micro_precision)
        # why is this becoming zero? gradient explosion?
        micro_f1=torch.nan_to_num(micro_f1,0.)
        # https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/2
        # https://stackoverflow.com/questions/62265351/measuring-f1-score-for-multiclass-classification-natively-in-pytorch
        # https://cran.r-project.org/web/packages/yardstick/vignettes/multiclass.html
        
        macro_TP = torch.logical_and(target, outputs).sum(axis=0).float()
        macro_TN = torch.logical_and(torch.logical_not(
            target), torch.logical_not(outputs)).sum(axis=0).float()
        macro_FP = torch.logical_and(torch.logical_not(
            target), (outputs)).sum(axis=0).float()
        macro_FN = torch.logical_and(
            (target), torch.logical_not(outputs)).sum(axis=0).float()
        macro_precision = macro_TP / \
            (macro_TP+macro_FP+1e-12)  # avoid 0/0 calculation
        # avoid 0/0 calculation
        macro_recall = macro_TP/(macro_TP+macro_FN+1e-12)
        macro_f1_all_codes = 2*macro_precision * \
            macro_recall/(macro_recall+macro_precision)
        macro_f1_all_codes = macro_f1_all_codes[~torch.isnan(macro_f1_all_codes)]
        macro_f1 = macro_f1_all_codes.mean()
        # if precision+recall=0
        macro_f1=torch.nan_to_num(macro_f1,0.)
        
        # micro_accuracy needs hard labels
        # micro_accuracy = micro_accuracy_loss(target_flattened, outputs_flattened)
        # https://scikit-learn.org/stable/modules/model_evaluation.html#roc-auc-multilabel
        # auc roc need probability score - not hard labels
        micro_auc_roc = roc_auc_score(
            target_flattened.cpu().numpy(), soft_outputs_flattened.cpu().numpy(), average='micro')
        auc_roc_list = []

        for x in range(output.shape[0]):
            auc_roc_list.append(roc_auc_score(
                target_cpu[x, :], soft_outputs[x, :]))
        macro_auc_roc = np.array(auc_roc_list).mean()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # putting it back to gpu
        # micro_f1 = torch.tensor(micro_f1).to(device)
        # macro_f1 = torch.tensor(macro_f1).to(device)
        # micro_accuracy = torch.tensor(micro_accuracy).to(device)
        micro_auc_roc = torch.tensor(micro_auc_roc).to(device)
        macro_auc_roc = torch.tensor(macro_auc_roc).to(device)
        # import pdb;pdb.set_trace()
        return macro_accuracy, micro_f1, macro_f1, micro_accuracy, micro_auc_roc, macro_auc_roc


def train(logger,model, device, data_loader, criterion, optimizer, epoch, print_freq=10):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    micro_f1 = AverageMeter()
    macro_f1 = AverageMeter()
    micro_accuracy = AverageMeter()
    micro_auc_roc = AverageMeter()
    macro_auc_roc = AverageMeter()
    
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if isinstance(input, tuple):
            input = tuple([e.to(device) if type(e) ==
                          torch.Tensor else e for e in input])
        else:
            input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input)
        # import pdb;pdb.set_trace()
        loss = criterion(output, target)
        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        loss.backward()
        # Possible gradient explosion leading to nan in f1 score
        # torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), target.size(0))
        package = compute_batch_accuracy(
            output, target)
        accuracy.update(package[0].item(), target.size(0))
        micro_f1.update(package[1].item(), target.size(0))
        macro_f1.update(package[2].item(), target.size(0))
        micro_accuracy.update(package[3].item(), target.size(0))
        micro_auc_roc.update(package[4].item(), target.size(0))
        macro_auc_roc.update(package[5].item(), target.size(0))
        # accuracy,micro_f1,macro_f1,micro_accuracy,micro_auc_roc,macro_auc_roc
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\n'
                  'Micro_F1 {acc1.val:.3f} ({acc1.avg:.3f})\t'
                  'Macro_F1 {acc2.val:.3f} ({acc2.avg:.3f})\t'
                  'micro_accuracy {acc3.val:.3f} ({acc3.avg:.3f})\t'
                  'Micro_AUC_ROC {acc4.val:.3f} ({acc4.avg:.3f})\t'
                  'Macro_AUC_ROC {acc5.val:.3f} ({acc5.avg:.3f})\n'.format(
                      epoch, i, len(data_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses,
                      acc=accuracy,
                      acc1=micro_f1,
                      acc2=macro_f1,
                      acc3=micro_accuracy,
                      acc4=micro_auc_roc,
                      acc5=macro_auc_roc))
    # accuracy,micro_f1,macro_f1,micro_accuracy,micro_auc_roc,macro_auc_roc
    
    return losses.avg, (accuracy.avg, micro_f1.avg, macro_f1.avg, micro_accuracy.avg, micro_auc_roc.avg, macro_auc_roc.avg)


def evaluate(logger,model, device, data_loader, criterion, print_freq=10,topk=5):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    micro_f1 = AverageMeter()
    macro_f1 = AverageMeter()
    micro_accuracy = AverageMeter()
    micro_auc_roc = AverageMeter()
    macro_auc_roc = AverageMeter()

    results = []

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(data_loader):

            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) ==
                              torch.Tensor else e for e in input])
            else:
                input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), target.size(0))
            package = compute_batch_accuracy(
                output, target)
            accuracy.update(package[0].item(), target.size(0))
            micro_f1.update(package[1].item(), target.size(0))
            macro_f1.update(package[2].item(), target.size(0))
            micro_accuracy.update(package[3].item(), target.size(0))
            micro_auc_roc.update(package[4].item(), target.size(0))
            macro_auc_roc.update(package[5].item(), target.size(0))

            y_true = target.detach().to('cpu').numpy().tolist()
            # need top 5 predictions
            topk_selected=torch.topk(output, topk)
            # [1] only for indexes
            y_pred = topk_selected[1].to('cpu').numpy().tolist()
            results.extend(list(y_pred))

            if i % print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\n'
                      'Micro_F1 {acc1.val:.3f} ({acc1.avg:.3f})\t'
                      'Macro_F1 {acc2.val:.3f} ({acc2.avg:.3f})\t'
                      'micro_accuracy {acc3.val:.3f} ({acc3.avg:.3f})\t'
                      'Micro_AUC_ROC {acc4.val:.3f} ({acc4.avg:.3f})\t'
                      'Macro_AUC_ROC {acc5.val:.3f} ({acc5.avg:.3f}\n)'.format(
                          i, len(data_loader), batch_time=batch_time, loss=losses,
                          acc=accuracy,
                          acc1=micro_f1,
                          acc2=macro_f1,
                          acc3=micro_accuracy,
                          acc4=micro_auc_roc,
                          acc5=macro_auc_roc))

    return losses.avg, (accuracy.avg, micro_f1.avg, macro_f1.avg, micro_accuracy.avg, micro_auc_roc.avg, macro_auc_roc.avg), results
