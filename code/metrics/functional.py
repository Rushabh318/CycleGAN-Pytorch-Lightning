import torch

def true_positives(pred, gt, threshold=0.5):
    p = torch.where(pred > threshold, 1, 0)
    tp = torch.sum(p * gt)
    return int(tp)

def false_positives(pred, gt, threshold=0.5):
    p = torch.where(pred > threshold, 1, 0)
    fp = torch.sum(p * torch.abs(gt-1))
    return int(fp)

def true_negatives(pred, gt, threshold=0.5):
    n = torch.where(pred < threshold, 1, 0)
    tn = torch.sum(n * torch.abs(gt-1))
    return int(tn)

def false_negatives(pred, gt, threshold=0.5):
    n = torch.where(pred < threshold, 1, 0)
    fn = torch.sum(n * gt)
    return int(fn)

def precision(pred, gt, threshold=0.5):
    tp = true_positives(pred, gt)
    fp = false_positives(pred, gt)
    
    if (tp + fp) == 0:
        return float('nan')
    
    return float(tp / (tp + fp))

def recall(pred, gt, threshold=0.5):
    tp = true_positives(pred, gt)
    fn = false_negatives(pred, gt)

    if (tp + fn) == 0:
        return float('nan')

    return float(tp / (tp + fn))


def iou(pred, gt, threshold=0.5):
    tp = true_positives(pred, gt, threshold)
    fp = false_positives(pred, gt, threshold)
    fn = false_negatives(pred, gt, threshold)

    intersection = tp
    union = tp + fp + fn

    if union == 0:
        return float('nan')
    
    return float(intersection / union)


def dice(pred, gt, threshold=0.5):
    tp = true_positives(pred, gt, threshold)
    fp = false_positives(pred, gt, threshold)
    fn = false_negatives(pred, gt, threshold)

    intersection = 2 * tp
    union = 2 * tp + fp + fn

    if union == 0:
        return float('nan')

    return float(intersection / union)


def accuracy(pred, gt, threshold=0.5):
    tp = true_positives(pred, gt, threshold)
    tn = true_negatives(pred, gt, threshold)
    fp = false_positives(pred, gt, threshold)
    fn = false_negatives(pred, gt, threshold)
    
    if (tp + tn + fp + fn) == 0:
        return 0
    
    acc = (tp + tn)/(tp + tn + fp + fn)

    return float(acc)