from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.add_state("tp", default=torch.zeros(num_classes, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)

        for class_idx in range(self.num_classes):
            pred_is_class = preds == class_idx
            target_is_class = target == class_idx

            self.tp[class_idx] += torch.sum(pred_is_class & target_is_class)
            self.fp[class_idx] += torch.sum(pred_is_class & ~target_is_class)
            self.fn[class_idx] += torch.sum(~pred_is_class & target_is_class)

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        support = self.tp + self.fn
        weighted_f1 = ((support * f1).sum() / (support.sum() + 1e-8))

        return weighted_f1

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError(f"[Shape Mismatch] Preds.shape is {preds.shape} but target.shape is {target.shape}")

        # [TODO] Cound the number of correct prediction
        correct = (preds == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
