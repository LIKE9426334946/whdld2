import numpy as np
import torch


class SegmentationMetric:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix[...] = 0

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        for p, t in zip(preds, targets):
            self.confusion_matrix += self._fast_hist(t.flatten(), p.flatten())

    def compute(self):
        hist = self.confusion_matrix.astype(np.float64)
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp

        iou = tp / np.clip(tp + fp + fn, a_min=1e-12, a_max=None)
        pa_per_class = tp / np.clip(hist.sum(axis=1), a_min=1e-12, a_max=None)
        precision_per_class = tp / np.clip(tp + fp, a_min=1e-12, a_max=None)
        recall_per_class = tp / np.clip(tp + fn, a_min=1e-12, a_max=None)

        return {
            "IoU_per_class": iou,
            "PA_per_class": pa_per_class,
            "Precision_per_class": precision_per_class,
            "Recall_per_class": recall_per_class,
            "mIoU": float(np.nanmean(iou)),
            "mPA": float(np.nanmean(pa_per_class)),
            "Precision": float(np.nanmean(precision_per_class)),
            "Recall": float(np.nanmean(recall_per_class)),
        }
