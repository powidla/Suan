import torch.nn as nn
import torch.nn.functional as F


class Loss():
    def __init__(self):
        pass

    def get_loss_fn(self):
        
        class_loss_fn = self.class_loss_fn

        def loss_fn(outputs, labels, num_items_in_batch=None):
            
            labels = nn.functional.pad(labels, (0, 1), value=-100)
            shift_labels = labels[..., 1:].contiguous()
            loss_mask = shift_labels != -100
            shift_labels[~loss_mask] = 0

            per_token_loss = class_loss_fn(outputs.logits, shift_labels)

            if num_items_in_batch is None:
                num_items_in_batch = loss_mask.sum()
            
            loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
            return loss
        
        return loss_fn
    
    def class_loss_fn(self, logits, labels):
        raise NotImplementedError("Subclasses must implement class_loss!")

