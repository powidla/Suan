import torch.nn.functional as F


class Loss():
    def __init__(self):
        pass

    def get_loss_fn(self, split=False):
        
        if split:
            class_loss_fn = self.class_loss_split_fn
        else:
            class_loss_fn = self.class_loss_fn
        
        return class_loss_fn
    
    def class_loss_fn(self, *args):
        raise NotImplementedError("Subclasses must implement class_loss!")
    
    def class_loss_split_fn(self, *args):
        raise NotImplementedError("Subclasses must implement class_loss_split!")
    

### https://arxiv.org/abs/2305.18290
class DPOLoss(Loss):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def class_loss_fn(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps):
        
        loss = -F.logsigmoid(self.beta * (chosen_logps - rejected_logps - ref_chosen_logps + ref_rejected_logps))

        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


class SafeDPOLoss(Loss):
    def __init__(self, beta=0.1, delta=10.0):
        super().__init__()
        self.beta = beta
        self.delta = delta

    def class_loss_fn(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, chosen_is_better, rejected_is_better):
        safety_mu = self.delta * (rejected_is_better - chosen_is_better)
        loss = -F.logsigmoid(self.beta * (chosen_logps - rejected_logps - ref_chosen_logps + ref_rejected_logps - safety_mu))
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards

### https://arxiv.org/abs/2310.12036
class IPOLoss(Loss):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def class_loss_fn(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps):
        
        delta = chosen_logps - rejected_logps - ref_chosen_logps + ref_rejected_logps - 0.5 / self.tau
        loss = delta * delta

        chosen_rewards = (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = (rejected_logps - ref_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards
    

class SiULoss(Loss):
    def __init__(self, beta=0.1, tau=1.0):
        super().__init__()
        self.beta = beta
        self.tau = tau

    def class_loss_fn(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps):

        sup_loss = - F.logsigmoid(chosen_logps - rejected_logps + self.tau)

        unsup_loss = (F.sigmoid(chosen_logps - ref_chosen_logps) - 0.5).detach()*chosen_logps + (F.sigmoid(rejected_logps - ref_rejected_logps) - 0.5).detach()*rejected_logps

        loss = sup_loss + self.beta * unsup_loss

        chosen_rewards = (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = (rejected_logps - ref_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards
    
