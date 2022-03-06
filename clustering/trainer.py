import torch
import torch.nn as nn
from torch.nn import functional as F
from .consts import TOKENIZER

eps = 1e-8  

class KLDiv(nn.Module):    
    def forward(self, predict, target):
        assert predict.ndimension()==2,'Input dimension must be 2'
        target = target.detach()
        p1 = predict + eps
        t1 = target + eps
        logI = p1.log()
        logT = t1.log()
        TlogTdI = target * (logT - logI)
        kld = TlogTdI.sum(1)
        return kld

class KCL(nn.Module):
    def __init__(self):
        super(KCL,self).__init__()
        self.kld = KLDiv()

    def forward(self, prob1, prob2):
        kld = self.kld(prob1, prob2)
        return kld.mean()

def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
    return (weight.t() / torch.sum(weight, 1)).t()


class ClusterLearner(nn.Module):
	def __init__(self, model, optimizer):
		super(ClusterLearner, self).__init__()
		self.model = model
		self.optimizer = optimizer

	def forward(self, inputs):
		anchor_batch, cl_batch = inputs
		
		_, anchor_embd = self.model(**anchor_batch)  #anchor
		_, cl_embd = self.model(**cl_batch)	 #positive

		# Instance-CL loss
		contrastive_loss = self.model.get_cl_loss(anchor_embd, cl_embd)
		loss = contrastive_loss
				
		loss.backward()
		self.optimizer.step()
		self.optimizer.zero_grad()
		return {"Instance-CL_loss":contrastive_loss.detach().cpu().item()}
