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
		self.cluster_loss = nn.KLDivLoss(reduction='sum')
		self.kcl = KCL()

	def forward(self, inputs, use_perturbation=False):
		batch1, batch2, batch3 = inputs
		
		_, embd0 = self.model(**batch1)
		_, embd1 = self.model(**batch2)
		_, embd2 = self.model(**batch3)

		# Instance-CL loss
		feat1 = self.model.head(embd1)
		feat2 = self.model.head(embd2)
		contrastive_loss = self.model.get_cl_loss(feat1, feat2)
		loss = contrastive_loss

        # clustering loss
		output = self.model.get_cluster_prob(embd0)
		target = target_distribution(output).detach()
		cluster_loss = self.cluster_loss((output+1e-08).log(),target)/output.shape[0]
		loss += cluster_loss

		# consistency loss (this loss is used in the experiments of our NAACL paper, we included it here just in case it might be helpful for your specific applications)
		local_consloss_val = 0
		if use_perturbation:
			local_consloss = self.model.local_consistency(embd0, embd1, embd2, self.kcl)
			loss += local_consloss
			local_consloss_val = local_consloss.detach().cpu().item()
				
		loss.backward()
		self.optimizer.step()
		self.optimizer.zero_grad()
		return {"Instance-CL_loss":contrastive_loss.detach().cpu().item(), "clustering_loss":cluster_loss.detach().cpu().item(), "local_consistency_loss":local_consloss_val}
