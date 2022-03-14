import torch
import torch.nn as nn


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