import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import Parameter

from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers.models.luke.modeling_luke import LukePreTrainedModel
from transformers import LukeConfig, LukeModel
class UCTopicConfig(LukeConfig):

    def __init__(
        self,
        vocab_size=50267,
        entity_vocab_size=500000,
        hidden_size=768,
        entity_emb_size=256,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        gradient_checkpointing=False,
        use_entity_aware_attention=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        alpha=1.0,
        temp=0.05,
        **kwargs
    ):
        super().__init__(
            vocab_size,
            entity_vocab_size,
            hidden_size,
            entity_emb_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
            initializer_range,
            layer_norm_eps,
            gradient_checkpointing,
            use_entity_aware_attention,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            **kwargs
        )
        # for contrastive learning
        self.alpha = alpha
        self.temp = temp

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class UCTopicModel(LukePreTrainedModel):
    def __init__(self, model_args, luke_config):
        super().__init__(luke_config)
        self.model_args = model_args
        self.model_name = model_args.model_name_or_path
        self.luke = LukeModel.from_pretrained(self.model_name)
        self.luke_config = luke_config
        self.lm_head = RobertaLMHead(self.luke_config)

        self.mlp = MLPLayer(self.luke_config)
        self.sim = Similarity(temp=self.model_args.temp)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        entity_ids=None,
        entity_attention_mask=None,
        entity_token_type_ids=None,
        entity_position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        phrase_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if phrase_emb:
            return self.phremb_forward(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                entity_ids=entity_ids,
                entity_attention_mask=entity_attention_mask,
                entity_token_type_ids=entity_token_type_ids,
                entity_position_ids=entity_position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return self.cl_forward(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                entity_ids=entity_ids,
                entity_attention_mask=entity_attention_mask,
                entity_token_type_ids=entity_token_type_ids,
                entity_position_ids=entity_position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )

    def cl_forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        entity_ids=None,
        entity_attention_mask=None,
        entity_token_type_ids=None,
        entity_position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        mlm_input_ids=None,
        mlm_labels=None,
    ):  

        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)
        entity_length = entity_ids.size(2)
        max_mention_length = entity_position_ids.size(-1)
        mlm_outputs = None
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
        entity_ids = entity_ids.view(-1, entity_length)
        entity_attention_mask = entity_attention_mask.view(-1, entity_length)
        entity_position_ids = entity_position_ids.view(-1, entity_length, max_mention_length)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
        
        if entity_token_type_ids is not None:
            entity_token_type_ids = entity_token_type_ids.view(-1, entity_length)

        # Get raw embeddings
        outputs = self.luke(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # MLM auxiliary objective
        if mlm_input_ids is not None:
            mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
            mlm_outputs = self.luke(
                mlm_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        entity_pooler = outputs['entity_last_hidden_state'] # (bs*num_sent, entity_length, hidden_size)
        entity_pooler = entity_pooler.view((batch_size, num_sent, entity_pooler.size(-1))) # (bs, num_sent, hidden) entity_length should be 1

        entity_pooler = self.mlp(entity_pooler)
        # # Separate representation
        z1, z2 = entity_pooler[:,0], entity_pooler[:,1]

        # # Gather all embeddings if using distributed training
        if dist.is_initialized() and self.training:
            
            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)

        correct_num = (torch.argmax(cos_sim, 1) == labels).sum().detach().cpu().item()
        
        # Calculate loss for MLM
        if mlm_outputs is not None and mlm_labels is not None:
            mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
            prediction_scores = self.lm_head(mlm_outputs.last_hidden_state)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.luke_config.vocab_size), mlm_labels.view(-1))
            loss = loss + self.model_args.mlm_weight * masked_lm_loss

        return {'loss': loss, 'logits': cos_sim, 'correct_num': correct_num, 'total_num': batch_size}

    def phremb_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        entity_ids=None,
        entity_attention_mask=None,
        entity_token_type_ids=None,
        entity_position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.luke_config.use_return_dict

        outputs = self.luke(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs


class UCTopic(LukePreTrainedModel):
    config_class = UCTopicConfig
    def __init__(self, config):
        super().__init__(config)
        self.luke = LukeModel(config)
        self.config = config
        self.mlp = MLPLayer(self.config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        entity_ids=None,
        entity_attention_mask=None,
        entity_token_type_ids=None,
        entity_position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):

        outputs = self.luke(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if return_dict:
            entity_pooler = outputs['entity_last_hidden_state'] # (bs, entity_length, hidden_size)
        else:
            entity_pooler = outputs.entity_last_hidden_state
        entity_pooler = self.mlp(entity_pooler)
        return outputs, entity_pooler.squeeze()

class UCTopicCluster(LukePreTrainedModel):
    config_class = UCTopicConfig
    def __init__(self, config, cluster_centers=None):
        super().__init__(config)
        self.luke = LukeModel(config)
        self.config = config
        self.mlp = MLPLayer(self.config)
        self.alpha = self.config.alpha
        self.sim = Similarity(temp=self.config.temp)
        self.softmax = nn.Softmax(dim=-1)
        # Instance-CL head
        
        if cluster_centers is not None:

            initial_cluster_centers = torch.tensor(
                cluster_centers, dtype=torch.float, requires_grad=True)
            self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        entity_ids=None,
        entity_attention_mask=None,
        entity_token_type_ids=None,
        entity_position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):

        outputs = self.luke(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if return_dict:
            entity_pooler = outputs['entity_last_hidden_state'] # (bs, entity_length, hidden_size)
        else:
            entity_pooler = outputs.entity_last_hidden_state
        entity_pooler = self.mlp(entity_pooler)
        return outputs, entity_pooler.squeeze()

    def get_cl_loss(self, anchor_embd, cl_embd):

        batch_size, hidden_size = anchor_embd.size()
        anchor_embd = anchor_embd.unsqueeze(1) ##(batch, 1, hidden_size)
        cl_embd = cl_embd.view([batch_size, -1, hidden_size])

        cos_sim = self.sim(anchor_embd, cl_embd) ##(batch, class_num)
        label_size = cos_sim.size(0)
        labels = torch.zeros(label_size, device=anchor_embd.device, dtype=torch.long) # (batch_size)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)
        
        return loss

    def get_cluster_prob(self, embeddings):
        cos = self.sim(embeddings.unsqueeze(1), self.cluster_centers.unsqueeze(0))
        return self.softmax(cos)

    def update_cluster_centers(self, cluster_centers):
        
        initial_cluster_centers = torch.tensor(
                cluster_centers, dtype=torch.float, requires_grad=True, device=self.luke.device)
        self.cluster_centers = Parameter(initial_cluster_centers)
