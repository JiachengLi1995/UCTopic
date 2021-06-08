import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers import LukeConfig, LukeModel

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

def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
    encoder,
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
    outputs = encoder(
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
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    entity_pooler = outputs['entity_last_hidden_state '] # (bs*num_sent, entity_length, hidden_size)
    entity_pooler = entity_pooler.view((batch_size, num_sent, entity_pooler.size(-1))) # (bs, num_sent, hidden) entity_length should be 1

    entity_pooler = cls.mlp(entity_pooler)

    # Separate representation
    z1, z2 = entity_pooler[:,0], entity_pooler[:,1]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        
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

    cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))    
    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)

    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(cos_sim, labels)

    correct_num = (torch.argmax(cos_sim, 1) == labels).sum().detach().cpu().item()

    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.luke_config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    return {'loss': loss, 'logits': cos_sim, 'correct_num': correct_num, 'total_num': batch_size, 'model_output': outputs}
    

def phremb_forward(
    cls,
    encoder,
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

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
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

class UCTopicModel(nn.Module):
    #_keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args
        self.model_name = model_args.model_name_or_path
        self.luke = LukeModel.from_pretrained(self.model_name)
        self.luke_config = LukeConfig.from_pretrained(self.model_name)
        self.lm_head = RobertaLMHead(self.luke_config)
        if 'base' in self.model_name:
            lm_ckpt = torch.load('./luke_lmhead/luke_base_lmhead.bin')
        else:
            lm_ckpt = torch.load('./luke_lmhead/luke_large_lmhead.bin')

        lm_model_state = self.lm_head.state_dict()
        for name, param in lm_ckpt.items():
            if name.startswith('lm_head.'):
                name = name[len('lm_head.'):]
            lm_model_state[name].copy_(param)

        cl_init(self, self.luke_config)

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
            return phremb_forward(self, self.luke,
                input_ids=input_ids,
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
            return cl_forward(self, self.luke,
                input_ids=input_ids,
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
