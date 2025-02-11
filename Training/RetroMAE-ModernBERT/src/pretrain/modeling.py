import logging
import os

import torch
from torch import nn
from transformers import ModernBertForMaskedLM, AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

from .arguments import ModelArguments
from .enhancedDecoder import BertLayerForDecoder

logger = logging.getLogger(__name__)


class RetroMAEForPretraining(nn.Module):
    def __init__(
            self,
            modernbert: ModernBertForMaskedLM,
            model_args: ModelArguments,
    ):
        super(RetroMAEForPretraining, self).__init__()
        self.lm = modernbert

        self.decoder_embeddings = self.lm.model.embeddings

        print(modernbert.config)
        self.c_head = BertLayerForDecoder(modernbert.config)
        self.c_head.apply(self.lm._init_weights)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args

    def gradient_checkpointing_enable(self, **kwargs):
        self.lm.gradient_checkpointing_enable(**kwargs)

    def forward(self,
                encoder_input_ids, input_ids_length, encoder_attention_mask, encoder_labels,
                decoder_input_ids, decoder_attention_mask, decoder_labels):

        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids, encoder_attention_mask,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True
        )
        
        all_hiddens = lm_out.hidden_states[-1]
    
        batch_size  = encoder_input_ids.size(0)
        max_len     = encoder_input_ids.size(1)
        hidden_size = all_hiddens.size(-1)

        padded_hiddens = all_hiddens.new_zeros((batch_size, max_len, hidden_size))
        
        offset = 0
        for i in range(batch_size):
            length = int(input_ids_length[i])  # number of valid tokens
            padded_hiddens[i, :length, :] = all_hiddens[offset:offset + length]
            offset += length
        
        def average_pool(last_hidden_states, attention_mask) :
            masked_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            sum_hidden = masked_hidden.sum(dim=1)
            valid_counts = attention_mask.sum(dim=1, keepdim=True)

            return sum_hidden / valid_counts
    
        mean_hiddens_per_example = average_pool(padded_hiddens, encoder_attention_mask)
        mean_hiddens = mean_hiddens_per_example.unsqueeze(1)
        decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
        hiddens = torch.cat([mean_hiddens, decoder_embedding_output[:, 1:]], dim=1)
        
        mean_expanded = mean_hiddens.expand(hiddens.size(0), hiddens.size(1), hiddens.size(2))
        query = self.decoder_embeddings(inputs_embeds=mean_expanded)
        
        matrix_attention_mask = self.lm.get_extended_attention_mask(
            decoder_attention_mask,
            decoder_attention_mask.shape,
            decoder_attention_mask.device
        )

        hiddens = self.c_head(query=query,
                              key=hiddens,
                              value=hiddens,
                              attention_mask=matrix_attention_mask)[0]
        pred_scores, loss = self.mlm_loss(hiddens, decoder_labels)

        return (loss + lm_out.loss,)

    def mlm_loss(self, hiddens, labels):
        if hasattr(self.lm, 'cls'):
            pred_scores = self.lm.cls(hiddens)
        elif hasattr(self.lm, 'decoder'):
            pred_scores = self.lm.decoder(hiddens)
        else:
            raise NotImplementedError

        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return pred_scores, masked_lm_loss

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(os.path.join(output_dir, "encoder_model"))
        torch.save(self.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args)
        return model