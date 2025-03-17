# import logging

# import torch
# import torch.nn.functional as F
# from .arguments import ModelArguments
# from .enhancedDecoder import BertLayerForDecoder
# from torch import nn
# from transformers import RobertaForMaskedLM, AutoModelForMaskedLM
# from transformers.modeling_outputs import MaskedLMOutput

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForMaskedLM, ModernBertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

from .arguments import ModelArguments
from .enhancedDecoder import BertLayerForDecoder

logger = logging.getLogger(__name__)


class DupMAEForPretraining(nn.Module):
    def __init__(
        self,
        modernbert: ModernBertForMaskedLM,
        model_args: ModelArguments,
    ):
        super(DupMAEForPretraining, self).__init__()
        self.lm = modernbert

        self.decoder_embeddings = self.lm.model.embeddings

        self.c_head = BertLayerForDecoder(modernbert.config)
        self.c_head.apply(self.lm._init_weights)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args

    def decoder_mlm_loss(self, sentence_embedding, decoder_input_ids, decoder_attention_mask, decoder_labels):
        sentence_embedding = sentence_embedding.view(decoder_input_ids.size(0), 1, -1)
        decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)

        hiddens = torch.cat([sentence_embedding, decoder_embedding_output[:, 1:]], dim=1)

        query_embedding = sentence_embedding.expand(hiddens.size(0), hiddens.size(1), hiddens.size(2))
        query = self.decoder_embeddings(inputs_embeds=query_embedding)

        matrix_attention_mask = self.lm.get_extended_attention_mask(
            decoder_attention_mask, decoder_attention_mask.shape, decoder_attention_mask.device
        )

        hiddens = self.c_head(query=query, key=hiddens, value=hiddens, attention_mask=matrix_attention_mask)[0]
        pred_scores, loss = self.mlm_loss(hiddens, decoder_labels)

        return loss

    def ot_embedding(self, logits, attention_mask):
        mask = (1 - attention_mask.unsqueeze(-1)) * -1000
        reps, _ = torch.max(logits + mask, dim=1)  # B V
        return reps

    def decoder_ot_loss(self, ot_embedding, bag_word_weight):
        input = F.log_softmax(ot_embedding, dim=-1)
        bow_loss = torch.mean(-torch.sum(bag_word_weight * input, dim=1))
        return bow_loss

    def forward(
        self,
        encoder_input_ids,
        input_ids_length,
        encoder_attention_mask,
        encoder_labels,
        decoder_input_ids,
        decoder_attention_mask,
        decoder_labels,
        bag_word_weight,
    ):

        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids,
            encoder_attention_mask,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True,
        )

        all_hiddens = lm_out.hidden_states[-1]

        batch_size = encoder_input_ids.size(0)
        max_len = encoder_input_ids.size(1)
        hidden_size = all_hiddens.size(-1)

        padded_hiddens = all_hiddens.new_zeros((batch_size, max_len, hidden_size))

        offset = 0
        for i in range(batch_size):
            length = int(input_ids_length[i])  # number of valid tokens
            padded_hiddens[i, :length, :] = all_hiddens[offset : offset + length]
            offset += length

        def average_pool(last_hidden_states, attention_mask):
            masked_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            sum_hidden = masked_hidden.sum(dim=1)
            valid_counts = attention_mask.sum(dim=1, keepdim=True)

            return sum_hidden / valid_counts

        mean_hiddens_per_example = average_pool(padded_hiddens, encoder_attention_mask)
        mean_hiddens = mean_hiddens_per_example.unsqueeze(1)

        mlm_loss = self.decoder_mlm_loss(mean_hiddens, decoder_input_ids, decoder_attention_mask, decoder_labels)

        ot_embedding = self.ot_embedding(lm_out.logits[:, 1:], encoder_attention_mask[:, 1:])
        bow_loss = self.decoder_ot_loss(ot_embedding, bag_word_weight=bag_word_weight)

        loss = mlm_loss + self.model_args.bow_loss_weight * bow_loss + lm_out.loss

        return (loss,)

        #  --------------------

        # decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
        # hiddens = torch.cat([mean_hiddens, decoder_embedding_output[:, 1:]], dim=1)

        # # decoder_position_ids = self.lm.model.embeddings.position_ids[:, :decoder_input_ids.size(1)]
        # # decoder_position_embeddings = self.lm.bert.embeddings.position_embeddings(decoder_position_ids)  # B L D
        # # query = decoder_position_embeddings + cls_hiddens

        # mean_hiddens = mean_hiddens.expand(hiddens.size(0), hiddens.size(1), hiddens.size(2))
        # query = self.decoder_embeddings(inputs_embeds=mean_hiddens)

        # matrix_attention_mask = self.lm.get_extended_attention_mask(
        #     decoder_attention_mask,
        #     decoder_attention_mask.shape,
        #     decoder_attention_mask.device
        # )

        # hiddens = self.c_head(query=query,
        #                       key=hiddens,
        #                       value=hiddens,
        #                       attention_mask=matrix_attention_mask)[0]
        # pred_scores, loss = self.mlm_loss(hiddens, decoder_labels)

        # return (loss + lm_out.loss,)

    #         hidden_states_flat = lm_out.hidden_states[-1]
    #         print(hidden_states_flat.shape)

    #         print('lm_out.hidden_states', lm_out.hidden_states[-1].shape)
    #         cls_hiddens = lm_out.hidden_states[-1][:, 0]
    #         print('cls_hiddens.shape', cls_hiddens.shape)

    # #        cls_hiddens = lm_out.hidden_states[-1][:, 0]
    #         mlm_loss = self.decoder_mlm_loss(cls_hiddens, decoder_input_ids, decoder_attention_mask, decoder_labels)
    #         ot_embedding = self.ot_embedding(lm_out.logits[:, 1:], encoder_attention_mask[:, 1:])
    #         bow_loss = self.decoder_ot_loss(ot_embedding, bag_word_weight=bag_word_weight)

    #         loss = mlm_loss + self.model_args.bow_loss_weight * bow_loss + lm_out.loss

    #         return (loss, )

    def mlm_loss(self, hiddens, labels):
        if hasattr(self.lm, "cls"):
            pred_scores = self.lm.cls(hiddens)
        elif hasattr(self.lm, "decoder"):
            pred_scores = self.lm.decoder(hiddens)
        else:
            raise NotImplementedError

        masked_lm_loss = self.cross_entropy(pred_scores.view(-1, self.lm.config.vocab_size), labels.view(-1))
        return pred_scores, masked_lm_loss

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_args: ModelArguments, *args, **kwargs):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args)
        return model
