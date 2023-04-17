# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GPT2Config"
_TOKENIZER_FOR_DOC = "GPT2Tokenizer"

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    # See all GPT-2 models at https://huggingface.co/models?filter=gpt2
]


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert (
                    pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False):
        '''
        nx:     token 嵌入维数 hidden_size
        n_ctx:  attention mask 的维数, 通常为 maximum sequence length
        config: 模型配置信息
        scale:  是否对注意力分数进行缩放以避免梯度消失或爆炸问题
        is_cross_attention: 是否为 cross_attention
        '''
        super().__init__()

        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        self.split_size = n_state = nx      # hidden_size
        assert n_state % config.n_head == 0 # head_size = hidden_size // n_head
        
        # 注册一个 bias 下三角矩阵，尺寸为 (1,1,n_ctx,n_ctx), 它指示了两个 n_ctx 长度序列元素间的关系，将被用作 causal mask
        self.register_buffer("bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx))
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        
        self.n_head = config.n_head
        self.scale = scale
        
        assert is_cross_attention == False  # DT 不使用 cross_attention
        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * n_state, nx)
            self.q_attn = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(3 * n_state, nx)   # c_attn 用于后面升维切出 qkv 矩阵（这个 transformer 库的 Conv1D 和 Linear 变换基本等价，这里做一个升维 ）
            
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()   # 已经剪枝的 attention heads

    def prune_heads(self, heads):
        '''
        attention head 剪枝，从 heads 中剪去 self.pruned_heads
        '''
        if len(heads) == 0:
            return
        
        # 从 heads 中剪去 self.pruned_heads 中的头，返回剩下的 heads 及相应索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        '''
        q:                  (batch_size, n_head, 3*seq_length, head_features)
        k:                  (batch_size, n_head, head_features, 3*seq_length)
        v:                  (batch_size, n_head, 3*seq_length, head_features)
        attention_mask:     (batch_size, 1, 1, 3*seq_length)
        head_mask:          None
        output_attentions:  False
        '''
        w = torch.matmul(q, k)  # (batch_size, n_head, 3*seq_length, 3*seq_length) 
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)  # 用根号 head_features 作为温度系数对注意力分数进行缩放，避免梯度消失或爆炸问题
        nd, ns = w.size(-2), w.size(-1)         # 都是序列长度 3*seq_length

        # 对 "normal" attention layer 施加 causal mask
        if not self.is_cross_attention:
            mask = self.bias[:, :, ns - nd: ns, :ns]
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))   # (batch_size, n_head, 3*seq_length, 3*seq_length)    

        # 施加 padding mask
        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        assert head_mask is None        # DT 中没有遮盖注意力头
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]  # (batch_size, n_head, 3*seq_length, head_features)
        if output_attentions:           
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        '''
        汇聚多头注意力结果 x: (batch_size, n_head, 3*seq_length, head_features)
        '''
        x = x.permute(0, 2, 1, 3).contiguous()                      # (batch_size, 3*seq_length, n_head, head_features)
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)    # (batch_size, 3*seq_length, n_head * head_features)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        '''
        将张量 x:(batch_size, 3*seq_length, hidden_size) 的嵌入维度按注意力头数量进行切分
        '''
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)  # (batch_size, 3*seq_length, n_head, head_features)
        x = x.view(*new_x_shape)            # (batch_size, 3*seq_length, n_head, head_features)
        if k:
            return x.permute(0, 2, 3, 1)    # (batch_size, n_head, head_features, 3*seq_length)
        else:
            return x.permute(0, 2, 1, 3)    # (batch_size, n_head, 3*seq_length, head_features)

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):  
        '''
        hidden_states:          (batch_size, 3*seq_length, hidden_size)
        attention_mask:         (batch_size, 1, 1, 3*seq_length)
        layer_past:             None
        head_mask:              None
        encoder_hidden_states:  None    (DT 没有 encoder 部分)
        encoder_attention_mask: None    (DT 没有 encoder 部分)
        output_attentions:      False   (这个应该是 encoder 输出来做 cross attn 的)
        '''
        assert encoder_hidden_states is None    # DT 没有 encoder 部分
        if encoder_hidden_states is not None:
            assert hasattr(
                self, "q_attn"
            ), "If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            # .c_attn(hidden_states) 将 hidden_states 升维到 (batch_size, 3*seq_length, 3*hidden_size)
            # .split(hidden_states)  将 hidden_states 切成 query, key, value 三个 (batch_size, 3*seq_length, hidden_size)
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self.split_heads(query)     # (batch_size, n_head, 3*seq_length, head_features)
        key = self.split_heads(key, k=True) # (batch_size, n_head, head_features, 3*seq_length)
        value = self.split_heads(value)     # (batch_size, n_head, 3*seq_length, head_features)
        
        assert layer_past is None           # for DT
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))  # (2, batch_size, n_head, head_features, 3*seq_length)
        else:
            present = (None,)

        # 注意力计算
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]         # (batch_size, n_head, 3*seq_length, head_features)

        # 多头注意力汇聚（拼接+线性变换）
        a = self.merge_heads(a)     # (batch_size, 3*seq_length, n_head*head_features=hidden_size)
        a = self.c_proj(a)          # (batch_size, 3*seq_length, hidden_size)
        a = self.resid_dropout(a)

        # 若设置了 output_attentions=True, 这里 attn_outputs[1:]= output_attentions]，但在 DT 中没有
        outputs = [a, present] + attn_outputs[1:]   
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class AdapterMLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class Block(nn.Module):
    '''
    GPT2 的 Block 中，在 Multi self-Attention 和 Feed Forward 层之前进行 Layer Norm 
    '''
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # self.adapter_ln = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)
        # self.adapter_mlp = AdapterMLP(512, config)  # ADAPTER

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        '''
        hidden_states:          (batch_size, 3*seq_length, hidden_size)
        attention_mask:         (batch_size, 1, 1, 3*seq_length)
        layer_past:             None
        head_mask:              None
        encoder_hidden_states:  None    (DT 没有 encoder 部分)
        encoder_attention_mask: None    (DT 没有 encoder 部分)
        output_attentions:      False   (这个应该是 encoder 输出来做 cross attn 的)
        '''
        # Layer Norm + Multi self-Attention
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )                               # attn_outputs: a, present, (attentions)
        attn_output = attn_outputs[0]   # (batch_size, 3*seq_length, hidden_size)
        outputs = attn_outputs[1:]      # [present, ]   (present是stack的key和value)
        
        # residual connection
        hidden_states = attn_output + hidden_states

        assert encoder_hidden_states is None    # DT 中没有 encoder
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        # Layer Norm + FFD
        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        
        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states
        # hidden_states = hidden_states + self.adapter_ln(self.adapter_mlp(hidden_states))

        outputs = [hidden_states] + outputs
        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # module.weight.data.fill_(.01)  # KL: Adapter change


@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`mc_labels` is provided):
            Multiple choice classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,batch_size, num_heads, sequence_length, embed_size_per_head)`).
            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mc_logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


GPT2_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""
GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`):
            :obj:`input_ids_length` = ``sequence_length`` if :obj:`past_key_values` is ``None`` else
            ``past_key_values[0].shape[-2]`` (``sequence_length`` of input past key value states). Indices of input
            sequence tokens in the vocabulary.
            If :obj:`past_key_values` is used, only ``input_ids`` that do not have their past calculated should be
            passed as ``input_ids``.
            Indices can be obtained using :class:`~transformers.GPT2Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are input IDs? <../glossary.html#input-ids>`__
        past_key_values (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            :obj:`past_key_values` output below). Can be used to speed up sequential decoding. The ``input_ids`` which
            have their past given to this model should not be passed as ``input_ids`` as they have already been
            computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:
            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
            If :obj:`past_key_values` is used, optionally only the last :obj:`inputs_embeds` have to be input (see
            :obj:`past_key_values`).
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""
PARALLELIZE_DOCSTRING = r"""
    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.
    Args:
        device_map (:obj:`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the gpt2 models have the
            following number of attention modules:
                - gpt2: 12
                - gpt2-medium: 24
                - gpt2-large: 36
                - gpt2-xl: 48
    Example::
            # Here is an example of a device map on a machine with 4 GPUs using gpt2-xl, which has a total of 48 attention modules:
            model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
            device_map = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
                          1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                          2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                          3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]}
            model.parallelize(device_map)
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.
    Example::
        # On a 4 GPU machine with gpt2-large:
        model = GPT2LMHeadModel.from_pretrained('gpt2-large')
        device_map = {0: [0, 1, 2, 3, 4, 5, 6, 7],
                    1: [8, 9, 10, 11, 12, 13, 14, 15],
                    2: [16, 17, 18, 19, 20, 21, 22, 23],
                    3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]}
        model.parallelize(device_map) # Splits the model across several devices
        model.deparallelize() # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
"""


@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.use_layers = None

    def set_layers(self, num_layers):
        assert 1 <= num_layers <= len(self.h)
        if num_layers is not None:
            num_layers -= 1
        self.use_layers = num_layers

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="gpt2",
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            attention_mask=None,            # GPT 的原始序列输入的 attn mask，这个 mask 主要用来遮盖 padding 部分
            input_ids=None,                 # GPT 的原始序列输入（应该是词表索引形式）input_ids 和 inputs_embeds 有且只有一个不为 None
            inputs_embeds=None,             # GPT 的原始序列输入（嵌入向量形式）input_ids 和 inputs_embeds 有且只有一个不为 None
            token_type_ids=None,            # 若非 None，是尺寸为 (batch_size, input_ids_length) 的 0/1 tensor, 用来区分输入序列中连接的两个句子
            position_ids=None,              # 若非 None，是尺寸为 (batch_size, sequence_length) 的 tensor，是位置嵌入中每个输入序列符号的位置索引
            head_mask=None,                 # 若非 None，是尺寸为 (num_layers, num_heads) 的 0/1 tensor, 用来遮盖失能一些选定的注意力头（0位置）
            use_cache=None,                 # Whether or not the model should return the last key/values attentions (not used by all models).
            output_attentions=None,         # Whether or not the model should returns all attentions.
            output_hidden_states=None,      # Whether or not the model should return all hidden-states.
            return_dict=None,               # Whether or not the model should return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            past_key_values=None,           # 当 use_cache = True 时，返回长度为 config.n_layers 的 FloatTensor 列表，
                                            # 每个张量的形状为 (2, batch_size, num_heads, sequence_length, embed_size_per_head)
                                            # 包含预先计算的隐藏状态（注意力块中的键和值），可用于加速顺序解码
            encoder_hidden_states=None,     # 当 output_hidden_states=True 时，返回 (batch_size, sequence_length, hidden_size, 2) 尺寸的 float tensor，
                                            # 每个元组包含编码器在每层输出处的隐藏状态和初始嵌入输出。
            encoder_attention_mask=None,    # 编码器 attention mask
    ):
        '''
        inputs_embeds:  (batch_size, 3*seq_length, hidden_size)
        attention_mask: (batch_size, 3*seq_length, )
        '''
        # 行为控制
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions               # False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states   # False
        use_cache = use_cache if use_cache is not None else self.config.use_cache                                               # True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict                                   # True

        # 从原始输入中获取 input_shape 和 batch_size
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1] # (batch_size, 3*seq_length,)
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # DT 里 past_key_values 也没用到，设为 [None]*len(self.h)
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)

        # 这两个在 DT 都没用到
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        
        # 这一段 DT 也没用到，应该是 GPT 用来计算 position embadding 用的
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        else:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask. 这个 mask 主要用来遮盖 padding 部分
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length], so we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attentionz used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Note that attention_mask is 1.0 for positions we want to attend and 0.0 for masked positions
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # 这部分涉及 cross_attention，在 DT 也没用上
        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, n_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # 这部分涉是设置注意力头 mask，在 DT 也没用上
        # Prepare head mask if needed（1.0 in head_mask indicate we keep the head）
        # attention_probs shape:(batch_size, n_heads, seq_length, seq_length)
        # head_mask shape:      (n_layer, batch_size, n_heads, seq_length, seq_length)
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)  # [None, None, None] for DT

        # 如果输入不是 inputs_embeds 形式而是 inputs_inds 形式，用 wte 做词向量嵌入，DT 没用这个
        '''
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids) 
        hidden_states = inputs_embeds + position_embeds
        '''
        assert inputs_embeds is not None    # DT 预处理时直接拿到了轨迹各个 token 的嵌入，应当满足这个
        hidden_states = inputs_embeds

        assert token_type_ids is None       # token 所属句子的嵌入(比如BERT上下句任务会用)，DT 没用这个
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)  # (batch_size, 3*seq_length, hidden_size)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if self.use_layers is not None and i >= self.use_layers:
                break

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                assert layer_past is None   # for DT
                if layer_past is not None:
                    layer_past = layer_past.to(hidden_states.device)
                
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                
                # Ensure that head_mask is always on the same device as hidden_states
                assert head_mask == [None] * self.config.n_layer    # for DT
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            
            assert output_hidden_states == False    # 这个应该是 Encoder 输出来作 cross attn 的，DT 不需要
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):
                # 使用 gradient_checkpointing，以降低后向传递的速度为代价来节省内存
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # checkpointing only works with tuple returns, not with lists
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                # DT 走这边
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )   # outputs: hidden_states, present, (attentions, cross_attentions)

            hidden_states, present = outputs[:2]    # hidden_states: (batch_size, 3*seq_length, hidden_size)
            if use_cache is True:
                presents = presents + (present,)

            assert output_attentions == False   # for DT
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            assert self.model_parallel == False   # 没有多卡
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        # 所有 transformer decoder 后的 Layer Norm
        hidden_states = self.ln_f(hidden_states)            # (batch_size, 3*seq_length, hidden_size)

        # 至此 GPT2 结构全部跑完
        hidden_states = hidden_states.view(*output_shape)   # (batch_size, 3*seq_length, hidden_size)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        # 返回 GPT2 输出 last_hidden_state 和其他中间参数
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
