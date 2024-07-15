#! /usr/bin/env python3


from typing import List

import torch
import torch.nn.functional as F
from gpt_model_simplest import GPTModel, GPTTokenizer

GPT_PARAM_PATH = "/u/qxc4fh/zeyu_workspace/gpt_params.pkl"
GPT_TOKENIZER_KERNEL_PATH = "/u/qxc4fh/zeyu_workspace/gpt_tokenizer_kernel.pkl"
MODEL_DEVICE = "cuda:0"


def pad_batch(batch, pad_id, max_len):
    context_lengths = []
    max_context_length = max([len(tokens) for tokens in batch])
    for tokens in batch:
        context_length = len(tokens)
        if context_length < max_context_length + max_len:
            tokens.extend([pad_id] * (max_context_length + max_len - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def tokenize_batch(tokenizer, sentences, max_len, add_BOS=True):
    if add_BOS:
        context_tokens = [[tokenizer.eos_id] + tokenizer.text_to_ids(s) for s in sentences]
    else:
        context_tokens = [tokenizer.text_to_ids(s) for s in sentences]
    context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eos_id, max_len)
    context_tokens_tensor = torch.LongTensor(context_tokens)
    context_length_tensor = torch.LongTensor(context_lengths)
    return context_tokens_tensor, context_length_tensor


def get_ltor_masks_and_position_ids(data):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    att_mask_batch = 1
    attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length
    )

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, position_ids


def get_batch(context_tokens):
    """Generate batch from context tokens."""
    # Move to GPU.
    tokens = context_tokens.contiguous()  # .cuda()
    # Get the attention mask and postition ids.
    attention_mask, position_ids = get_ltor_masks_and_position_ids(tokens)

    return tokens, attention_mask, position_ids


def repetition_penalty(logits, repetition_penalty, used_tokens):
    """Implement the repetition penalty, check paper
    https://arxiv.org/pdf/1909.05858.pdf
    """
    if used_tokens is not None and repetition_penalty != 1.0:
        logits_update = torch.gather(logits, 1, used_tokens)
        logits = torch.scatter(logits, 1, used_tokens, logits_update / repetition_penalty)
    return logits


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """This function has been mostly taken from huggingface conversational
    ai code at
        https://medium.com/huggingface/how-to-build-a-state-of-the-art-
             conversational-ai-with-transfer-learning-2d818ac26313"""

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def sample_sequence_batch(
    model,
    tokenizer,
    context_tokens,
    context_lengths,
    attention_mask,
    position_ids,
    tokens_to_generate,
    temperature_value=1.0,
    repetition_penalty_value=1.2,
    top_k_value=0,
    top_p_value=0.9,
):
    micro_batch_size = context_tokens.shape[0]
    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()

        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        eod_id = tokenizer.eos_id
        counter = 0

        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size], device=context_tokens.device).byte()
        tokens = context_tokens
        output_logits = None
        all_generated_indices = None  # used to track all generated indices
        # Generate enough tokens for the longest sequence
        maxlen = tokens_to_generate + context_lengths.max().item()

        if maxlen > model.encoder_seq_length + 1:
            maxlen = model.encoder_seq_length + 1

        while context_length < maxlen:
            # types2use = None
            if counter == 0:
                # Allocate memory for the entire context.
                set_inference_key_value_memory = True
                tokens2use = tokens[:, :context_length]
                positions2use = position_ids[:, :context_length]
                # not using type2use. uncomment it if it is used
                # if type_ids is not None:
                #     types2use = type_ids[:, :context_length]
            else:
                # Set this to false so the memory is not reallocated.
                set_inference_key_value_memory = False
                tokens2use = tokens[:, context_length - 1].view(batch_size, -1)
                positions2use = position_ids[:, context_length - 1].view(batch_size, -1)
                # not using type2use. uncomment it if it is used
                # if type_ids is not None:
                #     types2use = type_ids[:, context_length - 1].view(batch_size, -1)

            attention_mask_repeat = torch.concat([attention_mask for _ in range(micro_batch_size)])
            setkey_value_array = torch.tensor([set_inference_key_value_memory] * micro_batch_size)
            len_array = torch.tensor([maxlen] * micro_batch_size)

            # Only prompt learning models will have a prompt table, and require task ids
            batch = [tokens2use, attention_mask_repeat, positions2use, setkey_value_array, len_array]

            output = model(batch)

            assert output is not None
            output = output.float()
            logits = output[:, -1].view(batch_size, -1).contiguous()

            # make sure it won't sample outside the vocab_size range
            logits[:, tokenizer.vocab_size :] = -float("Inf")

            logits = logits.float()
            logits /= temperature_value
            # handle repetition penality
            logits = repetition_penalty(logits, repetition_penalty_value, all_generated_indices)
            logits = top_k_logits(logits, top_k=top_k_value, top_p=top_p_value)
            log_probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1).view(-1)

            started = context_lengths <= context_length

            # Clamp the predicted out of vocabulary tokens
            prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)
            new_tokens = switch(tokens[:, context_length].view(-1), prev, started)

            # Replace sampled tokens w/ done token if EOD has already been sampled
            new_tokens = switch(new_tokens, eod_id, is_done)

            # Insert either new predicted or next prompt token
            tokens[:, context_length] = new_tokens

            if output_logits is None:
                output = F.log_softmax(output[:, :context_length, :], 2)
                indices = torch.unsqueeze(tokens[:, 1 : context_length + 1], 2)
                output_logits = torch.gather(output, 2, indices).squeeze(2)
                all_generated_indices = indices[:, :, 0]
            else:
                output = F.log_softmax(output, 2)
                indices = torch.unsqueeze(new_tokens, 1).unsqueeze(2)
                new_output_logits = torch.gather(output, 2, indices).squeeze(2)

                # TODO(rprenger) we're copying output_logits every time.  Should pre-allocate
                output_logits = torch.cat([output_logits, new_output_logits], 1)
                all_generated_indices = torch.cat([all_generated_indices, indices[:, :, 0]], 1)

            done_token = (prev == eod_id).byte() & started.byte()
            is_done = is_done | done_token

            done = torch.all(is_done)

            yield tokens, output_logits

            context_length += 1
            counter += 1
            if done:
                break


def gpt_generate(inputs: List[str], max_gen_len):
    tokenizer = GPTTokenizer(GPT_TOKENIZER_KERNEL_PATH)
    context_tokens_tensor, context_length_tensor = tokenize_batch(tokenizer, inputs, max_gen_len)
    context_tokens_tensor = context_tokens_tensor.to(torch.device(MODEL_DEVICE))
    context_length_tensor = context_length_tensor.to(torch.device(MODEL_DEVICE))

    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)

    model = GPTModel(GPT_PARAM_PATH, 16, MODEL_DEVICE)
    batch_token_iterator = sample_sequence_batch(
        model,
        tokenizer,
        context_tokens_tensor,
        context_length_tensor,
        attention_mask,
        position_ids,
        max_gen_len,
    )

    for tokens, output_logits in batch_token_iterator:
        context_length += 1

    decode_tokens = tokens[:, :context_length]
    decode_tokens = decode_tokens.cpu().numpy().tolist()
    resp_sentences = []
    resp_sentences_seg = []
    for decode_token in decode_tokens:
        sentence = tokenizer.ids_to_text(decode_token)
        resp_sentences.append(sentence)

        words = []
        for token in decode_token:
            # Skip any soft prompt pseudo tokens
            if token not in tokenizer.tokenizer.decoder:
                print(1)
                continue
            word = tokenizer.tokenizer.decoder[token]
            word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode("utf-8", errors="replace")
            words.append(word)
        resp_sentences_seg.append(words)

    return resp_sentences, resp_sentences_seg


if __name__ == "__main__":
    inputs = ["How big is the universe?"]
    resp_sentences, resp_sentences_seg = gpt_generate(inputs, 100)
    print(resp_sentences)
