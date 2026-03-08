import glob
import hashlib
import hmac
import sys
from operator import mul
from typing import Optional, Callable, TypeVar, Any, Tuple

from PIL import Image
from omegaconf import OmegaConf
from tqdm.auto import trange

from scripts.make_samples import get_parser, load_model_and_dset, local_indeces, save_image


class DRBG(object):
    def __init__(self, key, seed : bytes):
        self.key = key
        self.val = b'\x01' * 64
        self.reseed(seed)

        self.byte_index = 0
        self.bit_index = 0

    @staticmethod
    def hmac(key, val):
        return hmac.new(key, val, hashlib.sha512).digest()

    def reseed(self, data=b''):
        self.key = self.hmac(self.key, self.val + b'\x00' + data)
        self.val = self.hmac(self.key, self.val)

        if data:
            self.key = self.hmac(self.key, self.val + b'\x01' + data)
            self.val = self.hmac(self.key, self.val)

    def generate_bits(self, n):
        xs = np.zeros(n, dtype=bool)
        for i in range(0, n):
            xs[i] = (self.val[self.byte_index] >> (7 - self.bit_index)) & 1

            self.bit_index += 1
            if self.bit_index >= 8:
                self.bit_index = 0
                self.byte_index += 1

            if self.byte_index >= 8:
                self.byte_index = 0
                self.val = self.hmac(self.key, self.val)

        self.reseed()
        return xs


# @title

import numpy as np

def limit_past(past):
    past = list(past)
    for i in range(len(past)):
        past[i] = past[i][:, :, :, -1022:]
    return past

# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2 ** i)
    return res


def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def is_sent_finish(token_idx, enc):
    token = enc.decoder[token_idx]
    return '.' in token or '!' in token or '?' in token


def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    i = 0
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break

    return i


def encode_context(raw_text, enc):
    context_tokens = [enc.encoder['<|endoftext|>']] + enc.encode(raw_text)
    return context_tokens


def get_vqgan_sflckr(model_directory_path: str, *, quiet: bool = False) -> Tuple[Any, Any]:
    if not os.path.exists(model_directory_path):
        raise ValueError(f"Cannot find {model_directory_path}")
    if os.path.isfile(model_directory_path):
        paths = model_directory_path.split("/")
        try:
            idx = len(paths) - paths[::-1].index("logs") + 1
        except ValueError:
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
        logdir = "/".join(paths[:idx])
        ckpt = model_directory_path
    else:
        assert os.path.isdir(model_directory_path), model_directory_path
        logdir = model_directory_path.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
    if not quiet:
        print(f"logdir:{logdir}")
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    config = OmegaConf.merge(*configs)

    gpu = torch.cuda.is_available()
    eval_mode = True
    dsets, model, _ = load_model_and_dset(config, ckpt, gpu, eval_mode)

    return dsets, model

# @title

import math


def bin_sort(l, token_indices, total, entropy, device):
    # compute entropy for upper bound on the number of bins we need

    bucket_size = total
    num_bins = 2 ** int(entropy + 1)
    bucket_size = total / num_bins

    bins = [torch.empty(0, dtype=torch.long, device=device)] * num_bins
    value_in_bins = [0] * num_bins
    space_left_after = [total - i * bucket_size for i in range(0, num_bins)]

    token_bins = [torch.empty(0, dtype=torch.long, device=device)] * num_bins

    # Figuring out what the search order should be
    step_size = num_bins / 4
    search_order = []
    priorities = [0] * num_bins
    priority = 0
    search_order.append(int(num_bins / 2))
    search_order.append(0)
    priorities[int(num_bins / 2)] = 0
    priorities[0] = 0
    while (step_size >= 1):
        priority += 1
        for x in range(num_bins - int(step_size), -1, -int(step_size * 2)):
            search_order.append(x)
            priorities[x] = priority
        step_size = step_size / 2

    # Adding the actual elements
    for (item, token_index) in zip(l.tolist(), token_indices.tolist()):
        found_single_bucket_fit = False
        single_bucket_index = -1
        single_bucket_value = bucket_size

        found_multi_bucket_bumpless_fit = False
        multi_bucket_bumpless_index = -1
        multi_bucket_bumpless_value = total

        found_multi_bucket_bumping_fit = False
        multi_bucket_bumping_index = -1
        multi_bucket_bumping_value = total

        for i in search_order:  # for index in search_order
            if (item > space_left_after[i]):
                continue
            if (value_in_bins[i] >= bucket_size):
                continue

            # Priority of choices
            #  1. Can i place this thing in an empty bucket all on its own?
            #  2. Can i plan this somewhere where is doesnt have to bump anything else around?
            #    2a. Minimize the wasted space.  Aka use the smallest space (of equal priority) that accomplishes this goal
            #  3. If not (1) and (2), then put it in the space the bumps stuff the least.

            if (value_in_bins[i] + item > bucket_size):  # Would overflow.

                space_before_next_block = bucket_size - value_in_bins[i]
                for j in range(i + 1, len(bins)):
                    if (value_in_bins[
                        j] > 0):  # We have found a bucket with something in it.  This is how much space we have here.
                        space_before_next_block = space_before_next_block + (bucket_size - value_in_bins[i])
                        break
                    else:  # This was a empty bucket
                        space_before_next_block = space_before_next_block + bucket_size

                if ((not found_multi_bucket_bumpless_fit) or (
                        found_multi_bucket_bumpless_fit and priorities[i] <= priorities[
                    multi_bucket_bumpless_index])):  # This could potentially be a match

                    # If this is a valid space to put this without bumping and it is a better fit than previous spaces
                    if (space_before_next_block > item and space_before_next_block < multi_bucket_bumpless_value):
                        # set this to be the pointer!  we can fit stuff here
                        found_multi_bucket_bumpless_fit = True
                        multi_bucket_bumpless_index = i
                        multi_bucket_bumpless_value = space_before_next_block

                    # Find the overflow that will bump the least
                    if (item - space_before_next_block < multi_bucket_bumping_value):
                        found_multi_bucket_bumping_fit = True
                        multi_bucket_bumping_index = i
                        multi_bucket_bumping_value = item - space_before_next_block

            if (value_in_bins[i] + item <= bucket_size):  # Would fit
                if (single_bucket_value > value_in_bins[i]):
                    found_single_bucket_fit = True
                    single_bucket_value = value_in_bins[i]
                    single_bucket_index = i

        if (single_bucket_index == multi_bucket_bumpless_index == multi_bucket_bumping_index == -1):
            bins[0] = torch.cat((torch.tensor([item], device=device), bins[0]), 0)
            token_bins[0] = torch.cat((torch.tensor([token_index], device=device), token_bins[0]), 0)
            continue

        if found_single_bucket_fit:
            # We found somewhere we can actually fit!
            bins[single_bucket_index] = torch.cat((bins[single_bucket_index], torch.tensor([item], device=device)), 0)
            token_bins[single_bucket_index] = torch.cat(
                (token_bins[single_bucket_index], torch.tensor([token_index], device=device)), 0)
            value_in_bins[single_bucket_index] += item
            for i in range(0, single_bucket_index + 1):
                space_left_after[i] -= item

        elif found_multi_bucket_bumpless_fit:
            # Found somewhere we can put this without upsetting the force
            part_in_bucket = bucket_size - value_in_bins[multi_bucket_bumpless_index]
            part_overflow = item - part_in_bucket
            bins[multi_bucket_bumpless_index] = torch.cat(
                (bins[multi_bucket_bumpless_index], torch.tensor([item], device=device)), 0)
            token_bins[multi_bucket_bumpless_index] = torch.cat(
                (token_bins[multi_bucket_bumpless_index], torch.tensor([token_index], device=device)), 0)
            value_in_bins[multi_bucket_bumpless_index] = bucket_size

            # Fill this bucket and continue overflowing
            j = multi_bucket_bumpless_index + 1
            for i in range(0, j):
                space_left_after[i] -= item

            while (part_overflow > 0):
                new_part_overflow = (value_in_bins[j] + part_overflow) - bucket_size
                value_in_bins[j] = min(bucket_size, part_overflow + value_in_bins[j])  # mark the bucket as filled
                space_left_after[j] -= part_overflow
                part_overflow = new_part_overflow
                j += 1

        else:
            part_in_bucket = bucket_size - value_in_bins[multi_bucket_bumping_index]
            part_overflow = item - part_in_bucket
            bins[multi_bucket_bumping_index] = torch.cat(
                (bins[multi_bucket_bumping_index], torch.tensor([item], device=device)), 0)
            token_bins[multi_bucket_bumping_index] = torch.cat(
                (token_bins[multi_bucket_bumping_index], torch.tensor([token_index], device=device)), 0)
            value_in_bins[multi_bucket_bumping_index] = bucket_size

            # Fill this bucket and continue overflowing
            j = multi_bucket_bumping_index + 1
            for i in range(0, j):
                space_left_after[i] -= item
            while (part_overflow > 0):
                new_part_overflow = (value_in_bins[j] + part_overflow) - bucket_size
                value_in_bins[j] = min(bucket_size, part_overflow + value_in_bins[j])  # mark the bucket as filled
                space_left_after[j] -= part_overflow
                part_overflow = new_part_overflow
                j += 1

    sorted_tensor = torch.cat(bins, 0)
    sorted_tokens = torch.cat(token_bins, 0)

    return sorted_tensor, sorted_tokens


def compute_ev(t, precision):
    expected_bits = []
    cum_probs = t.cumsum(0)

    for selection in range(0, len(cum_probs)):
        # Calculate new range as ints
        new_int_bottom = cum_probs[selection - 1] if selection > 0 else 0
        new_int_top = cum_probs[selection]

        # Convert range to bits
        new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
        new_int_top_bits_inc = list(
            reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

        # Consume most significant bits which are now fixed and update interval
        num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
        expected_bits.append(t[selection] * num_bits_encoded)

    return (float(sum(expected_bits).item()) / (2 ** precision))


def visualize_bins(values_in_bins, bucket_size):
    out_str = "["
    for b in values_in_bins:
        out_str = out_str + "  " + str(round(100 * b / bucket_size, 2)) + "  |"
    out_str = out_str + "]"
    print(out_str)


def visualize_distribution(l):
    total = sum(l)
    out_str = "["
    for b in l:
        out_str = out_str + "  " + str(round(100 * b / total, 2)) + "  |"
    out_str = out_str + "]"
    print(out_str)


def compute_entropy(lists):
    total = sum(lists)
    entropy = -1 * sum([(x / total) * math.log2(x / total) for x in lists])
    return entropy


# @title
import torch
import torch.nn.functional as F

import os

# Constants for HMAC-DRBG -- MUST CHANGE FOR SECURE IMPLEMENTATION
sample_key = b'0x01' * 64
sample_seed_prefix = b'sample'
sample_nonce_counter = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'


def encode_meteor(model, enc, message, context, finish_sent=False, device='cuda', temp=1.0, precision=16, topk=50000,
                  is_sort=False, randomize_key=False, input_key=sample_key, input_nonce=sample_nonce_counter):
    if randomize_key:
        input_key = os.urandom(64)
    mask_generator = DRBG(input_key, sample_seed_prefix + input_nonce)
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2 ** precision
    threshold = 2 ** (-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    prev = context
    output = context
    past = None

    total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0  # in bits
    total_entropy_ptau = 0
    total_num_sents = 0

    with torch.no_grad():
        i = 0
        sent_finish = False
        while i < len(message) or (finish_sent and not sent_finish):
            logits, past = model(prev.unsqueeze(0), past=past)
            past = limit_past(past)
            logits[0, -1, -1] = -1e20  # endoftext token can't happen
            logits[0, -1, 628] = -1e20  # 2 newlines token can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)

            # conditions for having reached the end of the message
            if i >= len(message):
                selection = 0
                sent_finish = is_sent_finish(indices[selection].item(), enc)
            else:
                # Cutoff low probabilities that would be rounded to 0
                cur_int_range = cur_interval[1] - cur_interval[0]
                cur_threshold = 1 / cur_int_range
                k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
                probs_temp_int = probs_temp[:k]  # Cutoff all but top k
                old_indices = indices
                indices = indices[:k]

                # Rescale to correct range
                probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range

                entropy_in_this_distribution = entropy(probs_temp, log_probs_temp)

                # Round probabilities to integers given precision
                probs_temp_int = probs_temp_int.round().long()

                if is_sort:
                    probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range,
                                                       entropy_in_this_distribution, device)
                cum_probs = probs_temp_int.cumsum(0)

                # Remove any elements from the bottom if rounding caused the total prob to be too large
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]

                # Add any mass to the top if removing/rounding causes the total prob to be too small
                cum_probs += cur_int_range - cum_probs[-1]  # add

                # Get out resulting probabilities
                probs_final = cum_probs.clone()
                probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                # Convert to position in range
                cum_probs += cur_interval[0]

                # Apply the mask to the message
                message_bits = message[i:i + precision]
                if i + precision > len(message):
                    message_bits = message_bits + [0] * (i + precision - len(message))

                mask_bits = mask_generator.generate_bits(precision)
                for b in range(0, len(message_bits)):
                    message_bits[b] = message_bits[b] ^ mask_bits[b]

                # Get selected index based on binary fraction from message bits
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                # Calculate new range as ints
                new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(
                    reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

                # Consume most significant bits which are now fixed and update interval
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                i += num_bits_encoded

                # Gather statistics
                total_log_probs += log_probs[selection].item()

                q = probs_final.double() / probs_final.sum()
                logq = q.log()
                total_kl += kl(q, logq, log_probs[:len(q)])
                total_entropy_ptau += entropy_in_this_distribution
                total_num_for_stats += 1

            # Update history with new token
            prev = indices[selection].view(1)
            output = torch.cat((output, prev))
            total_num += 1

            # For text->bits->text
            partial = enc.decode(output[len(context):].tolist())
            if '<eos>' in partial:
                break

    avg_NLL = -total_log_probs / total_num_for_stats
    avg_KL = total_kl / total_num_for_stats
    avg_Hq = total_entropy_ptau / total_num_for_stats
    words_per_bit = total_num_for_stats / i

    return output[len(context):].tolist(), avg_NLL, avg_KL, words_per_bit, avg_Hq


def decode_meteor(model, image, context, device='cuda', temp=1.0, precision=16, topk=50000, is_sort=False,
                  input_key=sample_key, input_nonce=sample_nonce_counter):

    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)
    mask_generator = DRBG(input_key, sample_seed_prefix + input_nonce)

    max_val = 2 ** precision
    threshold = 2 ** (-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    prev = context
    past = None
    message = []
    with torch.no_grad():
        i = 0
        while i < len(inp):
            logits, past = model(prev.unsqueeze(0), past=past)
            past = limit_past(past)
            logits[0, -1, -1] = -1e20  # endoftext can't happen
            logits[0, -1, 628] = -1e20  # 2 newlines can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            probs_temp = F.softmax(logits_temp, dim=0)

            # Cutoff low probabilities that would be rounded to 0
            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1 / cur_int_range
            k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
            probs_temp_int = probs_temp[:k]  # Cutoff all but top k

            # Rescale to correct range
            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
            entropy_in_this_distribution = entropy(probs_temp, log_probs_temp)

            # Round probabilities to integers given precision
            probs_temp_int = probs_temp_int.round().long()
            if is_sort:
                probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range, entropy_in_this_distribution,
                                                   device)
            cum_probs = probs_temp_int.cumsum(0)

            # Remove any elements from the bottom if rounding caused the total prob to be too large
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                k = overfill_index[0].item()

            # Add any mass to the top if removing/rounding causes the total prob to be too small
            cum_probs += cur_int_range - cum_probs[-1]  # add

            # Covnert to position in range
            cum_probs += cur_interval[0]

            rank = (indices == inp[i]).nonzero().item()

            # Handle most errors that could happen because of BPE with heuristic
            if rank >= k:
                true_token_text = enc.decoder[inp[i]]
                for rank_idx in range(k):
                    prop_token_text = enc.decoder[indices[rank_idx].item()]
                    # common case that is not caught
                    if inp[i] == 128 and indices[rank_idx] == 198:
                        rank = rank_idx
                        inp[i] = indices[rank_idx].item()
                        break

                    # Is there a more likely prefix token that could be the actual token generated?
                    if len(prop_token_text) <= len(true_token_text) and \
                            prop_token_text == true_token_text[:len(prop_token_text)]:
                        rank = rank_idx
                        suffix = true_token_text[len(prop_token_text):]
                        suffix_tokens = enc.encode(suffix)  # a list
                        inp[i] = indices[rank_idx].item()
                        inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                        break

                    # Is there a more likely longer token that could be the actual token generated?
                    elif len(prop_token_text) > len(true_token_text) and \
                            true_token_text == prop_token_text[:len(true_token_text)]:
                        whole_text = true_token_text
                        num_extra = 1
                        while len(whole_text) < len(prop_token_text):
                            whole_text += enc.decoder[inp[i + num_extra]]
                            num_extra += 1
                        if prop_token_text == whole_text[:len(prop_token_text)]:
                            rank = rank_idx
                            inp[i] = indices[rank_idx].item()
                            for j in range(1, num_extra):
                                del inp[i + j]

                            if len(whole_text) > len(prop_token_text):
                                suffix = whole_text[len(prop_token_text):]
                                suffix_tokens = enc.encode(suffix)  # a list
                                inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                            break
                else:
                    print('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, inp[i], text))
                    rank = 0

            selection = rank

            # Calculate new range as ints
            new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            # Convert range to bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(
                reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

            # Emit most significant bits which are now fixed and update interval
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            if i == len(inp) - 1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]

            # Get the mask and apply it to the recovered bits
            mask_bits = mask_generator.generate_bits(precision)
            for b in range(0, len(new_bits)):
                new_bits[b] = new_bits[b] ^ mask_bits[b]
            message += new_bits

            # Update history with new token
            prev = torch.tensor([inp[i]], device=device, dtype=torch.long)

            i += 1

    return message


def encode_arithmetic(model, enc, message, context, finish_sent=False, device='cuda', temp=1.0, precision=16,
                      topk=50000):
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2 ** precision
    threshold = 2 ** (-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    prev = context
    output = context
    past = None

    total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0  # in bits
    total_entropy_ptau = 0
    total_num_sents = 0

    with torch.no_grad():
        i = 0
        sent_finish = False
        while i < len(message) or (finish_sent and not sent_finish):
            logits, past = model(prev.unsqueeze(0), past=past)
            past = limit_past(past)
            logits[0, -1, -1] = -1e20  # endoftext token can't happen
            logits[0, -1, 628] = -1e20  # 2 newlines token can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)

            # conditions for having reached the end of the message
            if i >= len(message):
                selection = 0
                sent_finish = is_sent_finish(indices[selection].item(), enc)
            else:
                # Cutoff low probabilities that would be rounded to 0
                cur_int_range = cur_interval[1] - cur_interval[0]
                cur_threshold = 1 / cur_int_range
                k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
                probs_temp_int = probs_temp[:k]  # Cutoff all but top k

                # Rescale to correct range
                probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range

                # Round probabilities to integers given precision
                probs_temp_int = probs_temp_int.round().long()
                cum_probs = probs_temp_int.cumsum(0)

                # Remove any elements from the bottom if rounding caused the total prob to be too large
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]

                # Add any mass to the top if removing/rounding causes the total prob to be too small
                cum_probs += cur_int_range - cum_probs[-1]  # add

                # Get out resulting probabilities
                probs_final = cum_probs.clone()
                probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                # Convert to position in range
                cum_probs += cur_interval[0]

                # Get selected index based on binary fraction from message bits
                message_bits = message[i:i + precision]
                if i + precision > len(message):
                    message_bits = message_bits + [0] * (i + precision - len(message))
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                # Calculate new range as ints
                new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(
                    reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

                # Consume most significant bits which are now fixed and update interval
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                i += num_bits_encoded

                new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded
                new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

                cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
                cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1  # +1 here because upper bound is exclusive

                # Gather statistics
                total_log_probs += log_probs[selection].item()

                q = probs_final.double() / probs_final.sum()
                logq = q.log()
                total_kl += kl(q, logq, log_probs[:len(q)])
                total_entropy_ptau += entropy(probs_temp, log_probs_temp)
                total_num_for_stats += 1

            # Update history with new token
            prev = indices[selection].view(1)
            output = torch.cat((output, prev))
            total_num += 1

            # For text->bits->text
            partial = enc.decode(output[len(context):].tolist())
            if '<eos>' in partial:
                break

    avg_NLL = -total_log_probs / total_num_for_stats
    avg_KL = total_kl / total_num_for_stats
    avg_Hq = total_entropy_ptau / total_num_for_stats
    words_per_bit = total_num_for_stats / i

    return output[len(context):].tolist(), avg_NLL, avg_KL, words_per_bit, avg_Hq


def decode_arithmetic(model, enc, text, context, device='cuda', temp=1.0, precision=16, topk=50000):
    # inp is a list of token indices
    # context is a list of token indices
    inp = enc.encode(text)
    # common BPE error case: 128, 128 (2 newlines) is interpretted as 628 (2 newlines)
    i = 0
    while i < len(inp):
        if inp[i] == 628:
            inp[i] = 198
            inp[i + 1:i + 1] = [198]
            i += 2
        else:
            i += 1

    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2 ** precision
    threshold = 2 ** (-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    prev = context
    past = None
    message = []
    with torch.no_grad():
        i = 0
        while i < len(inp):
            logits, past = model(prev.unsqueeze(0), past=past)
            past = limit_past(past)
            logits[0, -1, -1] = -1e10  # endoftext can't happen
            logits[0, -1, 628] = -1e10  # 2 newlines can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)

            # Cutoff low probabilities that would be rounded to 0
            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1 / cur_int_range
            k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
            probs_temp_int = probs_temp[:k]  # Cutoff all but top k

            # Rescale to correct range
            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range

            # Round probabilities to integers given precision
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)

            # Remove any elements from the bottom if rounding caused the total prob to be too large
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                k = overfill_index[0].item()

            # Add any mass to the top if removing/rounding causes the total prob to be too small
            cum_probs += cur_int_range - cum_probs[-1]  # add

            # Covnert to position in range
            cum_probs += cur_interval[0]

            rank = (indices == inp[i]).nonzero().item()

            # Handle most errors that could happen because of BPE with heuristic
            if rank >= k:
                true_token_text = enc.decoder[inp[i]]
                for rank_idx in range(k):
                    prop_token_text = enc.decoder[indices[rank_idx].item()]
                    # common case that is not caught
                    if inp[i] == 128 and indices[rank_idx] == 198:
                        rank = rank_idx
                        inp[i] = indices[rank_idx].item()
                        break

                    # Is there a more likely prefix token that could be the actual token generated?
                    if len(prop_token_text) <= len(true_token_text) and \
                            prop_token_text == true_token_text[:len(prop_token_text)]:
                        rank = rank_idx
                        suffix = true_token_text[len(prop_token_text):]
                        suffix_tokens = enc.encode(suffix)  # a list
                        inp[i] = indices[rank_idx].item()
                        inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                        break

                    # Is there a more likely longer token that could be the actual token generated?
                    elif len(prop_token_text) > len(true_token_text) and \
                            true_token_text == prop_token_text[:len(true_token_text)]:
                        whole_text = true_token_text
                        num_extra = 1
                        while len(whole_text) < len(prop_token_text):
                            whole_text += enc.decoder[inp[i + num_extra]]
                            num_extra += 1
                        if prop_token_text == whole_text[:len(prop_token_text)]:
                            rank = rank_idx
                            inp[i] = indices[rank_idx].item()
                            for j in range(1, num_extra):
                                del inp[i + j]

                            if len(whole_text) > len(prop_token_text):
                                suffix = whole_text[len(prop_token_text):]
                                suffix_tokens = enc.encode(suffix)  # a list
                                inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                            break
                else:
                    print('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, inp[i], text))
                    rank = 0

            selection = rank

            # Calculate new range as ints
            new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            # Convert range to bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(
                reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

            # Emit most significant bits which are now fixed and update interval
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            if i == len(inp) - 1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]
            message += new_bits

            new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded
            new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

            cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
            cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1  # +1 here because upper bound is exclusive

            # Update history with new token
            prev = torch.tensor([inp[i]], device=device, dtype=torch.long)
            i += 1

    return message


def _save_chw_image(x: torch.Tensor, path: str) -> None:
    """Save a CHW tensor in [-1, 1] as an RGB PNG."""
    if x.dim() != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected CHW with C=3, got {tuple(x.shape)}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x_np = ((x.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype("uint8")
    Image.fromarray(x_np).save(path)


@torch.no_grad()
def sample_unconditional_codebook(
    model,
    h: int = 16,
    w: int = 16,
    temperature: float = 1.0,
    top_k: Optional[int] = 100,
    outdir: Optional[str] = None,
    save_every: int = 0,
) -> torch.Tensor:
    """"
    Sample a full (h*w) sequence of VQ codebook indices using model.transformer, starting from SOS.

    Returns:
        z_indices: LongTensor shape (1, h*w) suitable for model.decode_to_img(z_indices, zshape)
    """
    model.eval()
    device = next(model.parameters()).device

    steps = h * w
    vocab = int(model.transformer.config.vocab_size)
    block_size = int(model.transformer.get_block_size())
    sos = int(getattr(model, "sos_token", 0))

    seq = torch.tensor([[sos]], device=device, dtype=torch.long)  # (1, 1)
    generated = torch.empty((1, 0), device=device, dtype=torch.long)

    zshape = (1, 256, h, w)

    for step in range(steps):
        seq_cond = seq if seq.size(1) <= block_size else seq[:, -block_size:]

        logits, _ = model.transformer(seq_cond)      # (1, T, vocab)
        logits = logits[:, -1, :] / temperature      # (1, vocab)

        if top_k is not None:
            logits = model.top_k_logits(logits, top_k)

        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)  # (1, 1)

        seq = torch.cat([seq, next_tok], dim=1)
        generated = torch.cat([generated, next_tok], dim=1)

        if outdir is not None and save_every and ((step + 1) % save_every == 0 or (step + 1) == steps):
            # IMPORTANT: decode_to_img needs exactly (h*w) tokens
            padded = torch.zeros((1, steps), device=device, dtype=torch.long)
            padded[:, :generated.shape[1]] = generated
            x_preview = model.decode_to_img(padded, zshape).squeeze(0)  # CHW
            _save_chw_image(x_preview, os.path.join(outdir, f"preview_{step+1:04}.png"))

    return generated

@torch.no_grad()
def embed_message_in_image(model, dsets, message_str, key: bytes, nonce: bytes, init_context_steps: int = 3):
    temp = 0.95
    precision = 32
    temperature = 1.0
    batch_size = 1
    top_k = 10
    outdir = "examples"
    from torch.utils.data import _utils
    _T = TypeVar("_T")
    _T_co = TypeVar("_T_co", covariant=True)
    _worker_init_fn_t = Callable[[int], None]
    _collate_fn_t = Callable[[list[_T]], Any]
    default_collate: _collate_fn_t = _utils.collate.default_collate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate a full token grid from scratch (no initial image)

    if len(dsets.datasets) > 1:
        split = sorted(dsets.datasets.keys())[0]
        dset = dsets.datasets[split]
    else:
        dset = next(iter(dsets.datasets.values()))
    print("Dataset: ", dset.__class__.__name__)
    for start_idx in trange(0,len(dset)-batch_size+1,batch_size):
        indices = list(range(start_idx, start_idx+batch_size))
        example = default_collate([dset[i] for i in indices])

        x = model.get_input("image", example).to(model.device)
        for i in range(x.shape[0]):
            save_image(x[i], os.path.join(outdir, "originals",
                                          "{:06}.png".format(indices[i])))

        cond_key = model.cond_stage_key
        c = model.get_input(cond_key, example).to(model.device)

        scale_factor = 1.0
        quant_z, z_indices = model.encode_to_z(x) # quant_z is the tensor representation of x, z_indices are the indices used to encode x
        quant_c, c_indices = model.encode_to_c(c)

        # cshape: (batch_size, num_channels, height, width)
        cshape = quant_z.shape

        xrec = model.first_stage_model.decode(quant_z)
        for i in range(xrec.shape[0]):
            save_image(xrec[i], os.path.join(outdir, "reconstructions",
                                             "{:06}.png".format(indices[i])))

        idx = torch.zeros_like(z_indices)
        idx = idx.reshape(cshape[0],cshape[2],cshape[3])

        cidx = c_indices
        cidx = cidx.reshape(quant_c.shape[0],quant_c.shape[2],quant_c.shape[3])

        sample = False

        for i in range(cshape[2]):
            # define window sizes for each patch over rows and columns (index 2 and 3)
            local_i, i_start, i_end = local_indeces(i, cshape[2])

            for j in range(cshape[3]):
                local_j, j_start, j_end = local_indeces(j, cshape[3])


                patch = idx[:,i_start:i_end,j_start:j_end]
                patch = patch.reshape(patch.shape[0],-1)
                cpatch = cidx[:, i_start:i_end, j_start:j_end]
                cpatch = cpatch.reshape(cpatch.shape[0], -1)
                patch = torch.cat((cpatch, patch), dim=1)

                logits,_ = model.transformer(patch[:,:-1])
                logits = logits[:, -256:, :]
                logits = logits.reshape(cshape[0],16,16,-1)
                logits = logits[:,local_i,local_j,:]

                logits = logits/temperature

                if top_k is not None:
                    logits = model.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                idx[:,i,j] = ix

        xsample = model.decode_to_img(idx[:,:cshape[2],:cshape[3]], cshape)
        for i in range(xsample.shape[0]):
            save_image(xsample[i], os.path.join(outdir, "samples",
                                                "{:06}.png".format(indices[i])))

    print("Done")
    return

    z_indices = sample_unconditional_codebook(
        model,
        h=16,
        w=16,
        temperature=1.0,
        top_k=100,
        outdir=debug_dir,
        save_every=32,   # set 0 to disable intermediate previews
    )

    # Decode final image
    zshape = (1, 256, 16, 16)
    context_image = model.decode_to_img(z_indices, zshape).squeeze(0)  # CHW in [-1, 1]
    _save_chw_image(context_image, os.path.join(debug_dir, "final.png"))

    # Context is the generated token sequence (if you need it later)
    context = z_indices[0].tolist()

    return context_image

    # "Context is done once generated the first three tokens":
    # here your context could simply be `z_indices` (tokens), and `context_image` is the visualization.
    return context_image

    # message = decode_arithmetic(
    #     model, message_str, precision=40, topk=60000, device=device)
    #
    # # Next encode bits into cover text, using arbitrary context
    # Hq = 0
    # out, nll, kl, words_per_bit, Hq = encode_meteor(model, message, temp=temp,
    #                                                 finish_sent=finish_sent,
    #                                                 precision=precision, topk=topk, device=device, is_sort=meteor_sort,
    #                                                 randomize_key=meteor_random, input_key=key, input_nonce=nonce)
    # text = enc.decode(out)
    #
    # print("=" * 40 + " Encoding " + "=" * 40)
    # print(text)
    # print('=> ppl: %0.2f, kl: %0.3f, words/bit: %0.2f, bits/word: %0.2f, entropy: %.2f' %
    #       (math.exp(nll), kl, words_per_bit, 1 / words_per_bit, Hq / 0.69315))
    # print("=" * 90)
    #
    # stats = {\
    #     "ppl": math.exp(nll),
    #     "kl": kl,
    #     "wordsbit": words_per_bit,
    #     "entropy": Hq / 0.69315
    # }
    # return text


def retrieve_message_from_image(model, image, key:bytes, nonce:bytes):
    temp = 0.95
    precision = 32
    topk = 50000
    device='cuda' if torch.cuda.is_available() else 'cpu'

    meteor_sort = False

    message_rec = decode_meteor(model, image, temp=temp,
                                precision=precision, topk=topk, device=device, is_sort=meteor_sort, input_key=key,
                                input_nonce=nonce)

    reconst = encode_arithmetic(
        model, message_rec, message_ctx, precision=40, topk=60000, device=device)
    reconst = enc.decode(reconst[0])

    print("=" * 40 + " Recovered Message " + "=" * 40)
    print(reconst)
    print("=" * 99)

    # Remove <eos>
    return reconst[:-5]

def main():
    dsets, model = get_vqgan_sflckr()

    message_text = "Message to encode"

    image = embed_message_in_image(model, dsets, message_text, b'\x03' * 64, b'\x01' * 64)
    # decoded_message = retrieve_message_from_image(model, image, b'\x03' * 64, b'\x01' * 64)


if __name__ == '__main__':
    main()


# import glob
# import os
# import sys
#
# import torch
# from omegaconf import OmegaConf
#
# from scripts.make_samples import load_model_and_dset, run_conditional, get_parser
#
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
#
# import argparse
# if __name__ == "__main__":
#     sys.path.append(os.getcwd())
#
#     # Parse input
#     parser = get_parser()
#     parser.add_argument('--message', required=True, type=str)
#     opt, unknown = parser.parse_known_args()
#     print(f"{'='*40} STARTING ENCODING OF MESSAGE \"{opt.message}\" {'='*40}")
#
#
#
#
#     # TODO Set up the model
#     # TODO Convert the message to bits
#     # TODO Calculate the length of the next encode
#     # TODO If the entropy is low pass (entropy threshold is a hyperparameter, and so part of the steganographic key)
#     # TODO Encode the next bits
#
#
#
#
#     ckpt = None
#     if opt.resume:
#         if not os.path.exists(opt.resume):
#             raise ValueError("Cannot find {}".format(opt.resume))
#         if os.path.isfile(opt.resume):
#             paths = opt.resume.split("/")
#             try:
#                 idx = len(paths) - paths[::-1].index("logs") + 1 # search the logs dir from the absolute path
#             except ValueError:
#                 idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
#             logdir = "/".join(paths[:idx])
#             ckpt = opt.resume
#         else:
#             assert os.path.isdir(opt.resume), opt.resume
#             logdir = opt.resume.rstrip("/")
#             ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
#         print(f"logdir:{logdir}")
#         base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
#         opt.base = base_configs + opt.base
#
#     if opt.config:
#         if type(opt.config) == str:
#             opt.base = [opt.config]
#         else:
#             opt.base = [opt.base[-1]]
#
#     configs = [OmegaConf.load(cfg) for cfg in opt.base]
#     cli = OmegaConf.from_dotlist(unknown)
#     if opt.ignore_base_data:
#         for config in configs:
#             if hasattr(config, "data"): del config["data"]
#     config = OmegaConf.merge(*configs, cli)
#
#     print(ckpt)
#     gpu = torch.cuda.is_available()
#     eval_mode = True
#     show_config = True
#     if show_config:
#         print(OmegaConf.to_container(config))
#
#     dsets, model, global_step = load_model_and_dset(config, ckpt, gpu, eval_mode)
#     print(f"Global step: {global_step}")
#
#     outdir = os.path.join(opt.outdir, "{:06}_{}_{}".format(global_step,
#                                                            opt.top_k,
#                                                            opt.temperature))
#     os.makedirs(outdir, exist_ok=True)
#     print("Writing samples to ", outdir)
#     for k in ["originals", "reconstructions", "samples"]:
#         os.makedirs(os.path.join(outdir, k), exist_ok=True)
#     run_conditional(model, dsets, outdir, opt.top_k, opt.temperature)
#
#
