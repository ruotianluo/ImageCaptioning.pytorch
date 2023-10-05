import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.rewards import get_scores, get_self_cider_scores


def masked_mean(tensor, mask, dim=None, keepdim=False):
    assert tensor.shape == mask.shape, 'tensor and mask must have the same shape'
    if dim is None:
        return torch.sum(tensor * mask) / torch.sum(mask)
    else:
        return torch.sum(tensor * mask, dim=dim, keepdim=keepdim) / torch.sum(
            mask, dim=dim, keepdim=keepdim)


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward, reduction='mean'):
        N,L = input.shape[:2]
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = (seq>0).to(input)
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        output = - input * reward * mask

        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output


class StructureLosses(nn.Module):
    """
    This loss is inspired by Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018).
    """
    def __init__(self, opt):
        super(StructureLosses, self).__init__()
        self.opt = opt
        self.loss_type = opt.structure_loss_type

    def forward(self, input, seq, data_gts, reduction='mean'):
        """
        Input is either logits or log softmax
        """
        out = {}

        batch_size = input.size(0)# batch_size = sample_size * seq_per_img
        seq_per_img = batch_size // len(data_gts)

        assert seq_per_img == self.opt.train_sample_n, seq_per_img

        mask = (seq>0).to(input)
        mask = torch.cat([mask.new_full((mask.size(0), 1), 1), mask[:, :-1]], 1)

        scores = get_scores(data_gts, seq, self.opt)
        scores = torch.from_numpy(scores).type_as(input).view(-1, seq_per_img)
        out['reward'] = scores #.mean()
        if self.opt.entropy_reward_weight > 0:
            entropy = - (F.softmax(input, dim=2) * F.log_softmax(input, dim=2)).sum(2).data
            entropy = (entropy * mask).sum(1) / mask.sum(1)
            print('entropy', entropy.mean().item())
            scores = scores + self.opt.entropy_reward_weight * entropy.view(-1, seq_per_img)
        # rescale cost to [0,1]
        costs = - scores
        if self.loss_type == 'risk' or self.loss_type == 'softmax_margin':
            costs = costs - costs.min(1, keepdim=True)[0]
            costs = costs / costs.max(1, keepdim=True)[0]
        # in principle
        # Only risk need such rescale
        # margin should be alright; Let's try.

        # Gather input: BxTxD -> BxT
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        if self.loss_type == 'seqnll':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            target = costs.min(1)[1]
            output = F.cross_entropy(input, target, reduction=reduction)
        elif self.loss_type == 'risk':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1)
            input = input.view(-1, seq_per_img)

            output = (F.softmax(input.exp()) * costs).sum(1).mean()
            assert reduction=='mean'

            # test
            # avg_scores = input
            # probs = F.softmax(avg_scores.exp_())
            # loss = (probs * costs.type_as(probs)).sum() / input.size(0)
            # print(output.item(), loss.item())

        elif self.loss_type == 'max_margin':
            # input is logits
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input).max(1)[0] / 2
            output = output.mean()
            assert reduction=='mean'

            # sanity test
            # avg_scores = input + costs
            # scores_with_high_target = avg_scores.clone()
            # scores_with_high_target.scatter_(1, costs.min(1)[1].view(-1, 1), 1e10)

            # target_and_offender_index = scores_with_high_target.sort(1, True)[1][:, 0:2]
            # avg_scores = avg_scores.gather(1, target_and_offender_index)
            # target_index = avg_scores.new_zeros(avg_scores.size(0), dtype=torch.long)
            # loss = F.multi_margin_loss(avg_scores, target_index, size_average=True, margin=0)
            # print(loss.item() * 2, output.item())

        elif self.loss_type == 'multi_margin':
            # input is logits
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input)
            output = output.mean()
            assert reduction=='mean'

            # sanity test
            # avg_scores = input + costs
            # loss = F.multi_margin_loss(avg_scores, costs.min(1)[1], margin=0)
            # print(output, loss)

        elif self.loss_type == 'softmax_margin':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target, reduction=reduction)

        elif self.loss_type == 'real_softmax_margin':
            # input is logits
            # This is what originally defined in Kevin's paper
            # The result should be equivalent to softmax_margin
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target, reduction=reduction)

        elif self.loss_type == 'new_self_critical':
            """
            A different self critical
            Self critical uses greedy decoding score as baseline;
            This setting uses the average score of the rest samples as baseline
            (suppose c1...cn n samples, reward1 = score1 - 1/(n-1)(score2+..+scoren) )
            """
            baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1)
            scores = scores - baseline
            # self cider used as reward to promote diversity (not working that much in this way)
            if getattr(self.opt, 'self_cider_reward_weight', 0) > 0:
                _scores = get_self_cider_scores(data_gts, seq, self.opt)
                _scores = torch.from_numpy(_scores).type_as(scores).view(-1, 1)
                _scores = _scores.expand_as(scores - 1)
                scores += self.opt.self_cider_reward_weight * _scores
            output = - input * mask * scores.view(-1, 1)
            if reduction == 'none':
                output = output.sum(1) / mask.sum(1)
            elif reduction == 'mean':
                output = torch.sum(output) / torch.sum(mask)

        elif self.loss_type == 'best_of_n':
            """
            Supervised by the highest prediction.
            """
            # Convert scores to 0,1 tensor where 1 where score is the highest.
            scores = (scores == scores.max(1, keepdim=True)[0]).float()
            output = - input * mask * scores.view(-1, 1)
            if reduction == 'none':
                output = output.sum(1) / mask.sum(1)
            elif reduction == 'mean':
                output = torch.sum(output) / torch.sum(mask)

        out['loss'] = output
        return out

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask, reduction='mean'):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        N,L = input.shape[:2]
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)].to(input)

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask

        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None

    def forward(self, input, target, mask, reduction='mean'):
        N,L = input.shape[:2]
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)
        mask = mask.reshape(-1).to(input)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        output = self.criterion(input, true_dist).sum(1) * mask

        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output

class PPOLoss(nn.Module):
    def __init__(self, opt, model: nn.Module):
        super(PPOLoss, self).__init__()
        self.opt = opt
        self.cliprange = getattr(opt, 'ppo_cliprange', 0.2)
        self.kl_coef = getattr(opt, 'ppo_kl_coef', 0.02)

        if opt.use_ppo == 1:
            assert opt.ppo_old_model_path is not None, 'Must provide old model path for PPO'
            self.old_model = copy.deepcopy(model)
            logging.warning(
                'Make sure you are using the same model for PPO loss and the vocab must be the same.'
            )
            state_dict = torch.load(opt.ppo_old_model_path)
            if 'pytorch-lightning_version' in state_dict:
                # It is a lightning checkpoint.
                state_dict = state_dict['state_dict']
                del state_dict['_vocab']
                del state_dict['_opt']
            self.old_model.load_state_dict(state_dict)
            # Set old model to eval mode and disable gradient.
            self.old_model.eval()
            for p in self.old_model.parameters():
                p.requires_grad = False

    def forward(self, input, seq, data_gts, fc_feats, att_feats, att_masks, reduction='mean'):
        """
        Input is either logits or log softmax
        """
        out = {}

        batch_size = input.size(0)# batch_size = sample_size * seq_per_img
        seq_per_img = batch_size // len(data_gts)

        assert seq_per_img == self.opt.train_sample_n, seq_per_img

        mask = (seq>0).to(input)
        mask = torch.cat([mask.new_full((mask.size(0), 1), 1), mask[:, :-1]], 1)

        scores = get_scores(data_gts, seq, self.opt)
        scores = torch.from_numpy(scores).type_as(input).view(-1, seq_per_img)
        out['reward'] = scores #.mean()

        # NSC type reward/advantage.
        baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1)
        scores = scores - baseline

        # Gather input: BxTxD -> BxT
        word_logprob = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        logprobs = input

        # Prepend bos.
        model_input_seq = torch.cat([seq.new_full((seq.size(0), 1), 0), seq[:, :-1]], 1)

        with torch.no_grad():
            self.old_model.eval()
            logprobs_old = self.old_model(fc_feats, att_feats, model_input_seq, att_masks)
        word_logprob_old = logprobs_old.gather(2, seq.unsqueeze(2)).squeeze(2)

        ratio = torch.exp(word_logprob - word_logprob_old)

        # B x seq_per_img -> B*seq_per_img x 1
        scores = scores.view(-1, 1)

        pg_losses = -scores * ratio
        pg_losses2 = -scores * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)

        # follow https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L77.

        pg_loss = torch.max(pg_losses, pg_losses2)
        # the openai baseline deos not have kl loss.
        # In instruct gpt, it seems to be KL(RL|SFT), while in PPO, it is KL(SFT|RL).
        # we just guess then.
        # https://github.com/openai/summarize-from-feedback , no training code....
        # From summarize feedback, the reward is sequence level.
        # The lucdirian is word level it seems, anyway, it also has entropy what so ever, pretty weird.
        # The instructgpt does not say, using PPO clip in eq 2, but in appendix they say how they use clip.
        # The original PPO does not use both clip and kl, they are seprate two ways, but here it seems they use them together.

        # N,L
        kl_loss = F.kl_div(logprobs, logprobs_old, reduction='none', log_target=True).sum(-1)
        out['pg_loss'] = masked_mean(pg_loss, mask)
        out['kl_loss'] = masked_mean(kl_loss, mask)
        out['clipfrac'] = masked_mean(((ratio - 1.0).abs() > self.cliprange).float(), mask)
        if reduction == 'none':
            loss = pg_loss + self.kl_coef * kl_loss
            out['loss'] = masked_mean(loss, mask, 1)
        elif reduction == 'mean':
            loss = out['pg_loss'] + self.kl_coef * out['kl_loss']
            out['loss'] = loss
        return out