import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import time
import os
from collections import defaultdict

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.pth_loader import CaptionDataset
import captioning.utils.eval_utils as eval_utils
import captioning.utils.misc as utils
from captioning.utils.rewards import init_scorer, get_self_critical_reward
from captioning.modules.loss_wrapper import LossWrapper

import pytorch_lightning as pl

import detectron2.utils.comm as d2comm
from detectron2.utils.env import seed_all_rng
seed_all_rng(1234)


class LitModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # Intilaize dataset
        self.dataset = CaptionDataset(opt)
        opt.vocab_size = self.dataset.vocab_size
        opt.seq_length = self.dataset.seq_length
        self.batch_size = opt.batch_size

        # Build model
        opt.vocab = self.dataset.get_vocab()
        model = models.setup(opt)
        print(model)
        del opt.vocab

        # wrapper with loss in it.
        lw_model = LossWrapper(model, opt)

        self.model = model
        self.lw_model = lw_model

        self.struc_flag = None
        self.sc_flag = None

    def forward(self, *args, **kwargs):
        """
        I hate this design. Never pretend it as a nn.Module
        """
        raise NotImplementedError

    def train_dataloader(self):
        train_dataset = torch.utils.data.Subset(
            self.dataset,
            self.dataset.split_ix['train']
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.dataset.collate_func
        )
        return train_loader

    def val_dataloader(self, split='val'):
        val_dataset = torch.utils.data.Subset(
            self.dataset,
            self.dataset.split_ix[split]
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            collate_fn=self.dataset.collate_func
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader('test')

    def training_step(self, data, batch_idx):
        sc_flag, struc_flag, drop_worst_flag = self.sc_flag, self.struc_flag, self.drop_worst_flag

        tmp = [data['fc_feats'], data['att_feats'],
               data['labels'], data['masks'], data['att_masks']]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        if int(os.getenv('M2_cider', '0')) != 0:
            data['gts'] = data['rawgts']
        model_out = self.lw_model(fc_feats, att_feats, labels, masks, att_masks,
                                  data['gts'], torch.arange(0, len(data['gts'])), sc_flag, struc_flag, drop_worst_flag)
        if not drop_worst_flag:
            loss = model_out.pop('loss').mean()
        else:
            loss = model_out.pop('loss')
            loss = torch.topk(loss, k=int(loss.shape[0] * (1-self.opt.drop_worst_rate)), largest=False)[0].mean()

        # Prepare for logging info
        data_time = self.trainer.profiler.recorded_durations["get_train_batch"][-1]
        data_time = torch.tensor(data_time)

        logger_logs = model_out.copy()
        if struc_flag or sc_flag:
            logger_logs['reward'] = model_out['reward'].mean()
            logger_logs['reward_var'] = model_out['reward'].var(1).mean()

        logger_logs['scheduled_sampling_prob'] = torch.tensor(
            self.model.ss_prob)
        logger_logs['training_loss'] = loss
        logger_logs['data_time'] = data_time

        for k, v in logger_logs.items():
            self.log(k, v, on_epoch=(k=='training_loss'), prog_bar=(k=='data_time'))
        # logged

        return loss

    def validation_step(self, data, batch_idx):
        model = self.model
        crit = self.lw_model.crit

        opt = self.opt
        eval_kwargs = {'dataset': opt.input_json}
        eval_kwargs.update(vars(opt))

        verbose = eval_kwargs.get('verbose', True)
        verbose_beam = eval_kwargs.get('verbose_beam', 0)
        verbose_loss = eval_kwargs.get('verbose_loss', 1)
        # num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
        # lang_eval = eval_kwargs.get('language_eval', 0)
        dataset = eval_kwargs.get('dataset', 'coco')
        beam_size = eval_kwargs.get('beam_size', 1)
        sample_n = eval_kwargs.get('sample_n', 1)
        remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
        # Use this nasty way to make other code clean since it's a global configuration
        os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)

        predictions = []
        n_predictions = []

        loss = torch.tensor(0)

        tmp = [data['fc_feats'], data['att_feats'],
                data['labels'], data['masks'], data['att_masks']]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            loss = crit(model(fc_feats, att_feats,
                              labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:])

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp_eval_kwargs = eval_kwargs.copy()
        tmp_eval_kwargs.update({'sample_n': 1})
        seq, seq_logprobs = model(
            fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        seq = seq.data
        entropy = - (F.softmax(seq_logprobs, dim=2) *
                        seq_logprobs).sum(2).sum(1) / ((seq > 0).to(seq_logprobs).sum(1)+1)
        perplexity = - \
            seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(
                2).sum(1) / ((seq > 0).to(seq_logprobs).sum(1)+1)

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(fc_feats.shape[0]):
                print('\n'.join([utils.decode_sequence(model.vocab, _[
                        'seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(model.vocab, seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent,
                        'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + \
                    '" vis/imgs/img' + \
                    str(len(predictions)) + '.jpg'  # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %
                        (entry['image_id'], entry['caption']))

        if sample_n > 1:
            eval_utils.eval_split_n(model, n_predictions, [
                                    fc_feats, att_feats, att_masks, data], eval_kwargs)

        output = {
            'loss': loss,
            'predictions': predictions,
            'n_predictions': n_predictions,
        }
        return output

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def split_epoch_end(self, outputs, split='val'):
        outputs = d2comm.gather(outputs)
        # master node
        if d2comm.is_main_process():
            assert self.trainer.node_rank == 0 and self.trainer.local_rank == 0
            outputs = sum(outputs, [])

            opt = self.opt
            loss_mean = sum([_['loss'].item()
                                 for _ in outputs]) / len(outputs)

            predictions = sum([_['predictions'] for _ in outputs], [])
            if len(outputs[0]['n_predictions']) != 0:
                n_predictions = sum([_['n_predictions'] for _ in outputs], [])
            else:
                n_predictions = []

            lang_stats = None
            if len(n_predictions) > 0 and 'perplexity' in n_predictions[0]:
                n_predictions = sorted(
                    n_predictions, key=lambda x: x['perplexity'])

            if not os.path.isdir('eval_results'):
                os.mkdir('eval_results')
            torch.save((predictions, n_predictions), os.path.join(
                'eval_results/', '.saved_pred_' + opt.id + '_' + split + '.pth'))

            if opt.language_eval:
                lang_stats = eval_utils.language_eval(
                    opt.input_json, predictions, n_predictions, vars(opt), split)

            if opt.reduce_on_plateau:
                optimizer = self.trainer.optimizers[0]
                if 'CIDEr' in lang_stats:
                    optimizer.scheduler_step(-lang_stats['CIDEr'])
                else:
                    optimizer.scheduler_step(loss_mean)

            out = {
                'loss': loss_mean
            }
            out.update(lang_stats)
            out['to_monitor'] = lang_stats['CIDEr'] if lang_stats is not None else -loss_mean
        else:
            out = {}

        out = d2comm.all_gather(out)[0]  # Only the one from master node
        assert len(out) > 0  # make sure the head has index 0

        # must all be tensors
        out = {k: torch.tensor(v) if not torch.is_tensor(
            v) else v for k, v in out.items()}

        return out

    def validation_epoch_end(self, outputs):
        out = self.split_epoch_end(outputs, 'val')
        out['val_loss'] = out.pop('loss')
        for k,v in out.items():
            self.log(k, v)
        return out

    def test_epoch_end(self, outputs):
        out = self.split_epoch_end(outputs, 'test')
        out['test_loss'] = out.pop('loss')
        out = {'test_'+k if 'test' not in k else k: v
               for k, v in out.items()}
        for k,v in out.items():
            self.log(k, v)
        return out
 
    def configure_optimizers(self):
        opt = self.opt
        model = self.model
        if opt.noamopt:
            # assert opt.caption_model in ['transformer', 'bert', 'm2transformer'], 'noamopt can only work with transformer'
            optimizer = utils.get_std_opt(
                model, optim_func=opt.optim, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
        elif opt.reduce_on_plateau:
            optimizer = utils.build_optimizer(model.parameters(), opt)
            optimizer = utils.ReduceLROnPlateau(optimizer,
                                                factor=opt.reduce_on_plateau_factor,
                                                patience=opt.reduce_on_plateau_patience)
        else:
            optimizer = utils.build_optimizer(model.parameters(), opt)
        return [optimizer], []

    def optimizer_step(self, epoch, batch_idx, optimizer,
                       optimizer_idx, *args, **kwargs):
        # warm up lr
        opt = self.opt
        iteration = self.trainer.global_step
        if opt.use_warmup and (iteration < opt.noamopt_warmup):
            opt.current_lr = opt.learning_rate * \
                (iteration+1) / opt.noamopt_warmup
            utils.set_lr(optimizer, opt.current_lr)

        super().optimizer_step(epoch, batch_idx, optimizer,
                               optimizer_idx, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """
        Save the model state dict as well as opt and vocab
        """
        state_dict = self.model.state_dict(*args, **kwargs)
        device = next(iter(state_dict.values())).device
        assert '_vocab' not in state_dict and '_opt' not in state_dict, 'Just in case'
        state_dict.update({
            '_vocab': utils.serialize_to_tensor(self.model.vocab).to(device),
            '_opt': utils.serialize_to_tensor(self.opt).to(device)
        })
        return state_dict

    def load_state_dict(self, state_dict=None, strict=True):
        if '_vocab' in state_dict:
            self.model.vocab = utils.deserialize(state_dict['_vocab'])
            del state_dict['_vocab']
        elif strict:
            raise KeyError
        if '_opt' in state_dict:
            saved_model_opt = utils.deserialize(state_dict['_opt'])
            del state_dict['_opt']
            opt = self.opt
            # Make sure the saved opt is compatible with the curren topt
            need_be_same = ["caption_model",
                            "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                if getattr(saved_model_opt, checkme) in ['updown', 'topdown'] and \
                        getattr(opt, checkme) in ['updown', 'topdown']:
                    continue
                assert getattr(saved_model_opt, checkme) == getattr(
                    opt, checkme), "Command line argument and saved model disagree on '%s' " % checkme
        elif strict:
            raise KeyError
        self.model.load_state_dict(state_dict, strict)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


class OnEpochStartCallback(pl.Callback):

    def on_train_epoch_start(self, trainer, pl_module):
        # Update lr/training stage/scheduled sampling prob etc.
        opt = pl_module.opt
        model = pl_module.model
        epoch = trainer.current_epoch
        optimizer = trainer.optimizers[0]

        if not opt.noamopt and not opt.reduce_on_plateau:
            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (
                    epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)  # set the decayed rate
        # Assign the scheduled sampling prob
        if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
            frac = (
                epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(opt.scheduled_sampling_increase_prob *
                              frac, opt.scheduled_sampling_max_prob)
            model.ss_prob = opt.ss_prob

        # If start self critical training
        if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
            sc_flag = True
            init_scorer(opt.cached_tokens)
        else:
            sc_flag = False

        # If start structure loss training
        if opt.structure_after != -1 and epoch >= opt.structure_after:
            struc_flag = True
            init_scorer(opt.cached_tokens)
        else:
            struc_flag = False

        # drop_worst flag
        if opt.drop_worst_after != -1 and epoch >= opt.drop_worst_after:
            drop_worst_flag = True
        else:
            drop_worst_flag = False

        pl_module.struc_flag = struc_flag
        pl_module.sc_flag = sc_flag
        pl_module.drop_worst_flag = drop_worst_flag


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):

    def on_keyboard_interrupt(self, trainer, pl_module):
        # Save model when keyboard interrupt
        filepath = os.path.join(self.dirpath, pl_module.opt.id+'_interrupt.ckpt')
        self._save_model(trainer, filepath=filepath)


opt = opts.parse_opt()

checkpoint_callback = ModelCheckpoint(
    dirpath=opt.checkpoint_path,
    filename=opt.id+'_{epoch}-{step}',
    save_last=True,
    save_top_k=1,
    verbose=True,
    monitor='to_monitor',
    mode='max',
)
checkpoint_callback.CHECKPOINT_NAME_LAST = opt.id+"_last"


tb_logger = pl.loggers.TensorBoardLogger(opt.checkpoint_path +
                                         '/lightning_logs/',
                                         name='',
                                         version=0)
wandb_logger = pl.loggers.WandbLogger(name=opt.id,
                                      id=opt.id,
                                      project='captioning',
                                      log_model=True)

print("""
val_image_use,
save_checkpoint_very
save_every_epoch,
save_history-ckpt will be ignored.
""")

# Lightning defines batch size as batch size per gpu
assert opt.batch_size % torch.cuda.device_count() == 0
opt.batch_size = opt.batch_size // torch.cuda.device_count()

# If resume from last checkpoint
if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, f'{opt.id}_last.ckpt')):
    resume_from = os.path.join(opt.start_from, f'{opt.id}_last.ckpt')
else:
    resume_from = None

lit = LitModel(opt)
# warning grad_clip_mode is ignored.
trainer = pl.Trainer(
    logger=[tb_logger, wandb_logger],
    callbacks=[
        OnEpochStartCallback(),
        pl.callbacks.LearningRateMonitor(),
        checkpoint_callback,
    ],
    default_root_dir=opt.checkpoint_path,
    resume_from_checkpoint=resume_from,
    accelerator='ddp',
    check_val_every_n_epoch=1,
    max_epochs=opt.max_epochs,
    gradient_clip_algorithm=opt.grad_clip_mode,
    gradient_clip_val=opt.grad_clip_value,
    gpus=torch.cuda.device_count(),
    log_gpu_memory='min_max',
    log_every_n_steps=opt.losses_log_every,
    profiler='simple',
    num_sanity_val_steps=0,
    # limit_train_batches=100,
    # progress_bar_refresh_rate=0,
    # fast_dev_run=True,
)

if os.getenv('EVALUATE', '0') == '1':
    lit.load_state_dict(
        torch.load(resume_from, map_location='cpu')['state_dict'], strict=False)
    trainer.test(lit)
else:
    trainer.fit(lit)
