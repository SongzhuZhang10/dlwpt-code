import argparse
import datetime
import hashlib
import os
import shutil
import sys

import numpy as np
from matplotlib import pyplot

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

import p2ch14.dsets
import p2ch14.model

from util.util import enumerateWithEstimate
from util.logconf import logging
import traceback

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

METRICS_LABEL_NDX = 0
METRICS_PRED_LABEL_NDX = 1
METRICS_PRED_POS_NDX = 2
METRICS_LOSS_NDX = 3
METRICS_SIZE = 4

class ClassificationTrainingApp:
    def __init__(self, sys_argv=None):
        # -------- argument parsing --------
        if sys_argv is None:
            # Exclude the first element which contains the name of the script being run
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', default=24, type=int,
                            help='Batch size to use for training')

        parser.add_argument('--num-workers', default=8, type=int,
                            help='Number of worker processes for background data loading')

        parser.add_argument('--epochs', default=1, type=int,
                            help='Number of epochs to train for')

        parser.add_argument('--dataset', default='LunaDataset',
                            help='(Will be overridden automatically; no manual input needed)')

        parser.add_argument('--model', default='LunaModel',
                            help='Name of the model to be used.')

        parser.add_argument('--malignant', action='store_true', default=False,
                            help='Train the model to classify nodules as benign or malignant.')

        parser.add_argument('--pretrained-model-path', default='',
                            help='Path to a pretrained model checkpoint used for fine-tuning.')

        parser.add_argument('--finetune-depth', type=int, default=1,
                            help='Number of blocks (counted from the head) to include in fine-tuning')

        parser.add_argument('--resume-from', default='',
            help='Path to a checkpoint produced by saveModel; training resumes from it.')

        parser.add_argument('--tb-prefix', default='p2ch14', #TODO: Naming
                            help='TensorBoard run prefix')
        parser.add_argument('comment', nargs='?', default='tb',
                            help='Comment suffix for TensorBoard run.')

        self.cli_args = parser.parse_args(sys_argv)

        if self.cli_args.malignant and not self.cli_args.pretrained_model_path:
            print('Error: --pretrained-model-path must be provided when --malignant is set.', file=sys.stderr)
            sys.exit(1)

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        # data-augmentation defaults
        # These values were empirically chosen to have a reasonable impact, but better values probably exist.
        self.augmentation_dict = {
            'flip': True,
            'offset': 0.1,
            'scale': 0.2,
            'rotate': True,
            'noise': 25.0,
        }

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

        self.start_epoch = 1
        if self.cli_args.resume_from:
            if not os.path.exists(self.cli_args.resume_from):
                log.error(f"Checkpoint not found: {self.cli_args.resume_from}")
                sys.exit(1)
            self.start_epoch = self._restore_from_checkpoint(self.cli_args.resume_from)


    def _restore_from_checkpoint(self, ckpt_path: str) -> int:
        log.info(f"Resuming training from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # ---------- model ----------
        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module          # unwrap if needed
        model.load_state_dict(ckpt['model_state'])

        # ---------- optimizer ----------
        self.optimizer.load_state_dict(ckpt['optimizer_state'])

        # ---------- bookkeeping ----------
        self.totalTrainingSamples_count = ckpt.get('totalTrainingSamples_count', 0)
        return ckpt.get('epoch', 0) + 1   # next epoch index


    def initModel(self):
        model_cls = getattr(p2ch14.model, self.cli_args.model)
        model = model_cls()
        print(f"Using model class: {model_cls.__name__}")

        # If the path to a pretrained model exists, it means we will be doing finetuning.
        if self.cli_args.pretrained_model_path:
            # Get a list of names (strings) of trainable blocks
            # model.named_children() returns all top-level child modules of the model as (name, module) pairs.
            # child_module.parameters() returns a generator of the submodule's parameters.
            # If a submodule has no trainable parameters (like ReLU), then list(child_module.parameters()) will be empty.
            trainable_module_names = [
                name for name, child_module in model.named_children()
                # Filter out top-level modules that have trainable parameters
                if len(list(child_module.parameters())) > 0
            ]
            
            # Get a list of strings specifying which blocks should be fine-tuned.
            finetune_blocks = trainable_module_names[-self.cli_args.finetune_depth:]
            log.info(f"finetuning from {self.cli_args.pretrained_model_path}, blocks {' '.join(finetune_blocks)}")

            try:
                # load a PyTorch model checkpoint file and then move it to CPU memory to prevent possible CUDA errors.
                model_checkpoint = torch.load(self.cli_args.pretrained_model_path, map_location='cpu')
            except FileNotFoundError:
                log.error(f"Pretrained model file not found at: '{self.cli_args.pretrained_model_path}'")
                sys.exit(1)
            except RuntimeError as e:
                log.error(f"RuntimeError while loading pretrained model: {e}")
                log.debug(traceback.format_exc())
                sys.exit(1)
            except Exception as e:
                log.error(f"Unexpected error while loading pretrained model: {e}")
                log.debug(traceback.format_exc())
                sys.exit(1)

            """
            Partially load weights from a saved model checkpoint into a PyTorch model. It skips the last block during loading
            because the last one is usually task-specific, and you want to replace it when fine-tuning for a different task.
            """
            model.load_state_dict(
                {
                    k: v for k, v in model_checkpoint['model_state'].items()
                    if k.split('.')[0] not in trainable_module_names[-1]
                },
                strict=False, # Ignore missing or extra keys without raising errors.
            )

            for n, p in model.named_parameters():
                if n.split('.')[0] not in finetune_blocks:
                    # Disable gradient tracking, which means optimizer will not update this parameter tensor during training
                    p.requires_grad_(False)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            # If multiple GPUs exist, the model can be parallelized across them to speed up training.
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

        return model

    def initOptimizer(self):
        lr = 0.003 if self.cli_args.pretrained_model_path else 0.001
        return SGD(self.model.parameters(), lr=lr, weight_decay=1e-4)
        #return Adam(self.model.parameters(), lr=3e-4)

    def initTrainDl(self):
        """
        Select the appropriate training dataset automatically
        based on the --malignant flag, then build the DataLoader.
        """
        dataset_name = 'MalignantLunaDataset' if self.cli_args.malignant else 'LunaDataset'
        ds_cls = getattr(p2ch14.dsets, dataset_name)

        # TODO: Not passing augmentation_dict to training data?
        train_ds = ds_cls(
            val_stride=10,
            isValSet_bool=False,
            ratio_int=1,  # positive:negative = 1:1
            augmentation_dict=self.augmentation_dict
        )

        print(f"[INFO] Using {dataset_name} for training")

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return train_dl

    def initValDl(self):
        """
        Select the appropriate validation dataset automatically
        based on the --malignant flag, then build the DataLoader.
        """
        dataset_name = 'MalignantLunaDataset' if self.cli_args.malignant else 'LunaDataset'
        ds_cls = getattr(p2ch14.dsets, dataset_name)

        val_ds = ds_cls(
            val_stride=10,
            isValSet_bool=True,
        )

        print(f"[INFO] Using {dataset_name} for validation")

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )
        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)
            prefix = 'mal' if self.cli_args.malignant else 'cls'
            
            self.trn_writer = SummaryWriter(
                log_dir=log_dir + f'-trn_{prefix}-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + f'-val_{prefix}-' + self.cli_args.comment)


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_score = 0.0
        """
        When training from scratch, early model changes are unstable and validation
        is costly, so it's done every 5 epochs to save time.
        In fine-tuning, the model already performs well and changes quickly, so it's
        validated every epoch to catch small improvements.
        """
        validation_cadence = 5 if not self.cli_args.pretrained_model_path else 1

        end_epoch = self.start_epoch + self.cli_args.epochs - 1
        
        for epoch_ndx in range(self.start_epoch, end_epoch + 1):

            total_gpus = torch.cuda.device_count() if self.use_cuda else 1
            log.info(
                f"Epoch {epoch_ndx}/{self.cli_args.epochs} — "
                f"Training: {len(train_dl)} batches/epoch | "
                f"Validation: {len(val_dl)} batches/epoch | "
                f"Batch size: {self.cli_args.batch_size} samples/GPU x "
                f"{total_gpus} GPUs"
            )

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)

                # training a malignant classifier (--malignant is passed), the
                # saved model uses the 'mal' prefix instead of 'cls'.
                prefix = 'mal' if self.cli_args.malignant else 'cls'
                
                if self.cli_args.pretrained_model_path:
                    prefix += f"-finetune-depth-{self.cli_args.finetune_depth}"

                self.saveModel(prefix, epoch_ndx, score == best_score)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()


    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        train_dl.dataset.shuffleSamples()

        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        
        # batch_tup is defined in `sampleFromCandidateInfo_tup` method which
        # is invoked internally in `__getitem__` method.
        for batch_ndx, batch_tup in batch_iter:
            # Clear old gradients from the previous training step, ensuring
            # each batch has a clean start for gradient computation.
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx=batch_ndx,
                batch_tup=batch_tup,
                batch_size=train_dl.batch_size,
                metrics_g=trnMetrics_g,
                augment=True
            )

            # Compute new gradients w.r.t loss
            loss_var.backward()
            # Update model parameters using current gradients
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')


    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx=batch_ndx,
                    batch_tup=batch_tup,
                    batch_size=val_dl.batch_size,
                    metrics_g=valMetrics_g,
                    augment=False
                )

        return valMetrics_g.to('cpu')



    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g, augment=True):
        input_t, label_t, index_t, series_uid, center_irc_t = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        index_g = index_t.to(self.device, non_blocking=True)

        if augment:
            input_g = p2ch14.model.augment3d(input_g)

        logits_g, probability_g = self.model(input_g)

        # The second column indicates if the sample is a nodule.
        # Use reduction="none" to return one loss per sample in the batch.
        loss_g = nn.functional.cross_entropy(logits_g, label_g[:, 1], reduction="none")
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        """
        predLabel_g is a tensor of shape [batch_size] containing the predicted class index
        (e.g., 0 or 1) for each sample in the batch.
        """
        _, predLabel_g = torch.max(probability_g, dim=1, keepdim=False, out=None)

        # Each column represents a sample in the batch.
        # The index tensor stores the per-sample 1 or 0 which indicates if the sample is positive or not.
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = index_g
        metrics_g[METRICS_PRED_LABEL_NDX, start_ndx:end_ndx] = predLabel_g
        # probability_g[:,1] is the predicted probability of positive samples
        metrics_g[METRICS_PRED_POS_NDX, start_ndx:end_ndx] = probability_g[:,1]
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g

        # Compute batch-averaged loss
        return loss_g.mean()


    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        self.initTensorboardWriters()
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        if self.cli_args.dataset == 'MalignantLunaDataset':
            pos = 'mal'
            neg = 'ben'
        else:
            pos = 'pos'
            neg = 'neg'

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] == 0
        negPredLabel_mask = metrics_t[METRICS_PRED_LABEL_NDX] == 0

        posLabel_mask = ~negLabel_mask
        posPredLabel_mask = ~negPredLabel_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        # Number of true negatives (i.e., actual negatives that are predicted as negative).
        neg_correct = int((negLabel_mask & negPredLabel_mask).sum())
        pos_correct = int((posLabel_mask & posPredLabel_mask).sum())
        
        # Number of actual negatives that were predicted as positive, which is the false positive count.
        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct
        
        truePos_count = pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / metrics_t.shape[1] * 100
        metrics_dict['correct/neg'] = (neg_correct) / neg_count * 100
        metrics_dict['correct/pos'] = (pos_correct) / pos_count * 100

        precision = metrics_dict['pr/precision'] = truePos_count / np.float64(truePos_count + falsePos_count)
        recall = metrics_dict['pr/recall'] = truePos_count / np.float64(truePos_count + falseNeg_count)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)

        # Create a 1D tensor of 100 (default) evenly spaced threshold values ranging from 1.0 down to 0.0
        threshold = torch.linspace(1.0, 0.0, steps=100)
        
        # TPR (True Positive Rate) at each threshold = number of true positives at each threshold / number of actual positives
        tpr = (metrics_t[None, METRICS_PRED_POS_NDX, posLabel_mask] >= threshold[:, None]).sum(1).float() / pos_count
        fpr = (metrics_t[None, METRICS_PRED_POS_NDX, negLabel_mask] >= threshold[:, None]).sum(1).float() / neg_count

        # Compute change in FPR (delta x)
        fp_diff = fpr[1:]-fpr[:-1]
        # Compute average TPR between consecutive points (average y)
        # It is used to estimate the height of the trapezoid between fpr[i] and fpr[i+1]
        tp_avg  = (tpr[1:]+tpr[:-1])/2
        # Compute the area under the curve as the sum of trapezoid areas
        auc = (fp_diff * tp_avg).sum()
        metrics_dict['auc'] = auc

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
                 + "{correct/all:-5.1f}% correct, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score, "
                 + "{auc:.4f} auc"
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
                 + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_' + neg,
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
                 + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_' + pos,
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for metric_name, value in metrics_dict.items():
            # pos and neg are dynamic strings, depending on the task (e.g., pos may be replaced by mal)
            metric_name = metric_name.replace('pos', pos)
            metric_name = metric_name.replace('neg', neg)
            writer.add_scalar(
                tag=metric_name,
                # The y-axis in TensorBoard
                scalar_value=value,
                # The x-axis value in TensorBoard
                global_step=self.totalTrainingSamples_count
            )

        # Create a new, empty figure object and makes it the current active figure.
        # All plotting commands that follow will draw on this active figure.
        fig = pyplot.figure()
        # Draw the line plot on the figure object fig in memory rather than on TensorBoard.
        pyplot.plot(fpr, tpr)

        # Plot the Receiver Operating Characteristic curve to see how well the model separates classes
        # The figure (the ROC curve) is sent to TensorBoard for logging and visualization.
        writer.add_figure('roc', fig, self.totalTrainingSamples_count)
        # Log the Area Under the ROC score to monitor improvements over time.
        writer.add_scalar('auc', auc, self.totalTrainingSamples_count)

        bins = np.linspace(0, 1)

        writer.add_histogram(
            'label_neg',
            metrics_t[METRICS_PRED_POS_NDX, negLabel_mask],
            self.totalTrainingSamples_count,
            bins=bins
        )
        writer.add_histogram(
            'label_pos',
            metrics_t[METRICS_PRED_POS_NDX, posLabel_mask],
            self.totalTrainingSamples_count,
            bins=bins
        )

        if not self.cli_args.malignant:
            score = metrics_dict['pr/f1_score']
        else:
            score = metrics_dict['auc']

        return score

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'data-unversioned',
            'part2',
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_e{:02d}_{}.state'.format(
                type_str,
                self.time_str,
                epoch_ndx,
                self.totalTrainingSamples_count,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'data-unversioned',
                'part2',
                'models',
                self.cli_args.tb_prefix,
                '{}_{}_e{:02d}_{}_{}.state'.format(
                    type_str,
                    self.time_str,
                    epoch_ndx,
                    self.totalTrainingSamples_count,
                    'best',
                )
            )
            shutil.copyfile(file_path, best_path)

            log.debug("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())


if __name__ == '__main__':
    ClassificationTrainingApp().main()
