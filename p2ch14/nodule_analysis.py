import argparse
import glob
import os
import sys

import numpy as np
import scipy.ndimage.measurements as measurements
import scipy.ndimage.morphology as morphology

import torch
import torch.nn as nn
import torch.optim

from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from p2ch13.dsets import LunaSliceSegDataset
from .dsets import LunaDataset, getCt, getCandidateInfoDict, getCandidateInfoList, CandidateInfoTuple
from p2ch13.model import UNetWrapper

import p2ch14.model

from util.logconf import logging
from util.util import xyz2irc, irc2xyz

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# Capture all levels of logs: DEBUG, INFO, WARNING, ERROR, CRITICAL
#log.setLevel(logging.DEBUG)
# Ignore DEBUG and INFO messages from the module.
#logging.getLogger("p2ch13.dsets").setLevel(logging.WARNING)
#logging.getLogger("p2ch14.dsets").setLevel(logging.WARNING)

def print_confusion(label, confusions, do_mal):
    """
    confusion[...] is a 3x4 confusion matrix where:

    Rows = Ground truth:
        - Row 0: Non-nodule (no annotation)
        - Row 1: Benign nodule
        - Row 2: Malignant nodule

    Columns = Detection result:
        - Col 0: Not detected
        - Col 1: Detected by segmentation only (filtered out)
        - Col 2: Detected and predicted benign
        - Col 3: Detected and predicted malignant
    """
    row_labels = ['Non-Nodules', 'Benign', 'Malignant']

    if do_mal:
        col_labels = ['', 'Complete Miss', 'Filtered Out', 'Pred. Benign', 'Pred. Malignant']
    else:
        col_labels = ['', 'Complete Miss', 'Filtered Out', 'Pred. Nodule']
        confusions[:, -2] += confusions[:, -1]
        confusions = confusions[:, :-1]
    cell_width = 16
    f = '{:>' + str(cell_width) + '}'
    print(label)
    print(' | '.join([f.format(s) for s in col_labels]))
    for i, (l, r) in enumerate(zip(row_labels, confusions)):
        r = [l] + list(r)
        if i == 0:
            r[1] = ''
        print(' | '.join([f.format(i) for i in r]))

def match_and_score(detections, truth, threshold=0.5):
    # Returns 3x4 confusion matrix for:
    # Rows: Truth: Non-Nodules, Benign, Malignant
    # Cols: Not Detected, Detected by Seg, Detected as Benign, Detected as Malignant
    # If one true nodule matches multiple detections, the "highest" detection is considered
    # If one detection matches several true nodule annotations, it counts for all of them
    true_nodules = [c for c in truth if c.isNodule_bool]
    truth_diams = np.array([c.diameter_mm for c in true_nodules])
    truth_xyz = np.array([c.center_xyz for c in true_nodules])

    # Each element of detections contains (prob_nodule, prob_mal, center_xyz, center_irc).
    detected_center_xyz = np.array([n[2] for n in detections])

    # detection classes will contain
    # 1 -> detected by seg but filtered by cls
    # 2 -> detected as benign nodule (or nodule if no malignancy model is used)
    # 3 -> detected as malignant nodule (if applicable)
    detected_classes = np.array([
        1 if detection[0] < threshold else
        2 if detection[1] < threshold else
        3
        for detection in detections
    ])

    confusion = np.zeros((3, 4), dtype=int)
    if len(detected_center_xyz) == 0:
        # No nodules were detected. -> Col 0: Complete Miss
        # -> If it is malignant (Row 2), it increments confusion[2, 0].
        # -> If it is benign (Row 1), it increments confusion[1, 0].
        for tn in true_nodules:
            confusion[2 if tn.isMal_bool else 1, 0] += 1
    elif len(truth_xyz) == 0:
        for dc in detected_classes:
            confusion[0, dc] += 1
    else:
        """
        truth_xyz shape: [N, 3]
        truth_xyz[:, None] shape: [N, 1, 3]
        detected_center_xyz shape: [M, 3]
        detected_center_xyz[None] shape: [1, M, 3]
        truth_xyz[:, None] - detected_center_xyz[None] shape: [N, M, 3]
        
        normalized_dists shape: [N, M]
        normalized_dists[i, j] = the normalized Euclidean distance between:
            - the i-th true nodule, and
            - the j-th detection.
        """
        normalized_dists = np.linalg.norm(
            truth_xyz[:, None] - detected_center_xyz[None],
            # The type of norm is Euclidean norm which represents Euclidean distance of a difference vector.
            ord=2,
            # Compute the Euclidean distance along the last dimension (i.e., across x, y, z components)
            axis=-1
        ) / truth_diams[:, None] # Scale-invariant

        matches = (normalized_dists < 0.7)
        unmatched_detections = np.ones(len(detections), dtype=np.bool)
        matched_true_nodules = np.zeros(len(true_nodules), dtype=int)
        
        """
        i_tn: array of matching ground truth indices
        i_detection: array of matching detection indices

        Each (i_tn, i_detection) pair means:
            True nodule i_tn matched with Detection i_detection.

        For ground truth nodule i_tn, record the most confident class among all matching detections
        — using the highest class label (1 < 2 < 3).
        If multiple detections are within the matching distance to the same ground truth nodule:
        We prefer keeping the highest class value.
        """
        for i_tn, i_detection in zip(*matches.nonzero()):
            # The max function ensures that if multiple detections match the same nodule, the one with the
            # highest class value (i.e., most severe) is retained
            matched_true_nodules[i_tn] = max(matched_true_nodules[i_tn], detected_classes[i_detection])
            unmatched_detections[i_detection] = False

        for ud, dc in zip(unmatched_detections, detected_classes):
            if ud:
                confusion[0, dc] += 1

        for tn, dc in zip(true_nodules, matched_true_nodules):
            confusion[2 if tn.isMal_bool else 1, dc] += 1

    return confusion

class NoduleAnalysisApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.info(sys.argv)
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=4,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=4,
            type=int,
        )

        parser.add_argument('--run-validation',
            help='Run over validation rather than a single CT.',
            action='store_true',
            default=False,
        )

        parser.add_argument('--seg-path',
            help="Path to the saved segmentation model",
            nargs='?',
            default='data/part2/models/seg_2020-01-26_19.45.12_w4d3c1-bal_1_nodupe-label_pos-d1_fn8-adam.best.state',
        )

        parser.add_argument('--cls-model-class',
            help="Class name of the model for the nodule classifier.",
            action='store',
            default='LunaModel',
        )
        parser.add_argument('--cls-path',
            help="Path to the saved classification model",
            nargs='?',
            default='data/part2/models/cls_2020-02-06_14.16.55_final-nodule-nonnodule.best.state',
        )

        # mThe malignancy model uses the same model class as the nodule classifier execept that
        # the last two layers are finetuned (i.e., the weights in the last two layers are different).
        parser.add_argument('--mal-model-class',
            help="Class name of the model for the malignancy nodule classifier.",
            action='store',
            default='LunaModel',
        )
        parser.add_argument('--mal-path',
            help="Path to the saved malignancy classification model",
            nargs='?',
            default=None,
        )

        parser.add_argument('--tb-prefix',
            default='tb_top',
            help="Data prefix to use for Tensorboard run.",
        )

        parser.add_argument('series_uid',
            nargs='?',
            default=None,
            help="Series UID to use.",
        )

        self.cli_args = parser.parse_args(sys_argv)
        # self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        if not (bool(self.cli_args.series_uid) ^ self.cli_args.run_validation):
            raise Exception("Exactly one of --series_uid or --run-validation must be specified.")


        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if not self.cli_args.seg_path:
            self.cli_args.seg_path = self.initModelPath('seg')

        if not self.cli_args.cls_path:
            self.cli_args.cls_path = self.initModelPath('cls')

        self.seg_model, self.cls_model, self.mal_model = self.initModels()

    def initModelPath(self, type_str):
        # TODO: Better naming
        local_path = os.path.join(
            'data-unversioned', # Should keep this
            'part2', # Not needed
            'models',
            'p2ch13', # rename this
            type_str + '_{}_{}.{}.state'.format('*', '*', 'best'),
        )

        file_list = glob.glob(local_path)
        if not file_list:
            pretrained_path = os.path.join(
                'data',
                'part2', # Not needed
                'models',
                type_str + '_{}_{}.{}.state'.format('*', '*', '*'),
            )
            file_list = glob.glob(pretrained_path)
        else:
            pretrained_path = None

        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            log.info([local_path, pretrained_path, file_list])
            raise

    def initModels(self):
        log.info(f"Using segmentation model from: {self.cli_args.seg_path}")
        seg_dict = torch.load(self.cli_args.seg_path)

        seg_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        """
        We load only the models' weights (model_state) because the models are set up for inference, not training.
        If you plan to fine-tune or resume training, you should also load the optimizer_state to preserve training
        dynamics like momentum and learning rate. Only loading model_state is sufficient when models are used in
        .eval() mode without further updates.
        """
        seg_model.load_state_dict(seg_dict['model_state'])
        seg_model.eval()

        log.info(f"Using nodule classification model from: {self.cli_args.cls_path}")
        cls_dict = torch.load(self.cli_args.cls_path)

        model_class = getattr(p2ch14.model, self.cli_args.cls_model_class)
        cls_model = model_class()
        cls_model.load_state_dict(cls_dict['model_state'])
        cls_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)

            seg_model.to(self.device)
            cls_model.to(self.device)

        if self.cli_args.mal_path:
            model_class = getattr(p2ch14.model, self.cli_args.mal_model_class)
            mal_model = model_class()
            log.info(f"Using nodule malignancy model from: {self.cli_args.mal_path}")
            malignancy_dict = torch.load(self.cli_args.mal_path)
            mal_model.load_state_dict(malignancy_dict['model_state'])
            mal_model.eval()
            if self.use_cuda:
                mal_model.to(self.device)
        else:
            mal_model = None
        return seg_model, cls_model, mal_model


    def initSegmentationDl(self, series_uid):
        seg_ds = LunaSliceSegDataset(
                contextSlices_count=3,
                series_uid=series_uid,
                fullCt_bool=True,
            )
        seg_dl = DataLoader(
            seg_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return seg_dl

    def initClassificationDl(self, candidateInfo_list):
        cls_ds = LunaDataset(
            val_stride=0,
            isValSet_bool=True,
            sortby_str='series_uid',
            candidateInfo_list=candidateInfo_list,
        )
        cls_dl = DataLoader(
            cls_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return cls_dl


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
        )

        positive_set = set(
            candidateInfo_tup.series_uid
            for candidateInfo_tup in getCandidateInfoList()
            if candidateInfo_tup.isNodule_bool
        )

        if self.cli_args.series_uid:
            series_set = set(self.cli_args.series_uid.split(','))
            val_list = sorted(series_set)
        else:
            val_set = set(
                candidateInfo_tup.series_uid
                for candidateInfo_tup in val_ds.candidateInfo_list
            )
            val_list = sorted(val_set)

        candidateInfo_dict = getCandidateInfoDict()

        """
        You can pass any object that supports both __iter__() and __len__().
        Valid types include: list, tuple, range, DataLoader (from PyTorch),
        any custom object that implements both __iter__ and __len__
        """
        series_iter = enumerateWithEstimate(
            val_list,
            "Series",
        )
        
        all_confusion = np.zeros((3, 4), dtype=int)

        """
        The lung nodule analysis pipeline contain three major stages:
        1. Segmentation — Find regions in the lung CT that might contain nodules.
        2. Classification — Use a classifier to verify whether those regions are actually nodules.
        3. Malignancy classification — If a region is a nodule, classify it as benign or malignant.
        """
        for _, series_uid in series_iter:
            ct = getCt(series_uid)
            
            # 1. Detect 3D regions in the 3D CT scan that likely contain nodules using U-Net model
            mask_a = self.segmentCt(ct, series_uid)
            # 2. Group the detected 3D regions into list of candidates
            candidateInfo_list = self.groupSegmentationOutput(series_uid, ct, mask_a)
            # 3. Classify the candidates in the list
            classifications_list = self.classifyCandidates(ct, candidateInfo_list)

            """
            If our threshold was 0.3 instead of 0.5, we would present a few more candidates
            that turn out not to be nodules, while reducing the risk of missing actual nodules.
            """
            if not self.cli_args.run_validation:
                print(f"found nodule candidates in {series_uid}:")
                for prob_nodule, prob_mal, center_xyz, center_irc in classifications_list:
                    # For all candidates found by the segmentation where the classifier assigned
                    # a nodule probability of 50% or more
                    if prob_nodule > 0.5:
                        s = f"nodule prob {prob_nodule:.3f}, "
                        if self.mal_model:
                            s += f"malignancy prob {prob_mal:.3f}, "
                        s += f"center xyz {center_xyz}"
                        print(s)

            if series_uid in candidateInfo_dict:
                one_confusion = match_and_score(
                    classifications_list, candidateInfo_dict[series_uid]
                )
                all_confusion += one_confusion
                """
                print_confusion(
                    series_uid, one_confusion, self.mal_model is not None
                )
                """

        print_confusion(
            "Total", all_confusion, self.mal_model is not None
        )


    def classifyCandidates(self, ct, candidateInfo_list):
        # Get a data loader to loop over the candidate list
        cls_dl = self.initClassificationDl(candidateInfo_list)
        classifications_list = []
        for batch_ndx, batch_tup in enumerate(cls_dl):
            input_t, _, _, _, center_list = batch_tup
            # Send the inputs to the device
            input_g = input_t.to(self.device)

            with torch.no_grad():
                # Run the inputs through the nodule vs. non-nodule network
                _, probability_nodule_g = self.cls_model(input_g)
                if self.mal_model is not None:
                    _, probability_mal_g = self.mal_model(input_g)
                else:
                    probability_mal_g = torch.zeros_like(probability_nodule_g)

            zip_iter = zip(
                center_list,
                # The second column shows the probability of being a nodule (class 1).
                probability_nodule_g[:,1].tolist(),
                # Convert PyTorch tensor to a plain Python list because because the
                # zip() function operates on standard Python iterables
                probability_mal_g[:,1].tolist()
            )

            # Do the bookkeeping by constructing a list of the results
            for center_irc, prob_nodule, prob_mal in zip_iter:
                center_xyz = irc2xyz(
                    coord_irc=center_irc,
                    direction_a=ct.direction_a,
                    origin_xyz=ct.origin_xyz,
                    vxSize_xyz=ct.vxSize_xyz,
                )
                cls_tup = (prob_nodule, prob_mal, center_xyz, center_irc)
                classifications_list.append(cls_tup)

        return classifications_list


    def segmentCt(self, ct, series_uid):
        """
        Applies a pretrained segmentation model to a CT scan to identify potential nodule regions.

        This function processes the input 3D CT slice by slice, using a U-Net segmentation model,
        and reconstructs a full 3D mask indicating likely nodule locations.

        Parameters
        ----------
        ct : Ct
            A CT scan object containing Hounsfield Unit data and metadata.

        series_uid : str
            Unique identifier for the scan, used for dataset access and logging.

        Returns
        -------
        mask_a : np.ndarray
            A 3D array (same shape as CT scan) where each voxel indicates the probability
            of being part of a lung nodule.

        Notes
        -----
        This function serves as the first step in the nodule detection pipeline, narrowing
        down candidate regions for later classification.
        """
        with torch.no_grad():
            # Get a NumPy array of the same shape as the CT scan filled with zeros.
            # It will hold floating-point probability predictions for each slice.
            output_a = np.zeros_like(ct.hu_a, dtype=np.float32)
            # Initialize a data loader that returns batches of CT slices.
            seg_dl = self.initSegmentationDl(series_uid)
            """
            central_slice_ndx is a 1D tensor of shape [batch_size], containing the
            individual slice indices for each sample in the batch. This is because
            it is returned by the DataLoader and DataLoader automatically batches values.

            Each entry central_slice_ndx[i] is the slice-number in the full 3D scan
            that was used as the central slice for the i-th sample in the batch.
            The sequence of central slices enables us to reconstruct the full 3D
            probability map (output_a) by placing each 2D prediction back into
            the correct z-position of the CT volume. Without these indices, we
            wouldn't know where each slice's prediction belongs in the overall scan.
            """
            for input_t, _, _, central_slice_ndx in seg_dl:

                # Transfer the batch of input slices to the GPU
                input_g = input_t.to(self.device)

                # Run the segmentation model (U-Net) on the batch.
                prediction_g = self.seg_model(input_g)

                # Reconstruct the 3D volume one slice at a time
                for i, slice_ndx in enumerate(central_slice_ndx):
                    # Move the prediction tensor from GPU to CPU because NumPy arrays
                    # can only be created from CPU memory.
                    output_a[slice_ndx] = prediction_g[i].cpu().numpy()

            # Thresholds the probability outputs to get a binary output, and then
            # applies binary erosion as cleanup
            mask_a = output_a > 0.5
            mask_a = morphology.binary_erosion(mask_a, iterations=1)

        return mask_a

    def groupSegmentationOutput(self, series_uid,  ct, clean_a):
        """
        Extracts and converts segmentation mask regions into structured candidate objects.

        These candidates are used in the subsequent classification step to assign 
        probabilities of being nodules or malignant growths.
        """

        # Label connected voxel regions in the binary segmentation mask. Each region is a potential nodule.
        # candidateLabel_a: an array of the same shape as clean_a, where:
        #   - Each connected component gets a unique integer label: 1, 2, 3, ...
        #   - Background remains labeled as 0.
        candidateLabel_a, candidate_count = measurements.label(clean_a)

        # Calculate the center of mass for each region in a 3D array
        centerIrc_list = measurements.center_of_mass(
            # + 1001 Makes all values strictly positive, since center-of-mass is computed with weighted positions.
            # Negative values could cause misleading results or division issues.
            ct.hu_a.clip(-1000, 1000) + 1001,
            # Tell the function which connected regions to analyze.
            labels=candidateLabel_a,
            # Tell the function: “Compute center of mass for label 1, 2, ..., up to candidate_count.”
            # np.arange(start, stop) generates numbers from start up to but not including stop.
            index=np.arange(1, candidate_count+1),
        )

        candidateInfo_list = []
        for i, center_irc in enumerate(centerIrc_list):
            # Convert the voxel coordinates to real patient coordinates
            center_xyz = irc2xyz(
                center_irc,
                ct.origin_xyz,
                ct.vxSize_xyz,
                ct.direction_a,
            )

            assert np.all(np.isfinite(center_irc)), repr(['irc', center_irc, i, candidate_count])
            assert np.all(np.isfinite(center_xyz)), repr(['xyz', center_xyz])

            # Build our candidate info tuple and appends it to the list of detections
            candidateInfo_tup = CandidateInfoTuple(
                isNodule_bool=False,
                hasAnnotation_bool=False,
                isMal_bool=False,
                diameter_mm=0.0,
                series_uid=series_uid,
                center_xyz=center_xyz
            )

            candidateInfo_list.append(candidateInfo_tup)

        return candidateInfo_list

    def logResults(self, mode_str, filtered_list, series2diagnosis_dict, positive_set):
        count_dict = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        for series_uid in filtered_list:
            probablity_float, center_irc = series2diagnosis_dict.get(series_uid, (0.0, None))
            if center_irc is not None:
                center_irc = tuple(int(x.item()) for x in center_irc)
            positive_bool = series_uid in positive_set
            prediction_bool = probablity_float > 0.5
            correct_bool = positive_bool == prediction_bool

            if positive_bool and prediction_bool:
                count_dict['tp'] += 1
            if not positive_bool and not prediction_bool:
                count_dict['tn'] += 1
            if not positive_bool and prediction_bool:
                count_dict['fp'] += 1
            if positive_bool and not prediction_bool:
                count_dict['fn'] += 1


            log.info("{} {} Label:{!r:5} Pred:{!r:5} Correct?:{!r:5} Value:{:.4f} {}".format(
                mode_str,
                series_uid,
                positive_bool,
                prediction_bool,
                correct_bool,
                probablity_float,
                center_irc,
            ))

        total_count = sum(count_dict.values())
        percent_dict = {k: v / (total_count or 1) * 100 for k, v in count_dict.items()}

        precision = percent_dict['p'] = count_dict['tp'] / ((count_dict['tp'] + count_dict['fp']) or 1)
        recall    = percent_dict['r'] = count_dict['tp'] / ((count_dict['tp'] + count_dict['fn']) or 1)
        percent_dict['f1'] = 2 * (precision * recall) / ((precision + recall) or 1)

        log.info(mode_str + " tp:{tp:.1f}%, tn:{tn:.1f}%, fp:{fp:.1f}%, fn:{fn:.1f}%".format(
            **percent_dict,
        ))
        log.info(mode_str + " precision:{p:.3f}, recall:{r:.3f}, F1:{f1:.3f}".format(
            **percent_dict,
        ))



if __name__ == '__main__':
    NoduleAnalysisApp().main()
