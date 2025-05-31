import copy
import csv
import functools
import glob
import math
import os
import random

from collections import namedtuple

import SimpleITK as sitk
import numpy as np
import scipy.ndimage.morphology as morph

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
#log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch13_raw')

MaskTuple = namedtuple('MaskTuple', 'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask')

CandidateInfoTuple = namedtuple('CandidateInfoTuple', 'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # Get a list of all file paths that match the pattern.
    mhd_list = glob.glob('F:/Organized_LUNA16_Train_Data/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    candidateInfo_list = []
    
    """
    The file `annotations_with_malignancy.csv` provides expert-annotated information about known lung nodules found in CT scans.
    This file serves as the ground truth for training a model in supervised learning. Each line represents a single annotated lung nodule.
    """
    with open('data/part2/luna/annotations_with_malignancy.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            # Note that Python slices are start-inclusive, end-exclusive. So, there are 3 elements.
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            isMal_bool = float(row[5]) >= 1.0

            candidateInfo_list.append(
                CandidateInfoTuple(
                    isNodule_bool=True,
                    hasAnnotation_bool=True,
                    isMal_bool=isMal_bool,
                    diameter_mm=annotationDiameter_mm,
                    series_uid=series_uid,
                    center_xyz=annotationCenter_xyz,
                )
            )

    # Entries in candidates.csv are laid out as seriesuid(0), coordX(1), coordY(2), coordZ(3), class(4)
    with open('data/part2/luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            # class == 1 -> True nodule (matches an entry in annotations.csv)
            # class == 0 -> not confirmed to be a nodule (did not match any known annotation)
            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            """
            We treat entries with class being 0 as negative samples (i.e., non-nodule).
            This gives your training dataset a mix of positive and negative samples:
                Positives: From annotations.csv, marked as isNodule_bool = True
                Negatives: From candidates.csv, with class = 0, marked as isNodule_bool = False
            """
            if not isNodule_bool:
                candidateInfo_list.append(
                    CandidateInfoTuple(
                        isNodule_bool=False,
                        hasAnnotation_bool=False,
                        isMal_bool=False,
                        diameter_mm=0.0,
                        series_uid=series_uid,
                        center_xyz=candidateCenter_xyz,
                    )
                )
    """
    By sorting in descending order, true nodules come first, which helps when only a portion of the list is processed.
    Sorting provides a deterministic order for use across different runs, systems, or stages of the pipeline.
    Some modules like PrepcacheLunaDataset or findPositiveSamples() rely on positive samples being near the beginning of the list to be efficient.
    """
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

"""
@functools.lru_cache(1, typed=True) caches the most recent function result (up to 1 entry) to avoid recalculating it if the same arguments are
used again. The typed=True ensures that arguments of different types (e.g., 1 vs 1.0) are treated as distinct. It improves performance when
repeatedly accessing the same CT scan data, which is computationally expensive to load.
"""
@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        # Group each candidate by series_uid in a dictionary. If this CT scan ID hasn't appeared before, start a new list for it.
        # Then, add the current candidate to that list.
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid, []).append(candidateInfo_tup)

    return candidateInfo_dict

class Ct:
    def __init__(self, series_uid):
        # [0] is used to extract the first (and expected only) match from the list
        # A .mhd (MetaImage Header) file is a metadata file that:
        #   Describes a 3D medical image volume.
        #   Points to a corresponding .raw file (binary voxel data) that contains the actual data.
        mhd_path = glob.glob(
            'F:/Organized_LUNA16_Train_Data/subset*/{}.mhd'.format(series_uid)
        )[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        # Get the entire 3D volume of the CT scan as a NumPy array.
        # It has shape of (num_slices, height, width), which is a stack of 2D slices.
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        """
        CT scans are measured in Hounsfield Units (HU).
        In this scale, air (0 g/cc) corresponds to -1000 HU, and water (1 g/cc)
        corresponds to 0 HU.
        The lower bound removes artificially low values often used to mark regions
        outside the field of view (FOV).
        The upper bound eliminates abnormally high-intensity values and limits the
        maximum intensity of dense structures such as bone.
        """
        ct_a.clip(-1000, 1000, ct_a)
        self.hu_a = ct_a

        self.series_uid = series_uid

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        candidateInfo_list = getCandidateInfoDict()[self.series_uid]

        # This list is used to build the annotation mask. 
        self.positiveInfo_list = [
            candidate_tup
            for candidate_tup in candidateInfo_list
            if candidate_tup.isNodule_bool # Filter the candidates into a list containing only nodules
        ]
        self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list)

        """
        .sum(axis=(1,2)) performs a sum over the height and width dimensions for each slice. For each slice (depth index),
        it adds up the True values on that 2D plane. It gives a 1D array of shape (depth,), where each element tells how
        many positive pixels are present on that slice.
        .nonzero() identifies the indices in that 1D array where the value is non-zero. It returns a tuple of arrays.
        [0] extracts the actual NumPy array of indices from the tuple.
        .tolist() converts the NumPy array to a Python list.
        self.positive_indexes is a list of slice indices where nodules were found. 
        """
        self.positive_indexes = (self.positive_mask.sum(axis=(1,2)).nonzero()[0].tolist())

    """
    The purpose of this method is to create a 3D binary mask indicating the presence of positive
    annotations (nodules) in the CT volume. Each "positive" region (where a nodule exists) is
    estimated by:
        Converting the nodule center from real-world coordinates to voxel (IRC) coordinates.
        Estimating a bounding box around that center where HU values are above a threshold.
        Filling the corresponding region in the bounding box with True.
    """
    def buildAnnotationMask(self, positiveInfo_list, threshold_hu = -700):
        """
        self.hu_a is a 3D NumPy array representing a CT scan volume in Hounsfield Units (HU).
        Create a NumPy array of the same shape and type as self.hu_a but filled with zeros.
        dtype=np.bool overrides the data type to boolean which means all elements are False.
        """
        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool)

        # Loop over the nodules.
        for candidateInfo_tup in positiveInfo_list:

            center_irc = xyz2irc(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )

            # Get the center voxel indices
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            # Radius defines how far to expand in each direction from the center.
            index_radius = 2
            try:
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            """
            Set a 3D cuboid in the array to True, marking that space as containing part of a candidate nodule.
            The +1 ensures that the end index is inclusive, since Python slicing is end-exclusive.
            This cuboid serves as an initial estimate of the annotated nodule volume.
            """
            boundingBox_a[
                 ci - index_radius: ci + index_radius + 1,
                 cr - row_radius: cr + row_radius + 1,
                 cc - col_radius: cc + col_radius + 1] = True
        """
        (self.hu_a > threshold_hu) is a boolean array that acts as a density filter to isolate solid structures
        (e.g. tumors) within the CT scan volume.
        mask_a is a refined binary mask that represents the 3D 'real' parts of a nodule candidate within the 3D bounding box.
        """
        mask_a = boundingBox_a & (self.hu_a > threshold_hu)

        return mask_a

    def getRawCandidate(self, center_xyz, width_irc):
        """
        Extract a 3D subvolume (a cube-like patch) centered at the given physical location from:
            - the raw CT scan (self.hu_a)
            - the binary segmentation mask of nodules (self.positive_mask)

        center_xyz: real-world coordinates of the target center (in mm).
        width_irc: 3D patch size in number of voxels (index, row, col).
        """
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.debug("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.debug("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            # Append a slice object (equivalent to [start_ndx:end_ndx]) to the list which can be used later to extract a 3D subvolume
            # from a CT scan array using this set/list of slices.
            slice_list.append(slice(start_ndx, end_ndx))

        # For NumPy arrays, array[(slice1, slice2, slice3)] is equivalent to array[slice1, slice2, slice3]. It allows you to extract a 3D subregion using slices for each axis.
        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

"""
The decorator caches the function's return value to disk or memory using a custom caching system,
so repeated calls with the same inputs avoid re-computation.
The typed=True ensures argument types are considered when generating the cache key (e.g., 7 vs 7.0
are treated differently). This is used to speed up expensive CT data extraction operations by reusing
precomputed results.
"""
@raw_cache.memoize(typed=True)
def getCtPatch(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    ct_chunk.clip(min=-1000, max=1000, out=ct_chunk) # in-place clipping
    return ct_chunk, pos_chunk, center_irc

@raw_cache.memoize(typed=True)
def getCtNumSlices(series_uid):
    """
    Return the number of slices in the CT volume for the given series UID.
    """
    ct = Ct(series_uid)
    return int(ct.hu_a.shape[0])

@raw_cache.memoize(typed=True)
def getCtPositiveSliceIndices(series_uid):
    """
    Return a list of slice indices that contain nodules in the CT volume
    for the given series UID.
    """
    ct = Ct(series_uid)
    return ct.positive_indexes

# This is the data set that operates on full 2D slices, suitable for model inference or validation.
class LunaSliceSegDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 contextSlices_count=3,
                 fullCt_bool=False,
            ):
        self.contextSlices_count = contextSlices_count
        
        # In full scan mode (fullCt_bool=True), all slices of the CT scan are included.
        self.fullCt_bool = fullCt_bool

        # Check if the user provided a specific series uid which allows using the dataset for just one CT scan.
        if series_uid:
            # Store the single series uid inside a list.
            self.series_list = [series_uid]
        else:
            # Store the series uids of all available CT scan series.
            self.series_list = sorted(getCandidateInfoDict().keys())

        if isValSet_bool:
            assert val_stride > 0, val_stride
            # Keep only every val_stride-th element, starting with 0.
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            # When training, we delete every val_stride-th element.
            del self.series_list[::val_stride]
            assert self.series_list

        self.slice_index_list = []
        for series_uid in self.series_list:
            if self.fullCt_bool:
                slice_count = getCtNumSlices(series_uid)
                self.slice_index_list += [(series_uid, slice_ndx)
                                            for slice_ndx in range(slice_count)]
            else:
                positive_indexes = getCtPositiveSliceIndices(series_uid)
                self.slice_index_list += [(series_uid, slice_ndx)
                                            for slice_ndx in positive_indexes]

        self.candidateInfo_list = getCandidateInfoList()

        """
        We convert the list into a set to:
            1. automatically remove duplicates
            2. speedup membership checking in later steps
        """
        series_set = set(self.series_list)

        # Check membership in a set takes O(1).
        # Filter out candidates with series UID that are absent in the series set.
        self.candidateInfo_list = [it for it in self.candidateInfo_list
                                   if it.series_uid in series_set]

        # Get a list of actual nodules
        self.pos_list = [it for it in self.candidateInfo_list
                         if it.isNodule_bool]

        log.debug("{!r}: {} {} series, {} slices, {} nodules".format(
                self,
                len(self.series_list),
                {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
                len(self.slice_index_list),
                len(self.pos_list)
            )
        )

    def __len__(self):
        return len(self.slice_index_list)

    def __getitem__(self, ndx):
        # By applying ndx % len(...), any out-of-range index is "wrapped around" to a valid index,
        # enabling cyclic access and safe indexing.
        series_uid, slice_ndx = self.slice_index_list[ndx % len(self.slice_index_list)]
        return self.getitem_fullSlice(series_uid, slice_ndx)

    # Get the entire slice of data for the given series uid and slice index
    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct = getCt(series_uid)
        """
        self.contextSlices_count * 2 + 1 calculates the number of slices that will be
        included in the tensor ct_t.
        contextSlices_count determines how many context slices (slices before or after
        the current slice) are considered.
        contextSlices_count * 2 is the total number of context slices before and after
        the current slice. +1 to include the current slice itself.
        """
        ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))

        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1

        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            # Ensure the current slice index is in the valid range for the very first and last slice.
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            # Convert the NumPy array to a PyTorch tensor with float32 format and
            # then assigns it to the i-th slice in the CT tensor
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        # In the Hounsfield Unit (HU) scale, air (approximately 0 g/cc) corresponds to -1000 HU,
        # while water (approximately 1 g/cc) corresponds to 0 HU.
        # Applying the lower bound removes extreme negative values that typically represent regions
        # outside the field of view (FOV).
        # Applying the upper bound suppresses abnormally high intensities, limiting bone density
        # and eliminating artifacts.
        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

        return ct_t, pos_t, ct.series_uid, slice_ndx

# This is the training data set, extracting cropped patches around known nodule locations.
class LunaPatchSegDataset(LunaSliceSegDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2

    def __len__(self):
        return 300000

    def shuffleSamples(self):
        random.shuffle(self.candidateInfo_list)
        random.shuffle(self.pos_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)]
        return self.getitem_trainingCrop(candidateInfo_tup)

    def getitem_trainingCrop(self, candidateInfo_tup):
        ct_a, pos_a, center_irc = getCtPatch(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            (7, 96, 96), # Size of the chunk with shape (depth, rows, cols) to extract 
        )
        
        # Only slice 3 is the candidate/target slice. For the rest, they're just context slices.
        pos_a = pos_a[3:4]

        # Randomly choose a row and column offset between 0 and 31.
        # They will be used to crop a 64×64 patch starting from them.
        row_offset = random.randrange(0,32)
        col_offset = random.randrange(0,32)
        
        ct_t = torch.from_numpy(
            ct_a[:, row_offset:row_offset+64, col_offset:col_offset+64]
        ).to(torch.float32)
        
        pos_t = torch.from_numpy(
            pos_a[:, row_offset:row_offset+64, col_offset:col_offset+64]
        ).to(torch.long)

        slice_ndx = center_irc.index

        return ct_t, pos_t, candidateInfo_tup.series_uid, slice_ndx

"""
The main purpose of PrepcacheLunaDataset is to:

  - Trigger loading and caching of expensive computations such as reading CT scans from disk and
  processing them into usable forms (e.g., calling getCtPatch).

  - Precompute intermediate results and save them in memory or disk cache. This caching
  significantly accelerates subsequent dataset loading during training and inference.
"""
class PrepcacheLunaDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.candidateInfo_list = getCandidateInfoList()
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        self.seen_set = set()
        self.candidateInfo_list.sort(key=lambda x: x.series_uid)

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        # candidate_t, pos_t, series_uid, center_t = super().__getitem__(ndx)

        candidateInfo_tup = self.candidateInfo_list[ndx]
        getCtPatch(candidateInfo_tup.series_uid, candidateInfo_tup.center_xyz, (7, 96, 96))

        series_uid = candidateInfo_tup.series_uid
        if series_uid not in self.seen_set:
            self.seen_set.add(series_uid)
            getCtNumSlices(series_uid)
            getCtPositiveSliceIndices(series_uid)

        """
        The Intention Behind Returning (0, 1):
        Since the dataset object is passed to a PyTorch DataLoader for batch processing, the DataLoader expects a return tuple from __getitem__.
        Returning (0, 1) fulfills this minimal requirement without any overhead or complex processing.
        """

        return 0, 1


