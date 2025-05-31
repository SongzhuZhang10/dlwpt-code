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

raw_cache = getCache('part2ch14_raw')

MaskTuple = namedtuple('MaskTuple', 'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask')

CandidateInfoTuple = namedtuple('CandidateInfoTuple',
    'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
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
            # Indicate if the nodule is malignant or not
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
                Negatives: From candidates.csv, with class = 0 marked as isNodule_bool = False
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
    By sorting in descending order, true nodules come first. Sorting provides a deterministic
    order for use across different runs, systems, or stages of the pipeline.
    Some modules like PrepcacheLunaDataset or findPositiveSamples() rely on positive samples
    being near the beginning of the list to be efficient.
    """
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

"""
The decorator caches the most recent function result to avoid recalculating it if the same
arguments are used again. The typed=True ensures that arguments of different types are
treated as distinct. It improves performance when repeatedly accessing the same CT scan data.
"""
@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        # Group each candidate by series_uid in a dictionary. If this CT scan ID hasn't
        # appeared before, start a new list for it. Then, add the current candidate to that list.
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

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)


    """
    Extract a 3D subvolume (a cube-like patch) centered at the given physical location from:
        - the raw CT scan (self.hu_a)
        - the binary segmentation mask of nodules (self.positive_mask)

    center_xyz: real-world coordinates of the target center (in mm).
    width_irc: 3D patch size in number of voxels (index, row, col).
    """
    def extractSubVolume(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert 0 <= center_val < self.hu_a.shape[axis], (
                f"Invalid center index along axis {axis}: {center_val} not in [0, {self.hu_a.shape[axis]}) | "
                f"Series UID: {self.series_uid}, Center (xyz): {center_xyz}, Origin (xyz): {self.origin_xyz}, "
                f"Voxel Size (xyz): {self.vxSize_xyz}, Center (irc): {center_irc}"
            )

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            # Append a slice object (equivalent to [start_ndx:end_ndx]) to the list which can be used later to extract a 3D subvolume
            # from a CT scan array using this list of slices.
            slice_list.append(slice(start_ndx, end_ndx))

        # For NumPy arrays, array[(slice1, slice2, slice3)] is equivalent to array[slice1, slice2, slice3]. It allows you to extract
        # a 3D subregion using slices for each axis.
        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

"""
The decorator caches the function's return value to disk or memory using a custom caching system,
so repeated calls with the same inputs avoid re-computation.
The typed=True ensures argument types are considered when generating the cache key (e.g., 7 vs 7.0
are treated differently). This is used to speed up expensive CT data extraction operations by reusing
precomputed results.
TODO: Merge getCtPatch and extractSubVolume into a single function!
"""
@raw_cache.memoize(typed=True)
def getCtPatch(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.extractSubVolume(center_xyz, width_irc)
    return ct_chunk, center_irc



def getCtAugmentedCandidate(
        augmentation_dict,
        series_uid,
        center_xyz,
        width_irc,
        use_cache=True
    ):

    if use_cache:
        ct_chunk, center_irc = getCtPatch(series_uid, center_xyz, width_irc)
    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.extractSubVolume(center_xyz, width_irc)

    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_t = torch.eye(4)

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i,i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_t[i,3] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_t[i,i] *= 1.0 + scale_float * random_float


    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])

        transform_t @= rotation_t

    affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0).to(torch.float32),
            ct_t.size(),
            align_corners=False,
        )

    augmented_chunk = F.grid_sample(
            ct_t,
            affine_t,
            padding_mode='border',
            align_corners=False,
        ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 sortby_str='random',
                 ratio_int=0,
                 augmentation_dict=None,
                 candidateInfo_list=None
        ):

        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        phase = "validation" if isValSet_bool else "training"

        if not self.augmentation_dict:
            log.debug("No data augmentation will be applied for %s.", phase)
        else:
            log.debug("Configuration of %s data augmentation:", phase)
            for k, v in sorted(self.augmentation_dict.items()):
                log.info("  %-6s: %s", k, v)
            """
            required_keys = {"flip", "offset", "scale", "rotate", "noise"}
            missing_keys  = required_keys - self.augmentation_dict.keys()

            if missing_keys:
                raise ValueError(
                    "augmentation_dict must contain: "
                    f"{', '.join(sorted(required_keys))}. "
                    f"Missing: {', '.join(sorted(missing_keys))}"
                )
            """

        if candidateInfo_list:
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False
        else:
            self.candidateInfo_list = copy.copy(getCandidateInfoList())
            self.use_cache = True

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(set(candidateInfo_tup.series_uid for candidateInfo_tup in self.candidateInfo_list))

        if val_stride == 0 and candidateInfo_list:
            log.debug("val_stride is 0 because custom candidate info list is provided.")
        elif val_stride == 0 and not candidateInfo_list:
            log.debug("val_stride is 0 while custom candidate info list is empty.")

        if isValSet_bool:
            if val_stride > 0:
                self.series_list = self.series_list[::val_stride]
                assert self.series_list, "series_list became empty after slicing for validation."
        else:
            if val_stride > 0:
                del self.series_list[::val_stride]
                assert self.series_list, "series_list became empty after deleting validation samples."

        series_set = set(self.series_list)
        self.candidateInfo_list = [x for x in self.candidateInfo_list if x.series_uid in series_set]

        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            # key=... tells .sort() how to compare elements.
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.neg_list = [it for it in self.candidateInfo_list if not it.isNodule_bool]
        self.pos_list = [it for it in self.candidateInfo_list if it.isNodule_bool]
        self.ben_list = [it for it in self.pos_list if not it.isMal_bool]
        self.mal_list = [it for it in self.pos_list if it.isMal_bool]

        log.debug("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
            len(self.neg_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.candidateInfo_list)
            random.shuffle(self.neg_list)
            random.shuffle(self.pos_list)
            random.shuffle(self.ben_list)
            random.shuffle(self.mal_list)

    """
    If ratio_int is set, the dataset length is faked to 50,000.
    This forces the training loop to sample many more batches, implying oversampling.
    The dataset never duplicates arrays in memory. It pretends to be longer.
    """
    def __len__(self):
        if self.ratio_int:
            return 50000
        else:
            return len(self.candidateInfo_list)

    """
    For consective indexes, it returns one positive sample followed by ratio_int negatives,
    then it repeats this pattern forever.
    If ratio_int = k, the long-run class ratio becomes 1 : k.
    The dataset is divided into cycles, each cycle having a length of (ratio_int + 1) samples.
    Every cycle contains exactly one positive sample at its beginning.
    """
    def __getitem__(self, ndx):
        if self.ratio_int:
            """
            Calculates the index of the positive sample to use.
            pos_ndx is the index of the positive sample we are currently at.
            Example (ratio_int = 1):
                Cycle length = ratio_int + 1 = 1 + 1 = 2 (1 positive, 1 negative)
                For indices 0 and 1 -> pos_ndx = 0
                For indices 2 and 3 -> pos_ndx = 1, and so on.
            """
            pos_ndx = ndx // (self.ratio_int + 1)

            """
            Check the position within a cycle:
                ndx % (ratio_int + 1) == 0: It's the first position → positive sample.
                ndx % (ratio_int + 1) != 0: Any other position → negative sample.
            """
            if ndx % (self.ratio_int + 1):
                # This ensures a sequential index into neg_list: 0, 1, 2, 3, ...
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.neg_list)
                candidateInfo_tup = self.neg_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list)
                candidateInfo_tup = self.pos_list[pos_ndx]
        else:
            candidateInfo_tup = self.candidateInfo_list[ndx]

        return self.sampleFromCandidateInfo_tup(
            candidateInfo_tup, candidateInfo_tup.isNodule_bool
        )

    def sampleFromCandidateInfo_tup(self, candidateInfo_tup, label_bool):

        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.augmentation_dict,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            candidate_a, center_irc = getCtPatch(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = getCt(candidateInfo_tup.series_uid)
            candidate_a, center_irc = ct.extractSubVolume(
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
                not candidateInfo_tup.isNodule_bool,
                candidateInfo_tup.isNodule_bool
            ],
            dtype=torch.long,
        )
        
        label_t = torch.tensor([False, False], dtype=torch.long)

        # For non-nodules, index_t = 0.
        # For nodules, index_t = 1
        if not label_bool:
            label_t[0] = True
            index_t = 0
        else:
            label_t[1] = True
            index_t = 1

        # candidate_t needs be of shape [1, 32, 48, 48] for the model to work.
        return candidate_t, label_t, index_t, candidateInfo_tup.series_uid, torch.tensor(center_irc)


class MalignantLunaDataset(LunaDataset):
    def __len__(self):
        if self.ratio_int:
            return 100000
        else:
            return len(self.ben_list + self.mal_list)

    """
    For consecutive indexes, the items are returned in a repeating pattern:
        [benign, malignant, negative, malignant] — every 4 indices.
    mal_list: 50% of samples
    ben_list: 25%
    neg_list: 25%
    
    Without ratio_int (natural distribution):
        - ben_list first, then mal_list in sequence
        - No negatives sampled
    """
    def __getitem__(self, ndx):
        if self.ratio_int:
            # Odd indices -> mal_list
            if ndx % 2 != 0:
                candidateInfo_tup = self.mal_list[(ndx // 2) % len(self.mal_list)]
            # Even and divisible by 4 -> ben_list
            elif ndx % 4 == 0:
                candidateInfo_tup = self.ben_list[(ndx // 4) % len(self.ben_list)]
            # Even but not divisible by 4 
            else:
                candidateInfo_tup = self.neg_list[(ndx // 4) % len(self.neg_list)]
        else: # unbalanced mode
            if ndx >= len(self.ben_list):
                candidateInfo_tup = self.mal_list[ndx - len(self.ben_list)]
            else:
                candidateInfo_tup = self.ben_list[ndx]

        return self.sampleFromCandidateInfo_tup(
            candidateInfo_tup, candidateInfo_tup.isMal_bool
        )
