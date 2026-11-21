import os
import pickle

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


RELATIONS = [
    "xIntent",
    "xAttr",
    "xNeed",
    "xWant",
    "xEffect",
    "xReact",
    "oWant",
    "oEffect",
    "oReact",
]


def _resolve_existing_path(path, fallback_candidates=None):
    candidates = []
    if path:
        candidates.append(path)
    if fallback_candidates:
        candidates.extend(fallback_candidates)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "找不到数据文件，已尝试路径: {}".format(
            ", ".join([str(item) for item in candidates if item])
        )
    )


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class IEMOCAPDataset(Dataset):
    def __init__(self, train=True, data_path=None, knowledge_path=None, use_knowledge=False):
        root_dir = _project_root()
        data_path = _resolve_existing_path(
            data_path,
            [
                os.path.join(root_dir, "iemocap_multimodal_features.pkl"),
                os.path.join(root_dir, "data", "iemocap_multimodal_features.pkl"),
                os.path.join(os.path.dirname(__file__), "data", "iemocap_multimodal_features.pkl"),
            ],
        )
        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.roberta2,
            self.roberta3,
            self.roberta4,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
        ) = pickle.load(open(data_path, "rb"), encoding="latin1")

        self.use_knowledge = use_knowledge
        self.knowledge = {}
        if self.use_knowledge:
            knowledge_path = _resolve_existing_path(
                knowledge_path,
                [
                    os.path.join(root_dir, "iemocap_features_comet.pkl"),
                    os.path.join(root_dir, "data", "iemocap_features_comet.pkl"),
                    os.path.join(os.path.dirname(__file__), "data", "iemocap_features_comet.pkl"),
                ],
            )
            relation_values = pickle.load(open(knowledge_path, "rb"), encoding="latin1")
            if len(relation_values) != len(RELATIONS):
                raise ValueError(
                    "iemocap_features_comet.pkl 结构错误，期望 {} 个关系字典，实际为 {}。".format(
                        len(RELATIONS), len(relation_values)
                    )
                )
            self.knowledge = dict(zip(RELATIONS, relation_values))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def _get_knowledge_tensor(self, vid):
        relation_tensors = []
        for relation in RELATIONS:
            if vid not in self.knowledge[relation]:
                raise KeyError("知识特征缺少 vid: {} / relation: {}".format(vid, relation))
            relation_tensors.append(
                torch.from_numpy(np.asarray(self.knowledge[relation][vid], dtype=np.float32))
            )
        return torch.stack(relation_tensors, dim=1)

    def __getitem__(self, index):
        vid = self.keys[index]
        sample = [
            torch.from_numpy(np.asarray(self.videoText[vid], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.videoVisual[vid], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.videoAudio[vid], dtype=np.float32)),
        ]
        if self.use_knowledge:
            sample.append(self._get_knowledge_tensor(vid))
        sample.extend(
            [
                torch.FloatTensor(
                    [[1, 0] if x == "M" else [0, 1] for x in self.videoSpeakers[vid]]
                ),
                torch.FloatTensor([1] * len(self.videoLabels[vid])),
                torch.LongTensor(self.videoLabels[vid]),
                vid,
            ]
        )
        return tuple(sample)

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        textf = pad_sequence([item[0] for item in data])
        visuf = pad_sequence([item[1] for item in data])
        acouf = pad_sequence([item[2] for item in data])

        offset = 3
        knowledge = None
        if self.use_knowledge:
            knowledge = pad_sequence([item[3] for item in data])
            offset = 4

        qmask = pad_sequence([item[offset] for item in data])
        umask = pad_sequence([item[offset + 1] for item in data], batch_first=True)
        label = pad_sequence(
            [item[offset + 2] for item in data], batch_first=True, padding_value=-1
        )
        vid = [item[offset + 3] for item in data]

        if self.use_knowledge:
            return textf, visuf, acouf, knowledge, qmask, umask, label, vid
        return textf, visuf, acouf, qmask, umask, label, vid


class MELDDataset(Dataset):
    def __init__(self, path, train=True):
        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.roberta2,
            self.roberta3,
            self.roberta4,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
            _,
        ) = pickle.load(open(path, "rb"))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(self.videoText[vid]),
            torch.FloatTensor(self.videoVisual[vid]),
            torch.FloatTensor(self.videoAudio[vid]),
            torch.FloatTensor(self.videoSpeakers[vid]),
            torch.FloatTensor([1] * len(self.videoLabels[vid])),
            torch.LongTensor(self.videoLabels[vid]),
            vid,
        )

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label += self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        textf = pad_sequence([item[0] for item in data])
        visuf = pad_sequence([item[1] for item in data])
        acouf = pad_sequence([item[2] for item in data])
        qmask = pad_sequence([item[3] for item in data])
        umask = pad_sequence([item[4] for item in data], batch_first=True)
        label = pad_sequence([item[5] for item in data], batch_first=True, padding_value=-1)
        vid = [item[6] for item in data]
        return textf, visuf, acouf, qmask, umask, label, vid

