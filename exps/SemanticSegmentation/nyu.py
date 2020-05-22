import numpy as np
import torch
from datasets.BaseDataset import BaseDataset
import os
import cv2

class NYUv2(BaseDataset):
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None, s3client=None):
        super(NYUv2, self).__init__(setting, split_name, preprocess, file_length)
        self._split_name = split_name
        self._i_path = setting['i_root']
        self._g_path = setting['g_root']
        self._h_path = setting['h_root']
        self._t_source = setting['t_source']
        self._e_source = setting['e_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __getitem__(self, index):

        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]
        i_path = os.path.join(self._i_path ,names[0])
        g_path = os.path.join(self._g_path ,names[1])
        item_name = names[1].split("/")[-1].split(".")[0]
        h_path = os.path.join(self._h_path, item_name + '.jpg')
        iii, ggg, hhh = self._fetch_data(i_path, g_path, h_path)
        iii = iii[:, :, ::-1]
        hhh = hhh[:, :, ::-1]
        if self.preprocess is not None:
            iii, ggg, extra_dict = self.preprocess(iii, ggg, hhh)
        if self._split_name is 'train':
            iii = torch.from_numpy(np.ascontiguousarray(iii)).float()
            ggg = torch.from_numpy(np.ascontiguousarray(ggg)).long()
            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v)).float()

        output_dict = dict(ddd=iii, lll=ggg, fn=str(item_name),
                           n=len(self._file_names))
        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)
        return output_dict

    def _get_file_names(self, split_name, train_extra=False):
        assert split_name in ['train', 'val']
        source = self._t_source
        if split_name == "val":
            source = self._e_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            img_name, gt_name = self._process_item_names(item)
            file_names.append([img_name, gt_name])

        if train_extra:
            file_names2 = []
            source2 = self._train_source.replace('train', 'train_extra')
            with open(source2) as f:
                files2 = f.readlines()

            for item in files2:
                img_name, gt_name = self._process_item_names(item)
                file_names2.append([img_name, gt_name])

            return file_names, file_names2

        return file_names

    def _fetch_data(self, i_path, g_path, h_path, dtype=None):
        iii = self._open_image(i_path)
        ggg = self._open_image(g_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)
        hhh =  self._open_image(h_path)
        return iii, ggg, hhh

    @classmethod
    def get_class_names(*args):

        return ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
                'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
                'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
                'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']

    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        return label, new_name
