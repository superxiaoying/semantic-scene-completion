# Lab

### Prerequisites

- PyTorch 1.0.0 (Exactly)
- CUDA 9.0
- Ninja
- tqdm
- Apex

```shell
$ cd furnace/apex
$ python setup.py install --cpp_ext --cuda_ext
```

- Some tips
  - You should modify the `C.volna ` in the config.py so that this dir is the root dir of `DATA` dir. Path of `DATA` dir: `/mnt/lustre/chenxiaokang/` (SH 36)
  - Modify the `C.repo_name` in the config.py so that this string is the same with the name of the root dir (such as 'cxklab').

### Semantic Segmentation

```shell
$ cd exps/SemanticSegmentation
$ ./st-run.sh
$ vim log/val_last.log # this will show the results after training
```

You will expect around 51.2% mIoU on the NYU Depth V2 dataset for single-scale testing. For multi-scale testing, you only need to modify the `C.eval_scale_array` in the config.py (such as modify it to be the same with the `C.train_scale_array`). This will boost the mIoU by ~1%.

### Semantic Scene Completion

```shell
$ cd exps/SemanticSceneCompletion
$ ./st-run.sh
$ vim log/val_last.log # this will show the results after training
```

You will expect 41.1% SSC mIoU and 71% SC IoU on the NYU dataset.