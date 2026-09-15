# ATCSA-Net

Training, evaluation, and latency-profiling scripts for **ATCSA-Net**: a confusion-guided hierarchical framework for fine-grained DAS vibration recognition.

The private DAS recordings used in the paper cannot be released. Public UCR datasets used in Section 4.6.2 can be downloaded by the UCR script.

## Cascade topology

Five classes: Manual Digging (0), Mechanical Digging (1), Harvester (2), Rotary Tiller (3), Directional Drill (4).

1. Coarse 3-way router over `{0,4} / {2,3} / {1}` (CNN, no CCSTA, cross-entropy).
2. Fine expert **S2**: class 0 vs 4 (CNN + CCSTA, BCE).
3. Fine expert **S1**: class 2 vs 3 (CNN + CCSTA, BCE).
4. Class 1 is emitted at the coarse stage.

The manuscript reports cascade accuracy under **in-group** evaluation: the true superclass selects S1 or S2. Deployed inference uses the predicted coarse branch. `predict_on_testset.py` prints both protocols.

## Environment

```bash
pip install -r requirements.txt
```

GPU is recommended. Latency tables in the paper used batch size 1 with CUDA synchronization.

## Data (proprietary DAS)

Place npy files as:

```
分类数据/
├── 人工挖掘/
├── 机械挖掘/
├── 收割机/
├── 旋耕机/
└── 定向钻/
```

Each file is a `(200, 12397)` waterfall. Filenames contain `point_XXXXXX`. SCP crops a 128-wide spatial block around that index and resizes time to 128, yielding a `128×128` patch.

## Train (DAS)

```bash
python train_classification_model.py --data 分类数据 --epochs 50 --batch-size 32 --lr 0.001
```

Default protocol (also listed in Table 1 of the paper):

- seed 42; stratified 6:2:2; 108 test samples per class
- Adam + StepLR; 50 epochs
- CCSTA with `Lc=2`, 8 heads
- S2 triplet regularizer is off in the main protocol (`TRIPLET_WEIGHT = 0`)

Checkpoints are written to `checkpoints/` (not included in this repository).

## Evaluate

```bash
python predict_on_testset.py
```

Requires `atcsa_splits.json` and the cascade weights produced by training.

## Latency profiling

```bash
python run_gpu_latency.py
```

Requires a CUDA GPU and trained cascade weights. Attention time is a subset of the same synchronized forward as the full network.

## External DAS backbones and UCR

Flat TimeSformer / Mamba-1D on the same SCP split:

```bash
python train_classification_model.py --model timesformer
python train_classification_model.py --model mamba
```

Public UCR official TRAIN/TEST splits (FordA, Wafer, ElectricDevices):

```bash
python train_classification_model.py --ucr FordA
python train_classification_model.py --ucr Wafer
python train_classification_model.py --ucr ElectricDevices
```

Or run the helper:

```bash
python run_external_baselines.py
```

## Files

| File | Role |
| --- | --- |
| `atcsa_model.py` | CNN trunk and CCSTA |
| `train_classification_model.py` | SCP, split, ACGC cascade training |
| `predict_on_testset.py` | Test-split evaluation |
| `run_gpu_latency.py` | CUDA-synchronized latency / attention FLOPs |
| `baseline_models.py` | TimeSformer and Mamba-1D |
| `train_ucr.py` | UCR download and 1D ACGC check |
| `run_external_baselines.py` | Backbone + UCR helper |

## License

Research code released to support the ATCSA-Net manuscript. The DAS corpus remains proprietary.
