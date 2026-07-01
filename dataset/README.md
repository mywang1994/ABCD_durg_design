# Dataset

## CrossDocked2020

Training data (CrossDocked2020-based) can be downloaded from:
https://drive.google.com/file/d/1q5tfmZ0mekacgFZ289Y0DrykhqNb66K8/view?usp=drive_link

Extract to `dataset/crossdocked_pocket10/`.

## MMP Fragment Database (`mmp.db`)

`mmp.db` is a [CReM](https://github.com/DrrDom/crem) fragment database used by `pool_builders.py` (`build_local`) to generate **local MMP-like states** around a reference ligand (fragment MUTATE / optional GROW).

**Download:** [Zenodo](https://doi.org/10.5281/zenodo.16909329) — recommended `chembl33_sa25_f5.db.gz` (~1.9gb). Decompress and save as `dataset/mmp.db`.

**Enable CReM:**
```bash
pip install crem
export ABCM_FRAGMENT_SUBST_MODULE=crem.crem
python train.py --mode crossdock --mmp-db dataset/mmp.db
```

Without `mmp.db` or CReM, local states fall back to simple bond/leaf pruning.
