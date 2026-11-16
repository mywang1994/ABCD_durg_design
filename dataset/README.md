# Dataset

## Download

1. **CrossDocked:** Download [crossdocked_pocket10.tar.gz](https://drive.google.com/file/d/10KGuj15mxOJ2FBsduun2Lggzx0yPreEU/view) and extract it under `dataset/data/`. Name the extracted folder `crossdocked_v1.1_rmsd1.0_pocket10`; keep its target subfolders and filenames unchanged.
2. **Train/test split:** Download `split_by_name.pt` from the [Pocket2Mol data page](https://github.com/pengxingang/Pocket2Mol/blob/main/data/README.md) and place it in `dataset/data/`. Both splits use the structures above.
3. **MMP database:** Download `chembl33_sa25_f5.db.gz` from [Zenodo](https://doi.org/10.5281/zenodo.16909329), decompress it and rename the database to `mmp.db`.

## Directory Structure

After extraction, arrange the files as follows:

```text
dataset/data/
├── crossdocked_v1.1_rmsd1.0_pocket10/
│   └── TARGET_DIRECTORY/
│       ├── *_pocket10.pdb
│       └── *.sdf
├── split_by_name.pt
└── mmp.db
```
