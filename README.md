# MedSAM
Callistoの放射線画像アノテーションツールNova Slicerの機能"MedSAM"で使用するモデルのカスタマイズ版。
元のMedSAMでは画像形式がdicomに対応していないなどの問題点があるため、簡便にAnnotation Serverで使用できるように改良を行った。


## Installation
1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/bowang-lab/MedSAM`
4. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`


## Get Started
Download the [model checkpoint](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link) and place it at e.g., `work_dir/MedSAM/medsam_vit_b`

We provide three ways to quickly test the model on your images

1. Command line

```bash
python MedSAM_Inference.py # segment the demo image
```

Segment other images with the following flags
```bash
-i input_img
-o output path
--box bounding box of the segmentation target
```

### todo
- [ ] スクリプトで呼び出して、タスクを待機させておけるクラスを実装