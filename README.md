# 3DSS-Pytorch

**2D → 3D Semantic Segmentation**

---

## Setup 

1. **Clone the repository**

```bash
git clone https://github.com/ryan-mangeno/3DSS-Pytorch.git
cd 3DSS-Pytorch
```

2. **Create a virtual environment**

```bash
python3 -m venv .env
source .env/bin/activate  # Linux / Mac
# .env\Scripts\activate    # Windows
```

3. **Install requirements**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Download the dataset**

* Download `train.zip` and `train_masks.zip` from the [Carvana Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/data)
* Place them in the `data/` folder:

```
3DSS-Pytorch/data/train.zip
3DSS-Pytorch/data/train_masks.zip
```

5. **Run training**

```bash
python train.py
```

---

## Notes

* The training script automatically extracts the ZIP files to `data/unzipped/`.
* Make sure your machine has enough RAM and GPU memory if using a large batch size.
* Outputs:

  * Model checkpoint: `my_checkpoint.pth`
  * Training/validation loss and DICE plots
