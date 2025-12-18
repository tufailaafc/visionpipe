## Brief overview of installation and use.

Instructions also found at: https://deeplabcut.github.io/DeepLabCut/docs/installation.html
2025/12/18

Prerequisets:
Conda
Linux(Ubuntu)
Terminal


# Installation:
```bash
conda create -n DEEPLABCUT python=3.12
conda activate DEEPLABCUT
conda install -c conda-forge pytables==3.8.0
```

If you want gpu support install pytorch for correct cuda version

```bash
pip3 install torch torchvision
```

Verify cuda compatability

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

```bash
pip install "deeplabcut[gui,tf]"
```
or just if you do not need tensor flow
```bash
pip install "deeplabcut[gui]"
```
^ This is the one I did - Nathaniel Yeo

May need to downgrade to: 
```bash
pip install pandas==2.1.4 - Nathaniel Yeo
```