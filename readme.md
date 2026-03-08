# **MULTITASK MODEL IN THE DIAGNOSIS OF MIDDLE EAR DIEASE**
## **WORK SUMMARY**
 Using backbone DEEPLABV3-RESNET101, multitask learning models allow for the simultaneous optimization of multiple prediction tasks on the same backbone architecture. In my work is classification and segmentation of middle ear diseases.
## **Installation**
Ensure you are using **Python 3.10**. \
If you not install follow this to install:  [Python 3.10](https://www.python.org/downloads/release/python-3100/) or create virtual environment with anaconda:
1. Create virtual environment
```sh
conda create -n venv python=3.10
```
2. Activate environment
```sh
conda activate venv
```
3. Install required libraries
```sh
pip install -r requirements.txt
```
## **Train**
After configuring the environment at file `config.py`, Start training the model.\
Create folder `Experience` to save the training and testing results.\
Run file train.py:
```sh
python train.py
```
## **Test**
Run file inference.py:
```sh
python inference.py
```
