## LED-Net: Lightweight Encoder-Decoder Network for Underwater image enhancement

## 所需环境
torch==1.12.0

## 文件下载
The download addresses of UIEB, LSUI, EUVP, and RUIE datasets are as follows:
1. UIEB:[UIEB Dataset](https://li-chongyi.github.io/proj_benchmark.html)
2. LSUI:[LSUI Dataset](https://drive.google.com/file/d/10gD4s12uJxCHcuFdX9Khkv37zzBwNFbL/view)
3. EUVP:[EUVP Dataset](https://drive.google.com/drive/folders/1ZEql33CajGfHHzPe1vFxUFCMcP0YbZb3)
4. RUIE:[RUIE Dataset](https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark)

The download addresses of the weights we trained on different data sets are as follows：
1. UIEB:[Weight](https://drive.google.com/drive/folders/1I1BHKILdywNDRwDeHR6j44j3Pyfjz5e0)
2. LSUI:[Weight](https://drive.google.com/drive/folders/1I5Pj4J7_kj4qHCq0b445zlTP2DypL_dx)
3. EUVP:[Weight](https://drive.google.com/drive/folders/1oU_Tm7Ce9p4eu7nVaAT_nFJVxo64-zel)

## 训练
```python
python train.py
```
## 测试
```python
python test.py
```





