# faster-rcnn

implementing faster-rcnn...

## data

- dataset
  - `inverse_normalize(img)`
    逆正则化：将图片矩阵乘以标准差再加上均值（pytorch 模型$x\sigma+\mu$），只加上均值（caffee 模型$x+\mu$）。
  - `pytorch_normalize` pytorch 模型标准化。使用 torchversion 的函数对各个通道的数据减去均值再除以标准差,$\frac{x-\mu}{\sigma}$。
