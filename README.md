# faster-rcnn

implementing faster-rcnn...

## data

- dataset

  - `inverse_normalize(img)`
    逆正则化：将图片矩阵乘以标准差再加上均值（pytorch 模型$x\sigma+\mu$），只加上均值（caffe 模型$x+\mu$）。

  - `pytorch_normalize` pytorch 模型标准化。

    使用 torchversion 的函数对各个通道的数据减去均值再除以标准差,$\frac{x-\mu}{\sigma}$。

  - `caffe_normalize(img)` caffe 模型标准化。只减均值，$x-\mu$。

  - `preprocess(img, min_size=600, max_size=1000)`图像预处理。

    对输入的图像缩放到 600\*1000，即原图长宽不足的放大，比这更大的缩小，同时保证长宽中至少一个值为 600 或 1000，并且长宽分别不大于这个值。完成缩放后对图像正则化。

  - `Transform`类。

    封装了预处理函数，并对缩放后对应的 bbox 调整坐标，之后对图像随机进行水平翻转，提高模型健壮性。

  - `Dataset`类

    确定数据集参数，并指定获取样本迭代器方法。

  - `TestDataset`类

    指定测试集参数，并指定获取样本迭代器方法。

- util
  - `read_image(path, dtype=np.float32, color=True)`。从 path 读取图片，指定 numpy 类型和图片颜色信息。从 path 读取图片，转换为 numpy 矩阵，并把图片矩阵改为通道优先。
  - `resize_bbox(bbox, in_size, out_size)`。根据图片缩放后的大小关系调整 bbox 的坐标使其前后对应。
  - `flip_bbox(bbox, size, y_flip=False, x_flip=False)。`根据需要对 bbox 位置进行翻转，一般在对图片进行翻转之后调用。
  - `random_flip(img, y_random=False, x_random=False,return_param=False, copy=False)`。对图片进行随机翻转。`return_param`指定是否返回图像经过了翻转。
  - `_slice_to_bounds(slice_)` 确保`slice_`切片值为合法范围 0~np.inf之间而非None，用于指明图片被裁剪后的范围。
  - `crop_bbox(bbox,y_slice=None,x_slice=None,allow_outside_center=True,return_param=False)` 对bbox进行转换，使其适合裁剪后的图片。通常和图片裁剪方法一起使用。给出裁剪范围`y_slice`和`x_slice` ，`allow_outside_center` 指明是否保留bbox中心在裁剪后图片之外的bbox，`return_param`指明是否返回保留的bbox的索引。此方法会得到在裁剪后图片中合适的bbox坐标。
  - `translate_bbox(bbox, y_offset=0, x_offset=0)` 用于转换bbox的坐标。在bbox的基础上加上一个偏移值到新的位置。

- voc_dataset

  - `VOCBboxDataset`类。指定使用VOC 数据集Bounding Box的参数，并获得Bounding Box数据。

    - `__init__(self, data_dir, split='trainval',use_difficult=False, return_difficult=False,)` 初始化。`data_dir`指定训练数据的根目录，`split`指定数据集的一个划分如：train/val/trainval/test。查看`ImageSet/Main`获得具体标注文本文件。
    - `get_example`获得标注后再查找xml文件，获得图片、具体Bounding Box的数据、标签，难度。

  - VOC_BBOX_LABEL_NAMES

    VOC数据集的类别名称

## model

### utils

- bbox_tools
  - `loc2bbox(src_bbox,loc)`。将源bounding box 坐标通过偏移和缩放参数组得到新的坐标。
  - `bbox2loc(src_bbox,dst_bbox)`。给定两个bounding box坐标，计算两者之间的偏移值和缩放值参数。
  - `bbox_iou(bbox_a,bbox_b)`。计算两个bounding box的交并比。注意，结果为a中每个项与b中每个项的交并比。
  - `generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],anchor_scales=[8, 16, 32])`生成9个基础anchor坐标，以左上角为基准，生成长宽比为0.5,1,2的9个坐标，可通过坐标偏移，求得所有特征点的anchor。
- creator_tool
  - `AnchorTargetCreator`类。利用每张图bbox的真实标签来为所有anchor分配**二分类**的真实标签。
  - `ProposalCreator`类。提供Roi，其中使用了NMS筛选。
  - `ProposalTargetCreator`类。为Roi分配多类别的真是标签，进一步筛选，提供更少、更精确的Roi。
  - `_get_inside_index(anchor, H, W)`，获得在图片范围内部的anchor坐标。
  - `_unmap(data, count, index, fill=0)`，将Index的索引置为data的值，其他填充为fill。



## region_proposal_network





## faster_rcnn





## roi_module





## faster_rcnn_vgg16

