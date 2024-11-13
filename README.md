<h2>Tensorflow-Image-Segmentation-Breast-Cancer-Cell (2024/11/14)</h2>

This is the first experiment of Image Segmentation for Malignant Breast Cancer Cell
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
 <a href="https://drive.google.com/file/d/1g1LA72c4vhv1JznkFLcQ_gKnhk31q4CJ/view?usp=drive_link">
Malignant-Breast-Cancer-Cell-ImageMask-Dataset.zip</a>, which was derived by us from  
<a href="https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation">
Breast Cancer Cell Segmentation (58 histopathological images with expert annotations)
</a>
<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 896x768 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/images/ytma10_010704_malignant1_ccd.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/masks/ytma10_010704_malignant1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test_output/ytma10_010704_malignant1_ccd.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/images/ytma10_010704_malignant3_ccd.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/masks/ytma10_010704_malignant3.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test_output/ytma10_010704_malignant3_ccd.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/images/ytma49_111003_malignant1_ccd.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/masks/ytma49_111003_malignant1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test_output/ytma49_111003_malignant1_ccd.jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Breast-Cancer-CellSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>

<a href="https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation">
Breast Cancer Cell Segmentation (58 histopathological images with expert annotations)
</a>
<br><br>
<b>About Dataset</b><br>
In this dataset, there are 58 H&E stained histopathology images used in breast cancer cell <br
detection with associated ground truth data available. Routine histology uses the stain <br>b
combination of hematoxylin and eosin, commonly referred to as H&E. These images are stained<br>
 since most cells are essentially transparent, with little or no intrinsic pigment. Certain <br>
 special stains, which bind selectively to particular components, are be used to identify <br>
 biological structures such as cells. In those images, the challenging problem is cell segmentation<br>
 for subsequent classification in benign and malignant cells.<br>
<br>
<b>How to Cite this Dataset</b><br>
If you use this dataset in your research, please credit the authors.<br>
<br>
<b>Original Article</b><br>
E. Drelie Gelasca, J. Byun, B. Obara and B. S. Manjunath, <br>
"Evaluation and benchmark for biological image segmentation,"<br>
 2008 15th IEEE International Conference on Image Processing, <br>
 San Diego, CA, 2008, pp. 1816-1819<br>.
<br>
BibTeX<br>
@inproceedings{Drelie08-298,<br>
author = {Elisa Drelie Gelasca and Jiyun Byun and Boguslaw Obara and B.S. Manjunath},<br>
title = {Evaluation and Benchmark for Biological Image Segmentation},<br>
booktitle = {IEEE International Conference on Image Processing},<br>
location = {San Diego, CA},<br>
month = {Oct},<br>
year = {2008},<br>
url = {http://vision.ece.ucsb.edu/publications/elisa_ICIP08.pdf}}<br>

<br>
<h3>
<a id="2">
2 Breast-Cancer-Cell ImageMask Dataset
</a>
</h3>
 If you would like to train this Breast-Cancer-Cell Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1g1LA72c4vhv1JznkFLcQ_gKnhk31q4CJ/view?usp=drive_link">
Malignant-Breast-Cancer-Cell-ImageMask-Dataset.zip</a>,
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Breast-Cancer-Cell
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>

<b>Breast-Cancer-Cell Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/Breast-Cancer-Cell_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_Ssample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained Breast-Cancer-CellTensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Celland run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (7,7)
num_layers     = 8
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>


<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/asset/epoch_change_infer.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 58  by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/asset/train_console_output_at_epoch_58.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Breast-Cancer-Cell.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/asset/evaluate_console_output_at_epoch_58.png" width="720" height="auto">
<br><br>Image-Segmentation-Breast-Cancer-Cell

<a href="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Breast-Cancer-Cell/test was not low, and dice_coef not high as shown below.
<br>
<pre>
loss,0.2298
dice_coef,0.6124
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Breast-Cancer-Cell.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/images/ytma10_010704_malignant2_ccd.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/masks/ytma10_010704_malignant2.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test_output/ytma10_010704_malignant2_ccd.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/images/ytma23_022103_malignant1_ccd.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/masks/ytma23_022103_malignant1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test_output/ytma23_022103_malignant1_ccd.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/images/ytma23_022103_malignant3_ccd.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/masks/ytma23_022103_malignant3.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test_output/ytma23_022103_malignant3_ccd.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/images/ytma49_042003_malignant3_ccd.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/masks/ytma49_042003_malignant3.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test_output/ytma49_042003_malignant3_ccd.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/images/ytma49_042203_malignant1_ccd.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/masks/ytma49_042203_malignant1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test_output/ytma49_042203_malignant1_ccd.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/images/ytma49_111303_malignant3_ccd.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test/masks/ytma49_111303_malignant3.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Breast-Cancer-Cell/mini_test_output/ytma49_111303_malignant3_ccd.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Breast Cancer Cell Segmentation (58 histopathological images with expert annotations)</b><br>
<a href="https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation">
https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation
</a>

