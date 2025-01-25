# CF-OCT-segmentation
Semantic segmentation and quantification of cochlear fibrosis (CF) in a novel **Cochlear OCT** dataset

Benchmarking different Keras segmentation models on our new **Cochlear OCT** dataset:

Proposed 2D-OCT-UNET

![ARO_Unet_architecture](https://github.com/user-attachments/assets/17a8e4e8-7570-4400-8929-961ca4513651)

**Fig. 1.** Block diagram of the proposed 2D-OCT-UNET architecture for the multiclass OCT segmentation used in this work. The very deep 2D-
OCT-UNET processes two-dimensional OCT slices and consists of seven encoder-decoder blocks with skip connections. The numbers above the
encoder, bottleneck and decoder blocks indicate the number of filters in the convolutional Conv2D layers. GN stands for Group Normalization layers
with the number of groups parameter ng = 2. The input resolution of the UNET is set to be 1024 × 1024 pixels. We include Dropout(0.1) layers
only in the decoder. The number of filters in the last Conv2D layer is equal to the nc = 4 (number of classes).

Other models included are: VGG16-UNET, UEfficientNet, SegFormer and MST-DeepLabv3

To benchmark the SAM (Segment Anything Model) on the **Cochlear OCT** dataset we refer to the Github repository:
[https://github.com/mazurowski-lab/finetune-SAM]

## Dataset
The open-source dataset can be dowloaded from the Open Science Framework (OSF) 
The dataset link will be provided soon

![dataset_sample](https://github.com/user-attachments/assets/ffca9dd8-bb54-4edd-a72a-2f45519d94ac)
**Fig. 2.** Cochlear OCT dataset samples and the corresponding ground truth annotations from five annotated volumes. The CI/Track class is depicted
in red, the Fibrosis class in green and the ST/Free Space class in blue.

## Training
Use _OHSU_UNET_cochlear_multiclass_4paper.py_ to train the proposed 2D-OCT-UNET, UEfficientNet, VGG16-UNET and the MST-DeepLabv3 models.
Use _OHSU_UNET_cochlear_multiclass_Segformer_4paper.py_ to train the SegFormer model.

## Quantification of Fibrosis
The code for the quantification of Fibrosis is embedded into _OHSU_UNET_cochlear_multiclass_4paper.py_ and _OHSU_UNET_cochlear_multiclass_Segformer_4paper.py_.
The code will compute the amount of fibrosis for the raw volumes OCTV1L, OCTV7L, OCTV9L, OCTV10L and OCTV11L.

## Citation
Please cite our paper if you find our codes or paper helpful

Dietlmeier, J., Greenberg, B., He, W., Wilson, T., Xing, R., Hill, J., Fettig, A., Otto, M., Rounsavill, T., Reiss, L.A.J., Yi, J. O'Connor, N.E., Burwood, G.W.S. (2025). 
**Towards Investigating Residual Hearing Loss: Quantification of Fibrosis in a Novel Cochlear OCT Dataset**. 
IEEE Transactions on Biomedical Engineering, vol(no), pp-pp.
