# CF-OCT-segmentation
Semantic segmentation and quantification of fibrosis in a novel **Cochlear OCT** dataset

Benchmarking different Keras segmentation models on our new **Cochlear OCT** dataset:

Proposed 2D-OCT-UNET

VGG16-UNET

UEfficientNet

SegFormer

MST-DeepLabv3

To benchmark the SAM (Segment Anything Model) on the **Cochlear OCT** dataset we refer to the Github repository:
[https://github.com/mazurowski-lab/finetune-SAM]

## Dataset
The open-source dataset can be dowloaded from the Open Science Framework (OSF) 

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
