# Fit a ProtoPNet using the CUB-200 dataset
- Download and extract the dataset from http://www.vision.caltech.edu/visipedia/CUB-200.html
- Process the images by running the processImages.py file. You will have to adjust the path variable, so that the path contains the CUB_200_2011 directory.
- The step above should create a datasets directory rooted in the chosen path.
- Fit the model by running fit_model.py. Once again you will have to adjust the path variable. This time the path variable will have to contain the datasets directory.
- Batch sizes, number of epochs and so forth can be adjusted near the end of the file. The variables are set up to run one warm epoch and then run joint epochs, so we can compare speeds.
- epoch times can be observed in the trainlog.txt file.


Package info:

# Name                    Version                   Build  Channel
mpmath                    1.2.1            py39h06a4308_0
sphinxcontrib-jsmath      1.0.1              pyhd3eb1b0_0
opencv-python             4.5.5.64                 pypi_0    pypi
blosc                     1.21.0               h8c45485_0
nose                      1.3.7           pyhd3eb1b0_1006
pycosat                   0.6.3            py39h27cfd23_0
matplotlib                3.4.3            py39h06a4308_0
matplotlib-base           3.4.3            py39hbbc1b5f_0
matplotlib-inline         0.1.2              pyhd3eb1b0_2
jinja2-time               0.2.0              pyhd3eb1b0_2
pytorch                   1.10.2          py3.9_cuda11.3_cudnn8.2.0_0    pytorch
pytorch-mutex             1.0                        cuda    pytorch
torchaudio                0.10.2               py39_cu113    pytorch
torchvision               0.11.3               py39_cu113    pytorch
