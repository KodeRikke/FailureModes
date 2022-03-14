# Fit a ProtoPNet using the CUB-200 dataset
- Download and extract the dataset from http://www.vision.caltech.edu/visipedia/CUB-200.html
- Process the images by running the processImages.py file. You will have to adjust the path variable, so that the path contains the CUB_200_2011 directory.
- The step above should create a datasets directory rooted in the chosen path.
- Fit the model by running fit_model.py. Once again you will have to adjust the path variable. This time the path variable will have to contain the datasets directory.
- Batch sizes, number of epochs and so forth can be adjusted near the end of the file. The variables are set up to run one warm epoch and then run joint epochs, so we can compare speeds.
- epoch times can be observed in the trainlog.txt file.

Package info can be seen in package_info.txt
