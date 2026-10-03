Note: Navigate to "Codes" folder for executing the notebooks

The code files are organized in sequential order from 1 to 4 to reflect the workflow from preprocessing to multimodal analysis for easier navigation and understanding.

The python version = 3.11.11
-------------------------------------------------------------------

# Data Preprocessing #

**Combining User Data and Changing Sampling Frequency**

For merging data from all users into a single file and adjusting data frequency, refer to the provided example script "1.Combine_Data_and_Change_Frequency.ipynb". An example of changing the sample rate to 4 Hz is included in the code.


**Concatenating and Labeling Data**

After preprocessing, separate scripts are provided to concatenate and label different types of data, such as:

- Biometric signals

- Behavioral data

- Facial features

Each modality has a dedicated script for efficient processing.

--------------------------------------------------------------------

# Classification and Analysis #

**Classification Models**

The directory contains classification scripts for both unimodal and multimodal data, allowing for performance comparison and predictive modeling.

**Statistical Analysis**

The repository contains statistical analyses on biometric signals, including reported plots referenced in the associated paper.

--------------------------------------------------------------------

## Extracting Features from Video Data ##

To extract features from video data, run the "main.ipynb" file located in the "Feature_Extraction" folder. This script extracts multiple features, including:

- 2D and 3D facial landmarks

- Pose landmarks

- Facial action units

Note: 3D facial landmarks are not available in the dataset.


* For any questions or further clarifications, please refer to the provided scripts and documentation.