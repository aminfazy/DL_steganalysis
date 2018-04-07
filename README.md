# DL_steganalysis
Deep learning based image steganalysis in image spatial domain 
This is keras based implementation using tensorflow backend.
This program is written to work on both cpu as well as gpu with slight modification.
It istested on Nvidia GTX 1080 titan gpu.
Dependencies: 
  1. keras
  2. Tensorflow
  3. Numpy
  4. scipy
  5. Matplotlib
  6. PIL
  7. Sklearn

Dataset requirements:
  BOSSbase Dataset can be obtained from http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip
  
For data hiding any tool available can be used as per avialability
 For testing alethia can be used to generate stego images in batches
 Alethia can be downloaded from the respective author's github link provided below:
 https://github.com/daniellerch/aletheia.git
 Read the documentation of aletheia to find out the respective dependencies to run that
 
After obtaining stego and cover images you are ready to use the model
Make sure to save the fnet.py and other files in same directory where database of stego and cover images are kept
or else make the corresponding changes to the paths in the fnet.py
