# IRWSR
Python implementation of the Iterative Re-weighted Super Resolution (IRWSR) scheme developed by Köhler et al., published in the article [Robust Multiframe Super-Resolution Employing
Iteratively Re-Weighted Minimization](https://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2016/Kohler16-RMS.pdf).

The method is a Robust multi-frame super resolution scheme using a Bayesian regularization with spatial weighting. The scheme is implemented with an image registration method using optical flow through OpenCV. Three types of data usage is avalible:

Datatype | Description
------------ | -------------
Artifical Degradation | Takes a test image - degrades with known stepes - creates a super resolved image
Real data with reference | Upscales the LR frames with the IRWSR and compares with the reference image
Real data without reference | Upscales the LR frames with the IRWSR

The scripyt name *defineSystem.py* carries all the user-defined variables needed for running the IRWSR. 

The script named *degradeTestImage.py* degrades the given test images and upscales them. No motion estimation is needed as the translations are given by the degradation.

Further documentation and ease of use will hopefully be introduced soon. 


