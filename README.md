# auto coloring line drawing using tensorflow

# Abstract
The main purpose of this project is to create anime image only using line drawing.  

# Feature
Tensorflow is utilized for neural network of the work to transplant some environment and devices.  

# Future work
build up ios painting application  
deploy the web services   


# Result
I've been experienced some loss function to get most effective result. 

The exprience list is this 
only L1 Distance 
L1 Distance + DCGAN ( this is same pix2pix project) 
Wasserstein gan not added L1 Distance  
use Selu , activation function, for DCGAN  

The most better result in the above list is lead by L1 Distance + DCGAN.
And some experience show the result that the coefficent with DCGAN is abount 1.1.

This is the result. See below.

# reference
Some code of this project is on  
https://github.com/yenchenlin/pix2pix-tensorflow

The paper, pix2pix  
https://arxiv.org/pdf/1611.07004v1.pdf
