# Replication of Conditional Neural Processes 

This work is due equally to Igor Adamski, Conor Foy and Woramanot Yomjinda.

This repository contains notebook replications of the 1D regression and MNIST examples outlined in the Conditional Neural Processes paper [1](https://arxiv.org/pdf/1807.01613.pdf), baesd on tensorflow 2.0.0

Added **conditional_neural_process_with_tensorflow_ver_2.0.0_Ex1_1D_Regression** notebook, a modified version of the original  **conditional_neural_process** notebook to run with tensorflow version 2.0.0. The descriptions and functionality of the original have been kept where possible, but as a result the descriptions don't make perfect sense in the context of the new code. 
Added **CNP_1d_reg_e3_d5_v1.h5** the weights after training for 40,000 iterations.

Added **conditional_neural_process_with_tensorflow_ver_2.0.0_Ex2_MNIST** notebook, my own implimentation, replicating the second example described in the paper using tensorflow version 2.0.0.
Added **CNP_mnist_e3_d5_v1.h5** the weights after training for 200,000 iterations.

The notebook **music_completion_clean.ipynb** contains all instructions for replicating our music generation with CNPs results. Example samples generated with this process can be found in [here](https://soundcloud.com/igor-adamski1/sets/conditional-naural-processes-improv).

The notebooks have been tested with the following versions:

*   Numpy                        (1.18.1)
*   tensorflow-gpu               (2.0.0)                 
*   tensorflow-probability       (0.8.0)
*   Keras                        (2.3.1)
*   Matplotlib                   (3.0.3)


1. **Conditional Neural Processes**: Garnelo M, Rosenbaum D, Maddison CJ, Ramalho T, Saxton D, Shanahan M, Teh YW,
Rezende DJ, Eslami SM. *Conditional Neural Processes*. In International Conference
on Machine Learning 2018
