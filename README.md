This is a repository that contains my solution for assignments from the course EECS 498-007 [(2020 version)](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/). The lecture videos from 2019 are found [here](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r). The course teaches deep learning methods used for computer vision applications.

Here is a high-level overview of the topics covered in each assignment, the assignments are done using `pytorch`:
* A1: `PyTorch` practice and implemented a k-nearest neighbor classifier.
* A2: Implemented a multi-class **support vector machine classifier**, a **softmax regression classifier**, and a **two-layer neural network classifier**. Wrote vectorized gradient code for back-propagation.
* A3: 
  * `fully_connected_networks`: Implemented modularized **multi-layer fully-connected neural networks** from scratch, implemented various **optimizers** including SGD+Momentum, RMSProp, Adam, and implemented **dropout**.
* A4: 
  * `pytorch_autograd_and_nn`: Practiced using `autograd` to perform automatic gradient computation, used `nn.Module` and `nn.Sequential` to build fully-connected and convolutional networks, and implemented **PreResNet32** (a variant of ResNet) for CIFAR-10 image classification.
  * `rnn_lstm_attention_captioning`: Implemented **RNN**, **LSTM**, **LSTM + attention** architectures for training an image captioning model.
  * `network_visualization`: Implemented **saliency maps**, **adversarial attacks** and **synthesized images**
  * `style_transfer`: Performed neural style transfer by computing the **gram matrix** and using **total variation regularization**
* A6:
  * ` variational_autoencoders`: Implemented VAE and conditional VAE to generate new sample digits using the MNIST dataset
