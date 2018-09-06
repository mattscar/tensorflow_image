The code in this repository uses TensorFlow and Python to analyze and classify images. The cnn_cifar.py module classifies images from the CIFAR-10 database and the img_proc.py module analyzes the content of input_aircraft.png.

# Classifying CIFAR Images

To test practical image recognition applications, the Canadian Institute for Advanced Research (CIFAR) provides the CIFAR-10 and CIFAR-100 datasets. These datasets contain color images and classification labels.

The main site for the CIFAR-10 and CIFAR-100 datasets is [here](https://www.cs.toronto.edu/~kriz/cifar.html). The site provides three links for downloading the CIFAR-10 dataset: one for the Python version, one for the Matlab version, and a binary version. The code in the cnn_cifar.py module assumes that the images are available in the cifar-10-batches-py folder. For example, the first batch is cifar-10-batches-py/data_batch_1, the second is cifar-10-batches-py/data_batch_2, and so on.

## Reading CIFAR-10 Images and Labels

CIFAR serializes the data in the CIFAR-10 files using a process called *pickling*. To read the data in Python, an application needs to import the `pickle` module and call its `load` method. For example, the following code accesses the data in data_batch_2:

```
import pickle
with open(‘cifar-10-batches-py/data_batch_2’, ‘rb’) as imgfile:
    dict = pickle.load(imgfile)
    imgfile.close()
```

The result is a dictionary with four keys:

- `batch_label`: Description of the batch
- `labels`: A list of the 10,000 labels of the batch’s images
- `data`: An ndarray containing the batch’s image data
- `filenames`: A list of the 10,000 PNGs that contain image data

Each image label is provided as an integer between 0 and 9. These values correspond to the ten categories that identify the content of the corresponding image. The categories are airplane (0), automobile (1), bird (2), cat (3), deer (4), dog (5), frog (6), horse (7), ship (8), and truck (9).

The ndarray provided by the data key contains 8-bit unsigned integers in a 10,000-by-3,072 element matrix. This matrix contains 10,000 rows and each row contains a 32-by-32 image with red, green, and blue components (32 x 32 x 3 = 3,072).

## Operation of cnn_cifar.py

The cnn_cifar.py module starts by loading the CIFAR-10 training images and labels. Then it performs four operations:
- Concatenates the training images into one (50,000-by-3,072) ndarray. Concatenates the training labels into one (50,000-by-1) ndarray.
- Converts the elements of the image ndarray to floating-point values.
- Reshapes the image ndarray to [50,000, 32, 32, 3]. The last element identifies the number of channels per pixel (R, G, and B).
- Converts the label ndarray to a one-shot ndarray (50,000-by-10).

To process the image data, the application creates four convolution layers, three dropout layers, and three pooling layers. The convolution layers use 32 filters of size 5x5, and they all use a ReLU to serve as the activation function. The pooling layers set their block size to 2x2 and their strides to 2. As a result, each output image of the pooling layer has half the dimensions of the corresponding input image.

After the convolution and pooling layers, the module flattens the image data and passes it to two fully-connected layers. The first fully-connected layer has 256 nodes and uses a ReLU to serve as its activation function. The second fully-connected layer has 10 nodes, and when its processing is complete, its output is passed to a softmax function for classification into one of the ten categories.

# Analyzing Images with TensorFlow

A TensorFlow application can remove noise from an image by performing convolution with a 3-by-3 filter with constant elements. The img_proc.py module performs convolution and five other operations:
- Changes the image’s contrast by calling tf.image.adjust_contrast
- Mirrors the image horizontally by calling tf.image.flip_left_right
- Converts the data to PNG format and writes the data to a PNG file
- Generates summary data for viewing the image in TensorBoard
- Stores the resulting pixels in a file named output_aircraft.png

The process of generating summary data for an image is like that of generating data for a tensor. The only difference is that the application needs to call `tf.summary.image` instead of `tf.summary.scalar` or `tf.summary.histogram`. The function’s signature is given as follows:

```
tf.summary.image(name, tensor, max_outputs=3, collections=None, family=None)
```

The `name` parameter provides the label that TensorFlow will associate with the image. The function accepts the image data through the tensor parameter, and the tensor’s shape must be [batch_size, height, width, num_channels].  

The img_proc.py module creates an operation that generates summary data for `img_tensor` with the following code:

```
summary_op = tf.summary.image(‘Output’, img_tensor)
```

After creating this operation, the application executes it in a session and uses a FileWriter to print the protocol buffer to an event file. When launched, TensorBoard will read this event file and display the graphical content of `img_tensor`.

  
  


