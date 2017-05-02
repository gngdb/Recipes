# Network in Network MNIST Model
# Original source: https://github.com/mavenlin/cuda-convnet/blob/master/NIN/mnist_def
# License: unknown

# Download pretrained weights from:
# ????

from lasagne.layers import InputLayer, DropoutLayer, FlattenLayer, NINLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax

def build_model():
    net = {}
    net['input'] = InputLayer((None, 1, 28, 28))
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=96,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    net['cccp1'] = NINLayer(net['conv1'], num_units=64)
    net['cccp2'] = NINLayer(net['cccp1'], num_units=48)
    net['pool1'] = PoolLayer(net['cccp2'],
                             pool_size=3,
                             stride=2,
                             mode='max',
                             ignore_border=False)
    net['drop3'] = DropoutLayer(net['pool1'], p=0.5)
    net['conv2'] = ConvLayer(net['drop3'],
                             num_filters=128,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    net['cccp3'] = NINLayer(net['conv2'], num_units=96)
    net['cccp4'] = NINLayer(net['cccp3'], num_units=48)
    net['pool2'] = PoolLayer(net['cccp4'],
                             pool_size=3,
                             stride=2,
                             mode='max',
                             ignore_border=False)
    net['drop6'] = DropoutLayer(net['pool2'], p=0.5)
    net['conv3'] = ConvLayer(net['drop6'],
                             num_filters=128,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    net['cccp5'] = NINLayer(net['conv3'], num_units=96)
    net['cccp6'] = NINLayer(net['cccp5'], num_units=10)
    net['pool3'] = PoolLayer(net['cccp6'],
                             pool_size=7,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['output'] = FlattenLayer(net['pool3'])
    net['probs'] = NonlinearityLayer(net['output'], nonlinearity=softmax)

    return net

def build_faster_model():
    net = {}
    net['input'] = InputLayer((None, 1, 28, 28))
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=96,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    net['cccp1'] = ConvLayer(
        net['conv1'], num_filters=64, filter_size=1, flip_filters=False)
    net['cccp2'] = ConvLayer(
        net['cccp1'], num_filters=48, filter_size=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['cccp2'],
                             pool_size=3,
                             stride=2,
                             mode='max',
                             ignore_border=False)
    net['drop3'] = DropoutLayer(net['pool1'], p=0.5)
    net['conv2'] = ConvLayer(net['drop3'],
                             num_filters=128,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    net['cccp3'] = ConvLayer(
        net['conv2'], num_filters=96, filter_size=1, flip_filters=False)
    net['cccp4'] = ConvLayer(
        net['cccp3'], num_filters=48, filter_size=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['cccp4'],
                             pool_size=3,
                             stride=2,
                             mode='max',
                             ignore_border=False)
    net['drop6'] = DropoutLayer(net['pool2'], p=0.5)
    net['conv3'] = ConvLayer(net['drop6'],
                             num_filters=128,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    net['cccp5'] = ConvLayer(
        net['conv3'], num_filters=96, filter_size=1, flip_filters=False)
    net['cccp6'] = ConvLayer(
        net['cccp5'], num_filters=10, filter_size=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['cccp6'],
                             pool_size=7,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['output'] = FlattenLayer(net['pool3'])
    net['probs'] = NonlinearityLayer(net['output'], nonlinearity=softmax)

    return net
