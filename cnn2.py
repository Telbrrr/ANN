import numpy as np

def apply_convolution(image, kernel, stride=1, padding=0):
    '''Apply convolution operation to an image with given kernel.'''
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
    
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    output_height = (img_height - kernel_height) // stride + 1
    output_width = (img_width - kernel_width) // stride + 1
    
    output = np.zeros((output_height, output_width))
    
    for y in range(0, output_height):
        for x in range(0, output_width):
            y_start = y * stride
            y_end = y_start + kernel_height
            x_start = x * stride
            x_end = x_start + kernel_width
            
            window = image[y_start:y_end, x_start:x_end]
            
            output[y, x] = np.sum(window * kernel)
    
    return output

def relu_activation(x):
    '''Apply ReLU activation function.'''
    return np.maximum(0, x)

def max_pooling(image, pool_size=2, stride=2):
    '''Apply max pooling operation.'''
    img_height, img_width = image.shape
    
    output_height = (img_height - pool_size) // stride + 1
    output_width = (img_width - pool_size) // stride + 1
    
    output = np.zeros((output_height, output_width))
    
    for y in range(0, output_height):
        for x in range(0, output_width):
            y_start = y * stride
            y_end = y_start + pool_size
            x_start = x * stride
            x_end = x_start + pool_size
            
            window = image[y_start:y_end, x_start:x_end]
            
            output[y, x] = np.max(window)
    
    return output

def process_cnn(image, kernel, conv_stride=2, padding=1, pool_size=2, pool_stride=2):
    '''Complete CNN processing: Conv -> ReLU -> MaxPooling.'''
    conv_output = apply_convolution(image, kernel, stride=conv_stride, padding=padding)
    relu_output = relu_activation(conv_output)
    pool_output = max_pooling(relu_output, pool_size=pool_size, stride=pool_stride)
    return conv_output, relu_output, pool_output

def calculate_flatten_size(input_shape, conv1_params, pool1_params, conv2_params, pool2_params):
    '''Calculate the size of the flatten layer after two complete convolution layers.'''
    h, w = input_shape
    f_h, f_w = conv1_params['filter_size']
    s = conv1_params['stride']
    p = conv1_params['padding']
    
    h_conv1 = (h - f_h + 2*p) // s + 1
    w_conv1 = (w - f_w + 2*p) // s + 1
    
    p_h, p_w = pool1_params['filter_size']
    s_p = pool1_params['stride']
    
    h_pool1 = (h_conv1 - p_h) // s_p + 1
    w_pool1 = (w_conv1 - p_w) // s_p + 1
    
    f_h2, f_w2 = conv2_params['filter_size']
    s2 = conv2_params['stride']
    p2 = conv2_params['padding']
    
    h_conv2 = (h_pool1 - f_h2 + 2*p2) // s2 + 1
    w_conv2 = (w_pool1 - f_w2 + 2*p2) // s2 + 1
    
    p_h2, p_w2 = pool2_params['filter_size']
    s_p2 = pool2_params['stride']
    
    h_pool2 = (h_conv2 - p_h2) // s_p2 + 1
    w_pool2 = (w_conv2 - p_w2) // s_p2 + 1
    
    return h_pool2 * w_pool2

def main():
    R = np.array([
        [112, 125, 25, 80, 220, 110],
        [150, 95, 15, 100, 115, 152],
        [200, 100, 48, 90, 70, 175],
        [187, 56, 43, 86, 180, 200],
        [190, 87, 70, 37, 24, 35],
        [80, 75, 65, 45, 32, 20]
    ])

    G = np.array([
        [150, 125, 38, 80, 20, 10],
        [130, 95, 25, 100, 115, 152],
        [80, 100, 148, 90, 70, 175],
        [170, 160, 43, 160, 170, 180],
        [100, 150, 70, 37, 124, 135],
        [85, 75, 65, 45, 232, 120]
    ])

    B = np.array([
        [200, 125, 25, 80, 220, 150],
        [50, 95, 15, 150, 115, 152],
        [90, 110, 48, 190, 70, 175],
        [180, 135, 43, 106, 180, 110],
        [55, 98, 70, 37, 24, 35],
        [78, 150, 65, 45, 32, 80]
    ])

    kernel_R = np.array([
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [-1, 0, 0, 1],
        [-1, 0, 1, -1]
    ])

    kernel_G = np.array([
        [0, -1, -1, 0],
        [1, -1, 1, -1],
        [1, 0, 0, 1],
        [1, 0, 1, 1]
    ])

    kernel_B = np.array([
        [1, 1, 1, 0],
        [-1, 1, 1, 1],
        [0, 1, 0, 1],
        [-1, -1, 1, 1]
    ])

    print("Processing Red Channel:")
    conv_R, relu_R, pool_R = process_cnn(R, kernel_R, conv_stride=2, padding=1, pool_size=2, pool_stride=2)
    print("\nConvolution Output (R):")
    print(conv_R)
    print("\nReLU Activation (R):")
    print(relu_R)
    print("\nMax Pooling Output (R):")
    print(pool_R)

    print("\nProcessing Green Channel:")
    conv_G, relu_G, pool_G = process_cnn(G, kernel_G, conv_stride=2, padding=1, pool_size=2, pool_stride=2)
    print("\nConvolution Output (G):")
    print(conv_G)
    print("\nReLU Activation (G):")
    print(relu_G)
    print("\nMax Pooling Output (G):")
    print(pool_G)

    print("\nProcessing Blue Channel:")
    conv_B, relu_B, pool_B = process_cnn(B, kernel_B, conv_stride=2, padding=1, pool_size=2, pool_stride=2)
    print("\nConvolution Output (B):")
    print(conv_B)
    print("\nReLU Activation (B):")
    print(relu_B)
    print("\nMax Pooling Output (B):")
    print(pool_B)

    input_shape = (6, 6)

    conv1_params = {
        'filter_size': (4, 4),
        'stride': 2,
        'padding': 1
    }

    pool1_params = {
        'filter_size': (2, 2),
        'stride': 2
    }

    conv2_params = {
        'filter_size': (3, 3),
        'stride': 1,
        'padding': 0
    }

    pool2_params = {
        'filter_size': (2, 2),
        'stride': 2
    }

    flatten_size = calculate_flatten_size(input_shape, conv1_params, pool1_params, conv2_params, pool2_params)
    print(f"\nFlatten layer size after 2 complete convolution layers: {flatten_size}")

if __name__ == "__main__":
    main()