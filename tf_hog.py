import tensorflow as tf
from PIL import Image
import numpy as np
# https://github.com/KCC13/tf-HOG/blob/master/tf_filters.py
def plot_two_images_diagonally(i1, i2, fn):
    H,W,C=i1.shape
    if H==3: i1 = np.transpose(i1, (1,2,0))
    H,W,C=i2.shape
    if H==3: i2 = np.transpose(i2, (1,2,0))
    if i1.shape!=i2.shape: raise Exception(f'i1{i1.shape} != i2{i2.shape}')
    H,W,C=i1.shape
    y = np.zeros((H,W,C))
    i_upper = np.triu_indices(n=H, m=W)
    i_lower = np.tril_indices(n=H, m=W)
    for ch in range(C):
        y[:,:,ch][i_lower] = i1[:,:,ch][i_lower]  # i1[:,:,ch] means take image channel=ch
        y[:,:,ch][i_upper] = i2[:,:,ch][i_upper]
    return tf_save_img(y, fn)
    
def tf_save_img(y, fn):
    '''
    if the array has a shape of (height, width, 3) it 
    automatically assumes it's an RGB image and expects 
    it to have a dtype of uint8!
    '''
    print(f'tf_save_img says: {y.shape}')
           
    if not isinstance(y, np.ndarray):
        y = y.cpu().numpy()  # from tensorflow tensor
   
    if 4 == len(y.shape):
        print(f'tf_save_img expects shape CHW or HWC. got {y.shape}')
        y = np.squeeze(y)
 
    y = y. astype(np.uint8).squeeze()
    if len(y.shape)==3:
        H,W,C=y.shape
    elif len(y.shape)==2:
        H,W=y.shape
    else:
        H=y.shape
    if H==3:  # CHW to HWC  
        y = np.transpose(y, (1,2,0)) 
		
    y = Image.fromarray(y)  # y should_be_hwc for numpy:RGB to PIL
    y.save(fn)  # Mandatory CHW format


class tfg:
    @staticmethod
    def roll(x, shift:int, axis:int):
        '''
        The elements are shifted positively (towards larger indices) 
        by the offset of shift along the dimension of axis. Negative 
        shift values will shift elements in the opposite direction. 
        Elements that roll passed the last position will wrap around 
        to the first and vice versa. Multiple shifts along multiple 
        axes may be specified.
        '''
        return tf.roll(input=x, shift=shift, axis=axis)
    
    @staticmethod
    def squared_difference(a, b):
        return tf.math.squared_difference(a, b)
        
    @staticmethod
    def all(x, axis=None):
        # XXX according to axis or if None, all image
        return tf.math.reduce_all(x, axis=axis)
        
    @staticmethod
    def any(x, axis=None):
        # XXX according to axis or if None, all image
        return tf.math.reduce_any(x, axis=axis)
        
    @staticmethod
    def euclidean_norm(x, axis=None):
        # XXX according to axis or if None, all image
        return tf.math.reduce_euclidean_norm(x, axis=axis)
        
    @staticmethod
    def logsumexp(x, axis=None):
        # XXX according to axis or if None, all image
        return tf.math.reduce_logsumexp(x, axis=axis)
        
    @staticmethod
    def min(x, axis=None):
        # XXX according to axis or if None, all image
        return tf.math.reduce_min(x, axis=axis)
        
    @staticmethod
    def std(x, axis=None):
        # XXX according to axis or if None, all image
        return tf.math.reduce_std(x, axis=axis)
    
    @staticmethod
    def variance(x, axis=None):
        # XXX according to axis or if None, all image
        return tf.math.reduce_variance(x, axis=axis)
        
    @staticmethod
    def prod(x, axis=None):
        # XXX according to axis or if None, all image
        return tf.math.reduce_prod(x, axis=axis)
        
    @staticmethod
    def max(x, axis=None):
        # XXX according to axis or if None, all image
        return tf.math.reduce_max(x, axis=axis)
        
    @staticmethod
    def mean(x, axis=None):
        # mean according to axis or if None, all image
        return tf.math.reduce_mean(x, axis=axis)
        
    @staticmethod
    def sum(x, axis=None):
        # sum according to axis or if None, all image
        return tf.math.reduce_sum(x, axis=axis)
        
    @staticmethod
    def rgb2grey(x):
        C,H,W = x.shape[0], x.shape[1], x.shape[2]
        axis = 0 if C==3 else 2
        return tf.math.reduce_mean(x, axis=0)

    @staticmethod
    def subtract(a, b):
        return tf.subtract(a, b)
    
    @staticmethod
    def abs(a):
        return tf.abs(a)
        
    @staticmethod
    def blurImage(img, kernel_length:int=5, mean:float=0., sigma:float=1.):  # img is CHW
        '''
        2D image convolution with a 2D gaussian filter
        img    -> [conv2]
                  [     ]  -> [conv2] 
        kernel -> [     ]     [     ]
                              [     ] -> output
        kernel -------------> [     ]
        
        return tf image CHW layout
        '''
        # generate 1D normalized gaussian kernel
        mid = int((kernel_length-1)/2)
        kernel=[(1/(sigma*np.sqrt(2*np.pi)))*(1/(np.exp((i**2)/(2*sigma**2)))) 
            for i in range(-mid,mid+1)]  
        kernel = kernel/sum(kernel)
        return tfg.separable2DFilter(img, kernel)
        
    def separable2DFilter(img, kernel:list):  # img is CHW or HWC
        '''
        2D image convolution with a 1D filter
        '''
        if img.shape[0] != 3: # HWC -> CHW
            img = np.transpose(img, (2,0,1))
        c,h,w = img.shape
        # kernel is python list[float] and sum=1.0
        k = tf.constant(kernel, tf.float32)
        bk = tf.reshape(k, [1,len(kernel),1,1])  # put kernel_length in axis='W'
        img_r = tf.reshape(img, [c,h,w,1])  # DANGER. Use Transpose when rotating images!
        blurx = tf.nn.conv2d(img_r, bk, [1,1,1,1],'SAME')    
        bk = tf.reshape(k, [len(kernel),1,1,1])  # put kernel_length in axis='H'
        blury = tf.nn.conv2d(blurx, bk, [1,1,1,1],'SAME')
        return tf.reshape(blury, [c,h,w])  # DANGER. Use Transpose when rotating images!
    
    def get_gaussian_kernel(ksize = 3, sigma = -1.0):
        ksigma = 0.15*ksize + 0.35 if sigma <= 0 else sigma
        i, j   = np.mgrid[0:ksize,0:ksize] - (ksize-1)//2
        kernel = np.exp(-(i**2 + j**2) / (2*ksigma**2))
        return kernel / kernel.sum()


    def get_laplacian_of_gaussian_kernel(ksize = 3, sigma = -1.0):
        # ksize is kernel_size
        # sigma is sqrt(variance) of gaussian
        ksigma = 0.15*ksize + 0.35 if sigma <= 0 else sigma
        i, j   = np.mgrid[0:ksize,0:ksize] - (ksize-1)//2
        kernel = (i**2 + j**2 - 2*ksigma**2) / (ksigma**4) * np.exp(-(i**2 + j**2) / (2*ksigma**2))
        return kernel - kernel.mean()


    def kernel_prep_4d(kernel, n_channels):
        kernel = np.array(kernel, dtype=np.float32)
        return np.tile(kernel, (n_channels, 1, 1, 1)).swapaxes(0,2).swapaxes(1,3)

    def kernel_prep_3d(kernel, n_channels):
        kernel = np.array(kernel, dtype=np.float32)
        return np.tile(kernel, (n_channels, 1, 1)).swapaxes(0,1).swapaxes(1,2)
        
    def filter2d(batch, kernel, strides=(1,1), padding='SAME'):
        # batch is a tensor with 4D shape tuple/list
        batch = tfg.makeNHWC(batch)
        n_ch = batch.shape[3]
        tf_kernel = tf.constant(tfg.kernel_prep_4d(kernel, n_ch))
        return tf.nn.depthwise_conv2d(batch, tf_kernel, [1, strides[0], strides[1], 1], padding=padding)

    def get_sobel_kernel(ksize):
        if (ksize % 2 == 0) or (ksize < 1):
            raise ValueError("Kernel size must be a positive odd number")
        _base = np.arange(ksize) - ksize//2
        a = np.broadcast_to(_base, (ksize,ksize))
        b = ksize//2 - np.abs(a).T
        s = np.sign(a)
        return (a + s*b).astype(np.float32)
        
     
    def deriv(batch, ksize=3, padding='SAME'):
        batch = tfg.makeNHWC(batch)
        n_ch = batch.shape[3]
        gx = tfg.kernel_prep_3d(np.array([[ 0, 0, 0],
                                         [-1, 0, 1],
                                         [ 0, 0, 0]]), n_ch)    
        gy = tfg.kernel_prep_3d(np.array([[ 0,-1, 0],
                                         [ 0, 0, 0],
                                         [ 0, 1, 0]]), n_ch)   
        kernel = tf.constant(np.stack([gx, gy], axis=-1), name="DerivKernel", dtype=tf.float32)
        return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding, name="GradXY")
        

    def sobel(batch, ksize=3, padding='SAME'):
        batch = tfg.makeNHWC(batch)
        n_ch = batch.shape[3]
        gx = tfg.kernel_prep_3d(tfg.get_sobel_kernel(ksize),   n_ch)
        gy = tfg.kernel_prep_3d(tfg.get_sobel_kernel(ksize).T, n_ch)
        kernel = tf.constant(tf.stack([gx, gy], axis=-1), dtype=tf.float32)
        return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding)


    def makeNHWC(x):
        if 3 == len(x.shape):
            H,W,C = x.shape
            if H == 1 or H == 3:  # CHW -> HWC
                x = tf.transpose(x, [1, 2, 0])
            x = tf.expand_dims(x, 0)  # HWC -> 1HWC. like new_axis
        elif 4 == len(x.shape):
            return x
        elif 2 == x.shape:
            x = tf.expand_dims(x, 0)
            x = tf.expand_dims(x, 0)
            
        return x
    
    def sharr(batch, ksize=3, padding='SAME'):
        # sharr says: expecting shape[N,H,W,C]'
        batch = tfg.makeNHWC(batch)
        n_ch = batch.shape[3]
        gx = tfg.kernel_prep_3d([[ -3., 0.,  3.],
                                [-10, 0, 10],
                                [ -3, 0,  3]], n_ch)    
        gy = tfg.kernel_prep_3d([[-3.,-10.,-3.],
                                [ 0,  0, 0],
                                [ 3, 10, 3]], n_ch)
        kernel = tf.constant(tf.stack([gx, gy], axis=-1), dtype=tf.float32)
        return tf.nn.depthwise_conv2d(batch, kernel, [1, 1, 1, 1], padding=padding)


    def laplacian(batch, padding='SAME'):
        kernel = np.array([[0, 1, 0],
                           [1,-4, 1],
                           [0, 1, 0]], dtype=np.float32)    
        return tfg.filter2d(batch, kernel, padding=padding)

    def downsample(img, factor:int=2):
        # img is HWC
        # 2x2 Conv layers with a stride of 2 for downsampling. 
        # return is also HWC
        if 1==factor:
            return img
        if len(img.shape) == 2:
            img = tf.expand_dims(img, -1)  # Make HWC
        elif len(img.shape) == 3:
            if img.shape[2] == 1 or img.shape[2] == 3:
                # HWC. do nothing
                pass
            else:
                # CHW -> HWC
                img = tf.transpose(img, [1, 2, 0])
    
        x = tf.transpose(img, [2,0,1]) # HWC -> CHW
        C,H,W =  x.shape
        kShape = [1, 1, 1, 1]
        k = tf.Variable(tf.constant(1., shape=kShape, dtype=tf.float32))
        x = tf.reshape(x, [C,H,W,1])  # CHW -> CHW1  gonen why????
        # DANGER with tf.reshape(). Use Transpose when rotating images!
        x = tf.nn.conv2d(input=x, 
            filters=k, 
            strides=[1,factor,factor,1], 
            dilations=[factor,factor], 
            padding='VALID')
        # x is now [CHW1]
        # DANGER with tf.reshape(). Use Transpose when rotating images!
        x = tf.reshape(x, [x.shape[0], x.shape[1], x.shape[2]])  # now its [CHW]
        x = tf.transpose(x, [1,2,0])  # we need HWC for the rest of code
        return x


    def boxfilter(batch, ksize = 3, padding='SAME'):
        kernel = np.ones((ksize, ksize), dtype=np.float32) / ksize**2
        return tfg.filter2d(batch, kernel, padding=padding)
        
    def select_by_idx(a, idx):
        # tf.2.14 needs this exact syntax with named parameters
        return tf.raw_ops.SelectV2(
            condition=tf.equal(idx, 2), 
            t=a[:,:,:,2], 
            e=tf.raw_ops.SelectV2(
                condition=tf.math.reduce_all(tf.cast(idx, tf.bool), 1),  # same as np.all 
                t=a[:,:,:,1], 
                e=a[:,:,:,0]))

    def hog_descriptor(images, cell_size = 8, block_size = 2, block_stride = 1, n_bins = 9,
                          grayscale = False, oriented = False):
                          
        # images  - shape [N,H,W,C]
        if isinstance(images, np.ndarray) is False:
            raise Exception(f'"images" should be shaped [N,H,W,C]. got {images.shape}')
        if len(images.shape) != 4:
            raise Exception(f'"images" should be shaped [N,H,W,C]. got {images.shape}')
        batch_size, height, width, depth = images.shape
        half_pi = tf.constant(np.pi/2, name="pi_half")
        eps = tf.constant(1e-6, name="eps")
        scale_factor = tf.constant(np.pi * n_bins * 0.99999, name="scale_factor")
        img = tf.constant(images, name="ImgBatch", dtype=tf.float32)

        # gradients
        if grayscale:
            gray = tf.image.rgb_to_grayscale(img, name="ImgGray")
            grad = tfg.deriv(gray)
        else:
            grad = tfg.deriv(img)
            
        g_x = grad[:,:,:,0::2]
        g_y = grad[:,:,:,1::2]
        # maximum norm gradient selection
        g_norm = tf.sqrt(tf.square(g_x) + tf.square(g_y), "GradNorm")
        idx    = tf.argmax(g_norm, 3)
        g_norm = tf.expand_dims(tfg.select_by_idx(g_norm, idx), -1)
        g_x    = tf.expand_dims(tfg.select_by_idx(g_x,    idx), -1)
        g_y    = tf.expand_dims(tfg.select_by_idx(g_y,    idx), -1)

        # orientation and binning
        if oriented:
            # atan2 implementation needed 
            # lots of conditional indexing required
            raise NotImplementedError("`oriented` gradient not supported yet")
        else:
            g_dir = tf.atan(g_y / (g_x + eps)) + half_pi
            g_bin = tf.compat.v1.to_int32(g_dir / scale_factor, name="Bins")  

        # cells partitioning
        cell_norm = tf.nn.space_to_depth(g_norm, cell_size, name="GradCells")
        cell_bins = tf.nn.space_to_depth(g_bin,  cell_size, name="BinsCells")

        # cells histograms
        hist = list()
        zero = tf.zeros(cell_bins.get_shape()) 
        for i in range(n_bins):
            mask = tf.equal(cell_bins, tf.constant(i, name="%i"%i))
            hist.append(
                tf.reduce_sum(
                    tf.raw_ops.SelectV2(condition=mask, t=cell_norm, e=zero), 
                    3))
            
        hist = tf.transpose(tf.raw_ops.Pack(values=hist), [1,2,3,0], name="Hist")

        # blocks partitioning
        block_hist = tf.compat.v1.extract_image_patches(hist, 
                                              ksizes  = [1, block_size, block_size, 1], 
                                              strides = [1, block_stride, block_stride, 1], 
                                              rates   = [1, 1, 1, 1], 
                                              padding = 'VALID',
                                              name    = "BlockHist")

        # block normalization
        block_hist = tf.nn.l2_normalize(block_hist, 3, epsilon=1.0)
        
        # HOG descriptor
        hog_descriptor = tf.reshape(block_hist, 
                                    [int(block_hist.get_shape()[0]), 
                                         int(block_hist.get_shape()[1]) * \
                                         int(block_hist.get_shape()[2]) * \
                                         int(block_hist.get_shape()[3])], 
                                     name='HOGDescriptor')

        return hog_descriptor
        
img = Image.open('test.png')
img = np.array(img, dtype=np.float32)
H,W,C=img.shape
img_chw = img.transpose((2,0,1))  # HWC -> CHW

yy = tfg.hog_descriptor(img.reshape((1,H,W,C)))

dst = tfg.blurImage(img_chw, kernel_length=11, sigma=3.0)
tf_save_img(dst, 'out.png')

plot_two_images_diagonally(img, 
    tfg.blurImage(img_chw, kernel_length=11, sigma=3.0), 'out-compare-11-3.png')
plot_two_images_diagonally(img, 
    tfg.blurImage(img_chw, kernel_length=21, sigma=3.0), 'out-compare-21-3.png')

def test1(img):
	downsample_factor=8
	tf_save_img(img, f'out-img.png')
	src = tfg.downsample(img, factor=downsample_factor)  
	tf_save_img(src, f'out-src.png')
	sh = tfg.sharr(src); tf_save_img(sh[:,:,:,0:3], f'out-sharr.png')
	sh = tfg.deriv(src); tf_save_img(sh[:,:,:,0:3], f'out-deriv.png')
	sh = tfg.sobel(src); tf_save_img(sh[:,:,:,0:3], f'out-sobel.png')
	sh = tfg.laplacian(src); tf_save_img(sh, f'out-laplacian.png')
	sh = tfg.boxfilter(src); tf_save_img(sh, f'out-boxfilter.png')
	# test a composed tf graph from tfg.xyz elements...
	x1 = tfg.blurImage(src)
	x2 = tfg.blurImage(src, kernel_length=11, sigma=3.0)
	x = tfg.subtract(tfg.rgb2grey(x1), tfg.rgb2grey(x2))
	tf_save_img(x, f'out-subtract-factor-{downsample_factor}.png')
	x = tfg.abs(tfg.subtract(x1,x2))
	tf_save_img(x, f'out-subtract-factor-{downsample_factor}-color.png')

test1(img)
