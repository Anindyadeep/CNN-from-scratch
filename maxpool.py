import numpy as np

'''
Here we there are some shortcomings which are needed to be done
1. make the maxpool genaralised for most of the images
2. try to takr all the image portion in the im_region as in case of the odd image size some parts are not considered
   during the iteration process
3. also try to apply padding
4. try to apply strides
5. Try to do all the operations with the filters with different dimensioanlity i.e. instead of 
   taking the dimensionality of (h,w,num_filters), try to do with these (num_filters, h, w)
   This will make the image/feature_maps not to loose the actual dimensionality

Make all these changes once all the CNN part is being made
'''
class MaxPool2D:
    def regions_to_iterate(self, single_input_image):

        # store the conved image input in the cache, so that we can use it for later
        self.last_input = single_input_image

        h,w,_ = single_input_image.shape
        new_h, new_w = h//2, w//2
        for i in range(new_h):
            for j in range(new_w):
                im_region = single_input_image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield im_region, i, j

    def maxpool_forward(self, single_input_image):
        h,w,num_filters = single_input_image.shape
        h_new, w_new = h//2, w//2
        output_max_pool_2d = np.zeros(shape=(h_new, w_new, num_filters))

        for im_region, i, j in self.regions_to_iterate(single_input_image):
            output_max_pool_2d[i,j] = np.amax(im_region, axis = (0,1), keepdims = True)
        return output_max_pool_2d

    def maxpool_backward(self, dL_dOut):
        dL_dinput = np.zeros(self.last_input.shape)
        for im_region, i, j in self.regions_to_iterate(self.last_input):
            h,w,num_filters = im_region.shape
            amax = np.amax(im_region, axis = (0,1))

            # analyse this part for later to read carefully (how this has took place for f2)
            '''
            Honestly, I did't understand this part, but I will do it
            very soon, as have to make my own geneuine CNN 
            '''
            '''
            Though got the intuition, that from the dL/dOut, we are actuall copying each pixel of
            dL/dOut and making it to the dL/dX, such that it will return the conv image kind of
            '''
            for i2 in range(h):
                for j2 in range(w):
                    for k2 in range(num_filters):
                        if im_region[i2,j2,k2] == amax[k2]:
                            dL_dinput[i*2+i2, j*2+j2, k2] = dL_dOut[i,j, k2]
        return dL_dinput

