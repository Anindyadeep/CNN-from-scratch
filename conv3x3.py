import numpy as np

class Conv3x3:
    '''
    This initialisation of the class will be with the number of the filters
    in the network used for any layers

    The image we will be using must be non RGB for now
    Also the filters used here will always have to get a shape of (3,3)
    '''
    def __init__ (self, num_filters):
        self.num_filters = num_filters
        self.filters =  np.random.randn(num_filters, 3,3)/9 # xavier initialization
    
    def regions_to_iterate(self, single_input_image):
        h, w = single_input_image.shape
        for i in range(h-2):
            for j in range(w-2):
                im_regions = single_input_image[i:(i+3), j:(j+3)]
                yield im_regions,i,j
    
    def single_conv_forward(self, single_input_image):
        filters = self.filters
        h, w = single_input_image.shape
        # caching the main input image out of here for further needs

        self.last_input = single_input_image

        output_image = np.zeros(shape=(h-2,w-2, self.num_filters))
        for im_regions, i, j in self.regions_to_iterate(single_input_image):
            output_image[i, j] = np.sum(im_regions * filters, axis = (1,2))
        return output_image

    def single_conv_backward(self, dL_dOut, learning_rate):
        dL_dfilters = np.zeros(self.filters.shape)

        '''
        we know that can collect the (i,j) regions of the convolved image, and that part of the
        main input image from [i, i+3) and [j, j+3) for all the f filters.
        Using that we will compute the gradient of filters w.r.t the loss such that:

        dL/dfilters = dl/dOut * dOut/dfilters
        where :
              dL/dOut = the output from the maxpool and input for here
              dOut/dfilters = im_region of that {[i, j+3), f} where f is the filter number

              NOTE: PLS MAKE THE CONVEPT MORE CRYSTAL CLEAR SO THAT U CAN MAKE IT 
                    FOR YOUR MODEL, AS THE MATHEMATICAL THEORY REMAINS THE SAME 
        '''
        for im_region, i, j in self.regions_to_iterate(self.last_input):
            for f in range(self.num_filters):
                dL_dfilters += (dL_dOut[i, j, f] * im_region)
        
        # updating the filters
        self.filters -= learning_rate * dL_dfilters
        return None