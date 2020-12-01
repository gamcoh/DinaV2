import pandas as pd
import numpy as np

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }

class DataLoader():
    """ Class used to load csvs
    # Arguments
        path_vid    : path to the root folder containing the videos
        path_labels : path to the csv containing the labels
        path_train  : path to the csv containing a list of the videos used for the training
        path_val    : path to the csv containing a list of the videos used for the validation
        path_test   : path to the csv containing a list of the videos used for the test
    #Returns
        An instance of the DataLoader class  
    """
    def __init__(self, path_vid, path_labels, path_train=None, path_val=None, path_test=None):
        self.path_vid    = path_vid
        self.path_labels = path_labels
        self.path_train  = path_train
        self.path_val    = path_val
        self.path_test   = path_test

        self.get_labels(path_labels)

        if self.path_train:
            self.train_df = self.load_video_labels(self.path_train)

        if self.path_val:
            self.val_df = self.load_video_labels(self.path_val)

        if self.path_test:
            self.test_df = self.load_video_labels(self.path_test, mode="input")

    def get_labels(self, path_labels):
        """Loads the Dataframe labels from a csv and creates dictionnaries to convert the string labels to int and backwards
        # Arguments
            path_labels : path to the csv containing the labels
        """
        self.labels_df = pd.read_csv(path_labels, names=['label'])
        #Extracting list of labels from the dataframe
        self.labels = [str(label[0]) for label in self.labels_df.values]
        self.n_labels = len(self.labels)
        #Create dictionnaries to convert label to int and backwards
        self.label_to_int = dict(zip(self.labels, range(self.n_labels)))
        self.int_to_label = dict(enumerate(self.labels))

    def load_video_labels(self, path_subset, mode="label"):
        """ Loads a Dataframe from a csv
        # Arguments
            path_subset : String, path to the csv to load
            mode        : String, (default: label), if mode is set to "label", filters rows given if the labels exists in the labels Dataframe loaded previously
        #Returns
            A DataFrame
        """
        names=None
        if mode=="input":
            names=['video_id']
        elif mode=="label":
            names=['video_id', 'label']
        
        df = pd.read_csv(path_subset, sep=';', names=names) 
        
        if mode == "label":
            df = df[df.label.isin(self.labels)]

        return df
    
    def categorical_to_label(self, vector):
        """ Used to convert a vector to the associated string label
        # Arguments
            vector : Vector representing the label of a video
        #Returns
            Returns a String that is the label of a video
        """
        return self.int_to_label[np.where(vector==1)[0][0]]
 
class frame_queue(object):
    """
    push current img to a queue
    img: img input for the moment
    """
    def __init__(self, nb_frames, target_size):
        self.batch = np.zeros((1, nb_frames) + target_size + (3,))
        self.target_size = target_size
    def img_resiz(self,img, interpolation = 'nearest'):

        img = pil_image.fromarray(img) # change image format from nparray to jpg
        if self.target_size is not None:
            width_height_tuple = (self.target_size[1], self.target_size[0])
            if img.size != width_height_tuple:
                if interpolation not in _PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            interpolation,
                            ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
                resample = _PIL_INTERPOLATION_METHODS[interpolation]
                img = img.resize(width_height_tuple, resample)
        return img


    def img_to_array(self,img):
        img_array = np.asarray(img, dtype = 'float32')
        return img_array

    def img_inQueue(self, img):
        for i in range(self.batch.shape[1] - 1):
            self.batch[0, i] = self.batch[0, i+1]
        img = self.img_resiz(img)
        img = self.img_to_array(img)
        x = self.img_to_array(img)
        self.batch[0, self.batch.shape[1] - 1] = x / 255

        return self.batch