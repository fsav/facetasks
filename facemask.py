import os, sys
import dlib
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.patches as patches
import matplotlib as mpl

from PIL import Image, ImageDraw, ImageChops

from scipy.spatial import ConvexHull

from filepreproc import FileDatasetPreprocessor

# download here: https://github.com/davisking/dlib-models
_FACE_LANDMARKS_MODEL = os.path.expanduser("~/data/dlib/shape_predictor_68_face_landmarks.dat")
assert os.path.exists(_FACE_LANDMARKS_MODEL)

_MIN_IMG_DIM = 64
_MIN_FACE_DIM = 40

class FaceMaskPreprocessor(object):
    """Detects a face in the image and finds facial landmarks, both using dlib.
    Uses their convex hull to create binary mask for the facial region.

    This is meant to be used with FileDatasetPreprocessor:
    https://github.com/fsav/filepreproc

    Used with the "main" provided below, it will write a CSV file with the
    bounding box of the face found, or error messages if for some reason the
    image is rejected.

    The image is rejected if:
    - multiple faces are found, or none at all
    - image is too small (see _MIN_IMG_DIM)
    - face bounding box is too small (see _MIN_FACE_DIM)
    - obvious errors happen (I/O error etc.)
    """

    def __init__(self, save_debug_image=False):
        self.save_debug_image = save_debug_image
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(_FACE_LANDMARKS_MODEL)
        self.columns = ["width","height","x_min","x_max","y_min","y_max"]

    def create_mask(self, src_path, dest_path,
                    dump_debug_image=False):
        try:
            # loads image as ndarray
            img = dlib.load_rgb_image(src_path)
        except:
            msg = "Error during image load"
            return False, format_exc(msg, sys.exc_info()), {}

        if min(img.shape[0], img.shape[1]) < _MIN_IMG_DIM:
            return False, "Image too small %s" % (str(img.shape),), {}

        # returns a list of "dlib.rectangle" objects
        # second parameter is "upsample_num_times", must be >= 0
        # apparently upsampling (resizing) improves face detection
        try:
            face_locations = self.face_detector(img, 1)
        except:
            msg = "Error during face detection"
            return False, format_exc(msg, sys.exc_info()), {}

        if face_locations is None or len(face_locations) == 0:
            return False, "No faces detected", {}
        
        if len(face_locations) > 1:
            return False, "More than one face detected", {}

        loc = face_locations[0]

        try:
            landmarks = self.landmark_predictor(img, loc)
        except:
            msg = "Error during landmark prediction"
            return False, format_exc(msg, sys.exc_info()), {}

        # Not much point checking if "landmarks" is empty... it seems to always
        # return 68 landmarks... even in a blank image. So we have to rely on
        # the face detection.
        # 68 points
        points = []
        for v in landmarks.parts():
            points.append([v.x, v.y])

        assert len(points) == 68

        points = np.asarray(points)

        hull = ConvexHull(points)

        polygon = []

        for v in hull.vertices:
            polygon.append(tuple(points[v]))

        # 'L' for grayscale
        # 0 = black background
        mask = Image.new('L', (img.shape[1], img.shape[0]), 0)

        ImageDraw.Draw(mask).polygon(polygon,outline=255,fill=255)

        # bounding box of convex hull
        x = points[:,0]
        x_min = x.min()
        x_max = x.max()

        y = points[:,1]
        y_min = y.min()
        y_max = y.max()

        bb_width = x_max - x_min
        bb_height = y_max - y_min

        if min(bb_width, bb_height) < _MIN_FACE_DIM:
            return False, "Face too small %s, %s" % (bb_width, bb_height), {}

        mask.save(dest_path)

        self.write_debug_image(img, mask, dest_path)
        
        return True, "", self.create_metadata(img.shape[1], img.shape[0],
                                              x_min, x_max, y_min, y_max)

    def write_debug_image(self, img, mask, dest_path):
        if self.save_debug_image:
            ndmask = np.array(mask, dtype='float32') / 255.0
            ndmask = ndmask.reshape((img.shape[0], img.shape[1], 1))
            # For some reason the image needs to be inverted before
            # imsave... I didn't look into this too much.
            debugimg = (-img) * ndmask
            mpl.image.imsave(dest_path+".debug.jpg", debugimg)

    def create_metadata(self, width=None, height=None, 
                        x_min=None, x_max=None, y_min=None, y_max=None):
        return {"width": width,
                "height": height,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max}

def format_exc(message, exc_info):
    # [0] is the class e.g. NameError
    # [1] is the message e.g. "integer division or modulo by zero"
    msg = message + "|" + exc_info[0] + "|" + exc_info[1]

if __name__ == '__main__':
    preproc = FaceMaskPreprocessor(save_debug_image=False)
    src_dir = os.path.expanduser("~/data/imdbwiki/imdb_crop")
    dest_dir = os.path.expanduser("~/data/imdbwiki/imdb_masks")
    proc = FileDatasetPreprocessor(src_dir=src_dir,
                                   dest_dir=dest_dir,
                                   preprocess_fn=preproc.create_mask,
                                   input_extension="jpg",
                                   metadata_filename="preprocessed.csv",
                                   metadata_columns=preproc.columns,
                                   num_processes=4)
    proc.run()

