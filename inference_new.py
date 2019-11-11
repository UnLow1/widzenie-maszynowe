#
import os

import flask
from PIL import Image
from flask import send_file

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

from load_data import loadData
from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff
from keras.optimizers import RMSprop

import numpy as np
import pandas as pd

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

from skimage import morphology, color, io, exposure

from skimage import img_as_ubyte

import os.path  # from os.path import exists
import shutil  # shutil.copy2

import cv2
import cv2 as cv

import glob
import argparse

import sys

from utils_dl import mean_Iou_Dice

data_bow_legs_dir = 'data_bow-legs'
dataset_bow_legs_dir = 'dataset_bow-legs'

import keras.backend as K

import io as in_out


def binary_crossentropy_custom(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


# ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ 

def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)


def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)


def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, 
	predicted lung field filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))

    min_val = gt.min()
    max_val = gt.max()
    # print('min_val = {} max_val = {}\n'.format(min_val, max_val))

    # boundary = morphology.dilation(gt, morphology.disk(3)) - gt
    boundary = morphology.dilation(gt, morphology.disk(3)) ^ gt
    # boundary = morphology.dilation(gt, morphology.disk(1)) ^ gt

    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


# ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^ ~.~ ^.^

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



import werkzeug
app = flask.Flask(__name__)


# if __name__ == '__main__':

@app.route('/', methods=['POST'])
def handle_request():
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True

    imagefile = flask.request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save("result" + filename)
    print('\n')

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fname", type=str, required=False, default='models/trained_model.hdf5',
                        help="full path to model")
    # parser.add_argument( "-i", "--integer", type=int, default=50)
    parser.add_argument('-d', '--disp_test_images', type=str2bool, default=False,
                        help="whether to disp test image names. default: False")
    parser.add_argument('-s', '--save_out_images', type=str2bool, default=True,
                        help="whether to ave out images. default: True")

    args = parser.parse_args()

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.

    csv_path = dataset_bow_legs_dir + '/' + 'idx_test.csv'

    # Path to the folder with images. Images will be read from path + path_from_csv
    path = csv_path[:csv_path.rfind('/')] + '/'

    df = pd.read_csv(csv_path)
    if args.disp_test_images == True:
        print('\n')
        print(df)
        print('\n')

    # Load test data
    im_shape = (512, 256)
    X, y = loadData(df, path, im_shape, imagefile)

    print('\n[*]loadData() finished\n')

    n_test = X.shape[0]
    inp_shape = X[0].shape

    # Load model
    if ".hdf5" not in args.fname:
        list_of_files = glob.glob(args.fname + '/' + '*.hdf5')  #
        # print(list_of_files)
        model_weights = list_of_files[0]
    else:
        model_weights = args.fname

    # load json and create model
    json_file = open('models/model_bk.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    print("model_from_json() finished ...")

    # load weights into new model
    loaded_model.load_weights(model_weights)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    UNet = loaded_model
    model = loaded_model
    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])
    print("model compiled ")

    # For inference standard keras ImageGenerator is used.
    test_gen = ImageDataGenerator(rescale=1.)

    ious = np.zeros(n_test)
    dices = np.zeros(n_test)
    mean_IoU = np.zeros(n_test)

    i = 0
    '''	
    for xx, yy in test_gen.flow(X, y, batch_size=1):
    '''
    num_imgs = X.shape[0]
    for ii in range(num_imgs):
        xx_ = X[ii, :, :, :]
        yy_ = y[ii, :, :, :]
        xx = xx_[None, ...]
        yy = yy_[None, ...]
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0, 1))
        pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
        mask = yy[..., 0].reshape(inp_shape[:2])

        # Binarize masks
        gt = mask > 0.5
        pr = pred > 0.5

        pr_bin = img_as_ubyte(pr)
        pr_openned = morphology.opening(pr_bin)

        # Remove regions smaller than 2% of the image
        pr = remove_small_regions(pr,
                                  0.005 * np.prod(im_shape))  # pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
        pr_out = img_as_ubyte(pr)

        sub_dir_file_name = df.iloc[i][0]
        file_name = sub_dir_file_name[9:]
        sub_dir_name = sub_dir_file_name[:8]
        if args.disp_test_images == True:
            print('\n')
            print('sub_dir_name={}  file_name={}\n\n'.format(sub_dir_name, file_name))

        if args.save_out_images == True:
            dir_img_mask = 'results/bow-legs_test/{}'.format(sub_dir_name)
            if not os.path.exists(dir_img_mask):
                os.makedirs(dir_img_mask)
            img_name = '{}/{}'.format(dir_img_mask, file_name)
            if args.disp_test_images == True:
                print('img_name={}\n'.format(img_name))

            cv2.imwrite(img_name, pr_openned)

        file_name_no_ext = os.path.splitext(file_name)[
            0]  # ('file', '.ext')  --> os.path.splitext(file_name_no_ext)[0] ('file')
        file_name_in = dataset_bow_legs_dir + '/' + sub_dir_name + '/' + file_name_no_ext + '_mask' + '.png'  # dataset_bow-legs/mask_001/img_0001_mask.png
        if args.disp_test_images == True:
            print('file_name_in={}\n'.format(file_name_in))
        if args.save_out_images == True:

            file_name_out = 'results/bow-legs_test' + '/' + sub_dir_name + '/' + file_name_no_ext + '_mask_manual' + '.png'  # results/bow-legs_test/mask_006/img_0006_mask_manual.png

            img_exists = os.path.isfile(file_name_in)
            if img_exists == False:
                print('{} does not exists\n'.format(file_name_in))
                sys.exit("exiting ...")

            shutil.copy2(file_name_in, file_name_out)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        im_name_x_ray_original_size = data_bow_legs_dir + '/' + 'x-ray/' + file_name  # data_bow-legs/x-ray/img_0001.png
        im_name_x_ray_original_size_test = data_bow_legs_dir + '/' + 'x-ray_test/' + file_name  # data_bow-legs/x-ray/img_0001.png
        if args.disp_test_images == True:
            print('im_name_x_ray_original_size = {}\n'.format(im_name_x_ray_original_size_test))

        im_x_ray_original_size = cv2.imread(im_name_x_ray_original_size_test, cv2.IMREAD_GRAYSCALE)
        if im_x_ray_original_size is None:  ## Check for invalid input
            print("Could not open or find the image: {}. ".format(im_name_x_ray_original_size_test))
            shutil.copy2(im_name_x_ray_original_size, im_name_x_ray_original_size_test)
            print('Made a copy from {}\n'.format(im_name_x_ray_original_size))
            im_x_ray_original_size = cv2.imread(im_name_x_ray_original_size_test, cv2.IMREAD_GRAYSCALE)

        height, width = im_x_ray_original_size.shape[:2]  # height, width  -- original image size

        ratio = float(height) / width

        new_shape = (4 * 256, int(4 * 256 * ratio))

        im_x_ray_4x = cv2.resize(im_x_ray_original_size, new_shape)

        dir_img_x_ray_4x = 'results/bow-legs_test_4x/{}'.format(sub_dir_name)
        if not os.path.exists(dir_img_x_ray_4x):
            os.makedirs(dir_img_x_ray_4x)
        im_name_x_ray_4x = '{}/{}'.format(dir_img_x_ray_4x, file_name)
        cv2.imwrite(im_name_x_ray_4x, im_x_ray_4x)

        # mask	
        im_mask_original_size = cv2.imread(file_name_in, cv2.IMREAD_GRAYSCALE)
        im_mask_4x = cv2.resize(im_mask_original_size, new_shape)
        im_name_mask_4x = '{}/{}'.format(dir_img_x_ray_4x, '/' + file_name_no_ext + '_mask_manual' + '.png')
        cv2.imwrite(im_name_mask_4x, im_mask_4x)

        # Unet output
        pr_openned_4x = cv2.resize(pr_openned, new_shape)
        im_name_pr_openned_4x = '{}/{}'.format(dir_img_x_ray_4x, file_name_no_ext + '_mask_Unet' + '.png')
        if args.disp_test_images == True:
            print('im_name_pr_openned_4x={}\n'.format(im_name_pr_openned_4x))
        cv2.imwrite(im_name_pr_openned_4x, pr_openned_4x)

        gt_4x = cv2.resize(img_as_ubyte(gt), new_shape)

        gt_4x = gt_4x > 0.5
        pr_openned_4x = pr_openned_4x > 0.5
        im_x_ray_4x_ = im_x_ray_4x / 255.0
        if args.disp_test_images == True:
            print('img.max()={} gt.max()={} pr.max()={}\n'.format(im_x_ray_4x_.max(), gt_4x.max(), pr_openned_4x.max()))
        im_masked_4x = masked(im_x_ray_4x, gt_4x, pr_openned_4x, 0.5)  # img.max()=1.0 gt.max()=True pr.max()=True

        if args.save_out_images == True:
            dir_im_masked_4x = 'results/bow-legs_masked_4x'
            if not os.path.exists(dir_im_masked_4x):
                os.makedirs(dir_im_masked_4x)
            im_name_masked_4x = '{}/{}'.format(dir_im_masked_4x, file_name)

            im_masked_4x = img_as_ubyte(im_masked_4x)
            io.imsave(im_name_masked_4x, im_masked_4x)

            # convert numpy array to PIL Image
            img = Image.fromarray(im_masked_4x.astype('uint8'))
            # create file-object in memory
            file_object = in_out.BytesIO()
            # write PNG in file-object
            img.save(file_object, 'PNG')
            # move to beginning of file so `send_file()` it will read from start
            file_object.seek(0)

            return send_file(file_object, mimetype='image/PNG')

        ious[i] = IoU(gt, pr)
        dices[i] = Dice(gt, pr)
        print('{}  {:.4f} {:.4f}'.format(df.iloc[i][0], ious[i], dices[i]))

        with open("results/bow-legs_results.txt", "a", newline="\r\n") as f:
            print('{}  {:.4f} {:.4f}'.format(df.iloc[i][0], ious[i], dices[i]), file=f)

        i += 1
        if i == n_test:
            break

    print('Mean IoU:{:.4f} Mean Dice:{:.4f}'.format(ious.mean(), dices.mean()))
    with open("results/bow-legs_results.txt", "a", newline="\r\n") as f:
        print('Mean IoU:{:.4f} Mean Dice:{:.4f}'.format(ious.mean(), dices.mean()), file=f)
        print('\n', file=f)

    with open("results/bow-legs_IoU_Dice.txt", "a", newline="\r\n") as f:
        print('Mean IoU:{:.4f} Mean Dice:{:.4f}'.format(ious.mean(), dices.mean()), file=f)

    return "DONE"

print('Press any key to exit ')
cv2.waitKey(0)
cv2.destroyAllWindows()

app.run(host="localhost", port=8080, debug=True, threaded=False)
