import numpy as np
from pycocotools.coco import COCO
from PIL import Image
from matplotlib.pyplot import imread
# import scipy.misc as misc
import os.path


def load_image(image_id, mode):

	try:
		if mode == 'training':
			# loaded_image = imread('./train2014/COCO_train2014_'+str((image_id)).zfill(12)+'.jpg'), mode = 'RGB')
			loaded_image = imread('./train2014/COCO_train2014_'+str((image_id)).zfill(12)+'.jpg')

		elif mode == 'testing':
			# loaded_image = imread('./val2014/COCO_val2014_'+str((image_id)).zfill(12)+'.jpg', mode = 'RGB')
			loaded_image = imread('./val2014/COCO_val2014_'+str((image_id)).zfill(12)+'.jpg')

		else:
			raise ValueError('MS COCO Image Id {} Not Found'.format(image_id))
	except:
		print('cannot read image')
		# exit(-1)
		
	return loaded_image

def apply_mask(coco, raw_image, mask, keep, crop):

	''' Takes in an MSCOCO image and keeps either the context or object.
		coco: MSCOCO instance.
		raw_image: MSCOCO image as (height, width, channel)
		mask: Mask for the object within the image.
		keep: Tells function what part of image to keep. Given as a string, either 'context' or 'object'.
	'''

	# Turn segmentation mask into a bounding box


	n = np.shape(raw_image)[0]
	m = np.shape(raw_image)[1]

	x_crop = int(((np.max(np.where(mask)[1]) - np.min(np.where(mask)[1])) / 2.) * crop)
	y_crop = int(((np.max(np.where(mask)[0]) - np.min(np.where(mask)[0])) / 2.) * crop)

	x_min = np.clip(np.min(np.where(mask)[1]) - x_crop, 0, m)
	x_max = np.clip(np.max(np.where(mask)[1]) + x_crop, 0, m)
	y_min = np.clip(np.min(np.where(mask)[0]) - y_crop, 0, n)
	y_max = np.clip(np.max(np.where(mask)[0]) + y_crop, 0, n)

	if x_min == x_max or y_min == y_max: raise ValueError('Mask is too small.')

	if keep == 'object':
		return raw_image[y_min:y_max, x_min:x_max,...]
	elif keep == 'context':
		masked_image = raw_image.copy()
		masked_image[y_min:y_max, x_min:x_max,...] = 0
		return masked_image
	else:
		raise ValueError("Parameter 'keep' needs to be either object or context")


def fit_in_square(im, img_dim):

	''' Constrain the image to a 224x224x3 image and center the object or context.
		input_image: masked_image containing either only the object or context.
		img_dim: (height, width, channels)
	'''
	try:
		im2 = Image.fromarray(im, 'RGB')
	except:
		print(np.shape(im))
		raise ValueError('Cant turn into Image')
	im2.thumbnail((img_dim, img_dim), resample = Image.BILINEAR)
	# im3 = np.transpose(im, (2, 1, 0))

	im3 = np.array(im2)
	# im3 = np.rollaxis(im3, 2, 0)
	frame = np.zeros((img_dim, img_dim, 3))
	m,n,p = np.shape(im3)
	d = img_dim
	try:
		frame[d//2-m//2:d//2+m//2+m%2, d//2-n//2:d//2+n//2+n%2, :] = im3
	except Except as e:
		print('error', e)
		# pass

	return frame

def prepare_input(input_img):

	''' Prepare the image for the right dimensions for training
		input_img: image to be prepared.
	'''

	output_img = input_img

	output_img = np.expand_dims(output_img, axis=0)

	output_img[0:4,:] = output_img[0:4,:] - 127.5

	# output_img_t = np.transpose(output_img, (0, 2,3,1))
	return output_img