from pycocotools.coco import COCO

from generator import create_generator
from models import build_model, load_model

from keras.callbacks import ModelCheckpoint

# from keras_diagram import ascii
from keras.optimizers import Adam

def train(add, num_testing, object_dim, job_name, **kwargs):

	coco_train = COCO('./annotations/instances_train2014.json')
	coco_test = COCO('./annotations/instances_val2014.json')

	training_generator = create_generator(coco = coco_train, mode = 'training', add = add, object_dim = object_dim, **kwargs)
	testing_generator = create_generator(coco = coco_test, mode = 'testing', add = add, object_dim = object_dim, **kwargs)

	model = build_model(add = add, object_dim = object_dim, **kwargs)

	model.compile(loss = 'categorical_crossentropy', optimizer = Adam(1e-6, beta_1=.9, beta_2=.99), metrics = ['accuracy'])

	# callbacks_list = [ModelCheckpoint('./'+job_name+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')]

	callbacks_list = [ModelCheckpoint('./'+job_name+'.h5', monitor='val_loss', verbose=1, mode='min')]

	print(model.summary())

	history = model.fit_generator(
		training_generator, \
		validation_steps = num_testing, \
		validation_data = training_generator, \
		steps_per_epoch = 100, # 5000, \
		epochs = 5, # 500, \
		# callbacks = callbacks_list,\
		verbose=1, \
		max_queue_size = 2, # 10, \
		workers = 1, \
		)

if __name__ == '__main__':

	add = None
	add = 'gist'

	num_testing = 100 # 1000

	object_dim = 224

	context_dim = 448

	job_name = 'JOB_NAME'

	train(add, num_testing, object_dim, job_name, context_dim = context_dim, testing_size = num_testing)





