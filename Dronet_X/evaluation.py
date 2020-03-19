import gflags
import numpy as np
import os
import sys
import glob
from random import randint
from sklearn import metrics

from keras import backend as K

import utils
import utilsY
from constants import TEST_PHASE
from common_flags import FLAGS


# Functions to evaluate steering prediction

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    #print ypred,y
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary


def compute_explained_variance(predictions, real_values):
    """
    Computes the explained variance of prediction for each
    steering and the average of them
    """
    #print predictions.shape,real_values.shape

    assert np.all(predictions.shape == real_values.shape)
    ex_variance = explained_variance_1d(predictions, real_values)
    print("EVA = {}".format(ex_variance))
    return ex_variance


def compute_sq_residuals(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    sr = np.mean(sq_res, axis = -1)
    print("MSE = {}".format(sr))
    return sq_res


def compute_rmse(predictions, real_values,f1):



    for i in range (0,len(predictions)):
	#print predictions[i]
    	f1.write(str(real_values[i])+' '+str(predictions[i])+'\n')
	#print i
    

    assert np.all(predictions.shape == real_values.shape)
    #print predictions, real_values
    mse = np.mean(np.square(predictions - real_values))
    rmse = np.sqrt(mse)
    print("RMSE = {}".format(rmse))
    return rmse


def compute_highest_regression_errors(predictions, real_values, n_errors=20):
    """
    Compute the indexes with highest error
    """
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    highest_errors = sq_res.argsort()[-n_errors:][::-1]
    return highest_errors


def random_regression_baseline(real_values):
    mean = np.mean(real_values)
    std = np.std(real_values)
    return np.random.normal(loc=mean, scale=abs(std), size=real_values.shape)


def constant_baseline(real_values):
    mean = np.mean(real_values)
    return mean * np.ones_like(real_values)


def evaluate_regression(predictions, real_values, fname,f1):

    
    evas = compute_explained_variance(predictions, real_values)
    rmse = compute_rmse(predictions, real_values,f1)
    highest_errors = compute_highest_regression_errors(predictions, real_values,
            n_errors=20)
    dictionary = {"evas": evas.tolist(), "rmse": rmse.tolist(),
                  "highest_errors": highest_errors.tolist()}
    #print dictionary
    utils.write_to_file(dictionary, fname)


# Functions to evaluate collision

def read_training_labels(file_name):
    labels = []
    try:
        labels = np.loadtxt(file_name, usecols=0)
        labels = np.array(labels)
    except:
        print("File {} failed loading labels".format(file_name)) 
    return labels


def count_samples_per_class(train_dir):
    experiments = glob.glob(train_dir + "/*")
    num_class0 = 0
    num_class1 = 0
    for exp in experiments:
        file_name = os.path.join(exp, "labels.txt")
        try:
            labels = np.loadtxt(file_name, usecols=0)
            num_class1 += np.sum(labels == 1)
            num_class0 += np.sum(labels == 0)
        except:
            print("File {} failed loading labels".format(file_name)) 
            continue
    return np.array([num_class0, num_class1])


def random_classification_baseline(real_values):
    """
    Randomly assigns half of the labels to class 0, and the other half to class 1
    """
    return [randint(0,1) for p in range(real_values.shape[0])]


def weighted_baseline(real_values, samples_per_class):
    """
    Let x be the fraction of instances labeled as 0, and (1-x) the fraction of
    instances labeled as 1, a weighted classifier randomly assigns x% of the
    labels to class 0, and the remaining (1-x)% to class 1.
    """
    weights = samples_per_class/np.sum(samples_per_class)
    return np.random.choice(2, real_values.shape[0], p=weights)


def majority_class_baseline(real_values, samples_per_class):
    """
    Classify all test data as the most common label
    """
    major_class = np.argmax(samples_per_class)
    return [major_class for i in real_values]

            
def compute_highest_classification_errors(predictions, real_values, n_errors=20):
    """
    Compute the indexes with highest error
    """
    assert np.all(predictions.shape == real_values.shape)
    dist = abs(predictions - real_values)
    highest_errors = dist.argsort()[-n_errors:][::-1]
    return highest_errors


def evaluate_classification(pred_prob, pred_labels, real_labels, fname):
    ave_accuracy = metrics.accuracy_score(real_labels, pred_labels)
    print('Average accuracy = ', ave_accuracy)
    precision = metrics.precision_score(real_labels, pred_labels)
    print('Precision = ', precision)
    recall = metrics.precision_score(real_labels, pred_labels)
    print('Recall = ', recall)
    f_score = metrics.f1_score(real_labels, pred_labels)
    print('F1-score = ', f_score)
    highest_errors = compute_highest_classification_errors(pred_prob, real_labels,
            n_errors=20)
    #print ave_accuracy
    #print precision
    #print 'HOLA '

    dictionary = {"ave_accuracy": ave_accuracy.tolist(), "precision": precision.tolist(),
                  "recall": recall.tolist(), "f_score": f_score.tolist(),
                  "highest_errors": highest_errors.tolist()}
    #print dictionary
    utils.write_to_file(dictionary, fname)




def _main():

    # Set testing mode (dropout/batchnormalization)
    K.set_learning_phase(TEST_PHASE)

    # Generate testing data
    test_datagen_X = utils.DroneDataGenerator(rescale=1./255)
    test_generator_X = test_datagen_X.flow_from_directory(FLAGS.test_dir,
                          shuffle=False,
                          color_mode=FLAGS.img_mode,
                          target_size=(FLAGS.img_width, FLAGS.img_height),
                          crop_size=(FLAGS.crop_img_height, FLAGS.crop_img_width),
                          batch_size = FLAGS.batch_size)

    test_datagen_Y = utilsY.DroneDataGenerator(rescale=1./255)
    test_generator_Y = test_datagen_Y.flow_from_directory(FLAGS.test_dir,
                          shuffle=False,
                          color_mode=FLAGS.img_mode,
                          target_size=(FLAGS.img_width, FLAGS.img_height),
                          crop_size=(FLAGS.crop_img_height, FLAGS.crop_img_width),
                          batch_size = FLAGS.batch_size)

    # Load json and create model
    json_model_path_X = os.path.join('/home/tev/Desktop/VisionNET_X/model/model_struct_X.json')
    json_model_path_Y = os.path.join('/home/tev/Desktop/VisionNET_X/model/model_struct_Y.json')

    model1 = utils.jsonToModel(json_model_path_X)
    model2 = utils.jsonToModel(json_model_path_Y)

    # Load weights
    weights_load_path_X = os.path.join('/home/tev/Desktop/VisionNET_X/model/weights_X.h5')
    weights_load_path_Y = os.path.join('/home/tev/Desktop/VisionNET_X/model/weights_Y.h5')

    try:
        model1.load_weights(weights_load_path_X)
	model2.load_weights(weights_load_path_Y)
        print("Loaded model from {}".format(weights_load_path_X))
	print("Loaded model from {}".format(weights_load_path_Y))
    except:
        print("Impossible to find weight path. Returning untrained model")

    print '1-'
    # Compile model
    model1.compile(loss='mse', optimizer='adam')
    model2.compile(loss='mse', optimizer='adam')

    # Get predictions and ground truth
    n_samples_X = test_generator_X.samples
    nb_batches_X = int(np.ceil(n_samples_X / FLAGS.batch_size))

    n_samples_Y = test_generator_Y.samples
    nb_batches_Y = int(np.ceil(n_samples_Y / FLAGS.batch_size))

    #print nb_batches_X, nb_batches_Y

    predictions_Y, ground_truth_Y, t = utils.compute_predictions_and_gt(
            model2, test_generator_Y, nb_batches_Y, verbose = 1)

    predictions_X, ground_truth_X, t = utils.compute_predictions_and_gt(
            model1, test_generator_X, nb_batches_X, verbose = 1)


    #print ground_truth
    # Param t. t=1 steering, t=0 collision
    t_mask = 1


    # ************************* Steering evaluation ***************************
    print '###############################      X     ##################################'
    # Predicted and real steerings
    f1=open('RMSE_x.txt',"w")
    f1.write('real_values predictions'+'\n')
    #print predictions_X.shape()
    pred_steeringsX = predictions_X #[t_mask]
    real_steeringsX = ground_truth_X #[t_mask]
    print len(pred_steeringsX)   
    print len(real_steeringsX) 

    # Compute random and constant baselines for steerings
    random_steeringsX = random_regression_baseline(real_steeringsX)
    constant_steeringsX = constant_baseline(real_steeringsX)

    #print random_steeringsX
    #print constant_steeringsX





    # Create dictionary with filenames
    dict_fname = {'test_X.json': pred_steeringsX,
                  'random_X.json': random_steeringsX,
                  'constant_X.json': constant_steeringsX}

    #print dict_fname.items()
    # Evaluate predictions: EVA, residuals, and highest errors

    #evaluate_regression(dict_fname[1], real_steeringsX, abs_fname[1])

    for fname, pred in dict_fname.items():
        abs_fname = os.path.join(FLAGS.experiment_rootdir, fname)
	#print abs_fname
	#print pred
	if fname == './model/test_X.json' :
        	evaluate_regression(pred, real_steeringsX, abs_fname,f1)
	else:
		pred = np.hstack(pred)
		a=pred.shape
		pred=pred.reshape((a[0],))
		real_steeringsX=real_steeringsX.reshape((a[0],))
		evaluate_regression(pred, real_steeringsX, abs_fname,f1)
		
    # Write predicted and real steerings

    dict_test = {'pred_X': pred_steeringsX.tolist(),
                 'real_X': real_steeringsX.tolist()}
    utils.write_to_file(dict_test,os.path.join(FLAGS.experiment_rootdir,
                                               'predicted_and_real_X.json'))


    f1.close()

    # *********************** Collision evaluation ****************************
    print '###############################      Y     ##################################'
    # Predicted probabilities and real labels
    #pred_prob = predictions[~t_mask,1]
    #pred_labels = np.zeros_like(pred_prob)
    f1=open('RMSE_y.txt',"w")
    f1.write('real_values predictions'+'\n')

    pred_steeringsY = predictions_Y #[t_mask]
    real_steeringsY = ground_truth_Y #[t_mask]
    print len(pred_steeringsY)   
    print len(real_steeringsY)        
    # Compute random and constant baselines for steerings

    random_steeringsY = random_regression_baseline(real_steeringsY)
    constant_steeringsY = constant_baseline(real_steeringsY)



    # Create dictionary with filenames
    dict_fname = {'test_Y.json': pred_steeringsY,
                  'random_Y.json': random_steeringsY,
                  'constant_Y.json': constant_steeringsY}


    # Evaluate predictions: EVA, residuals, and highest errors
    for fname, pred in dict_fname.items():

        abs_fname = os.path.join(FLAGS.experiment_rootdir, fname)
        pred = np.hstack(pred)
	a=pred.shape
	pred=pred.reshape((a[0],))
	real_steeringsY=real_steeringsY.reshape((a[0],))
	evaluate_regression(pred, real_steeringsY, abs_fname,f1)



    # Write predicted and real steerings
    dict_test = {'pred_Y': pred_steeringsY.tolist(),
                 'real_Y': real_steeringsY.tolist()}
    utils.write_to_file(dict_test,os.path.join(FLAGS.experiment_rootdir,
                                               'predicted_and_real_Y.json'))
    f1.close()
    print '8-' 
    

def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
      sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
