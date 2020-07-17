import os
import glob
import argparse
import matplotlib
import numpy as np

from multiprocessing import Process

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, save_output
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
parser.add_argument('--output', default='output/', type=str, help='Output Dir.')
# parser.add_argument('--folder', default='folder', type=str, help='Output folder.')
# parser.add_argument('--start', default=1, type=int, help='Start file number')
# parser.add_argument('--end', default=10, type=int, help='End file number')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
file_name = args.input
# start = args.start
# end = args.end
output_dir = args.output
# folder = args.folder

def model_predictions(folder, start, end):
    batch_siz = 500
    iterator_count = int(((end - (start-1)) / batch_siz ))+1
    for j in range(1, iterator_count):
        print("Batch ", j, " started..")
        jstart = (j - 1)*batch_siz + start
        jend = j*batch_siz + (start - 1)
        inputs = load_images(file_name, folder, jstart, jend)
        outputs = predict(model, inputs, batch_size=100)
        save_output(outputs, output_dir, folder, jstart, jend)  
    return True

for i in range(1, 101):
    start = (i - 1)*4000 + 1
    end = i*4000
    print("Processing BG : ", i)  
    folder = f'{i}'
    model_predictions(folder, start, end)
    # p = Process(target=model_predictions, args=(folder, start, end, ))
    # p.start()
    # p.join()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    

    # viz = display_images(outputs.copy(), inputs.copy())
    # plt.figure(figsize=(10,5))
    # plt.imshow(viz)
    # plt.savefig('test.png')
    # plt.show()

    
