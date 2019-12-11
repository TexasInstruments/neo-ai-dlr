from __future__ import print_function
from dlr import DLRModel
import numpy as np
import os
import sys
import platform
from timeit import default_timer as timer

# Load the model
model_path = sys.argv[1]
batch_size = int(sys.argv[2])
input_node = sys.argv[3]

device = 'cpu'
model = DLRModel(model_path, device)
# Run the model
imageIN = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dog.npy')).astype(np.float32)
imageIN = np.squeeze(imageIN, axis=0)
print(imageIN.shape)
image = np.concatenate([imageIN[np.newaxis, :, :]]*batch_size)
print(image.shape)
image = image * 0.25
image = image.transpose(0, 2, 3, 1)
print(image.shape)
#flatten within a input array
input_data = {input_node: image}
print("Testing inference on " + sys.argv[1])
start_time = timer()
probabilities = model.run(input_data) #need to be a list of input arrays matching input names
end_time = timer()
print("First time:" + str(end_time - start_time))
start_time = timer()
output_tensor = model.run(input_data) #need to be a list of input arrays matching input names
end_time = timer()

#np.set_printoptions(threshold=np.inf)
probabilities = output_tensor[0]
for i in range(0,probabilities.shape[0]):
  print("MAX VALUE:" + str(np.amax(probabilities[i,:])) + " at index:" + str(np.argmax(probabilities[i,:])))
print("Second time:" + str(end_time - start_time) + " seconds,  number of inferences:" + str(batch_size))
print("Time per inference:" + str((end_time - start_time) / float(batch_size)) + " seconds")
