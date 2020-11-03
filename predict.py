import cv2 
import numpy as np
import mapping as mp

# Works for hdf5 models
def preprocess(path):
    image = cv2.imread(path) 
    image = cv2.resize(image, (100, 100)) 
    image = image/255.0
    img_batch = np.expand_dims(image, axis=0)
    return img_batch

def predict(model, image):
    pred = model.predict(image)
    pred = pred.tolist()[0]
    pred = mp.id_pred_dict[pred.index(max(pred))]
    return pred
    
