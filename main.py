from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from model_create import model_create


if __name__=='__main__':
    # to train model again uncomment the following 2 lines
    # model is defined in class model_create in file model_create.py
#    model_obj = model_create()
#    model_obj.train_model()
    
    # load model from saved file
    model = load_model('my_model.h5')
    
    
    imgFileName = input('Enter image file name:')
    if not imgFileName:
        imgFileName = 'images.png'
        
    img = image.load_img(imgFileName, target_size = (28, 28), color_mode = 'grayscale')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    img_pred=model.predict(img).tolist()[0]
    
    print('\nGiven image is of digit', img_pred.index(max(img_pred)), 'with probability', max(img_pred))
