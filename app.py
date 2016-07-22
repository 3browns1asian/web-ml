# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:13:51 2016

@author: akshaybudhkar
"""
import os
import socketio
import eventlet
import sklearn
from flask import Flask, render_template
from sklearn.externals import joblib
import numpy as np
import pandas as pd
from scipy.fftpack import dct

def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def feature_extraction(data, train=False):
    new_data = []
    
    for f_data in data:
        
        left_vals = np.array(f_data["values"]["left"])
        right_vals = np.array(f_data["values"]["right"])
        
        a = left_vals
        b = right_vals
        features = []            
        
        # Left hand features
        if len(a) != 0 and len(a[0]) != 0:
            # Feature 1: Mean of DCT of Acceleration of X
            transformed_values_x = np.array(dct(a[:, 0]))
            features.append(round(np.mean(transformed_values_x), 3))
            
            # Feature 2: Mean of DCT of Acceleration of Y
            transformed_values_y = np.array(dct(a[:, 1]))
            features.append(round(np.mean(transformed_values_y), 3))
            
            # Feature 3: Mean of DCT of Acceleration of Z
            transformed_values_z = np.array(dct(a[:, 2]))
            features.append(round(np.mean(transformed_values_z), 3))
            
            # Feature 4/5: Mean Absolute Deviation and Mean of gyro in X
            features.append(round(mad(a[:, 3])))
            features.append(round(np.mean(a[:, 3])))
            
            # Feature 6/7: Mean Absolute Deviation and Mean of gyro in Y
            features.append(round(mad(a[:, 4])))
            features.append(round(np.mean(a[:, 4])))
            
            # Feature 8/9: Mean Absolute Deviation and Mean of gyro in Z
            features.append(round(mad(a[:, 5])))
            features.append(round(np.mean(a[:, 5])))
            
            # Feature 10/11: Standard Absolute Deviation and Mean of flex 1
            features.append(round(np.std(a[:, 6])))
            features.append(round(np.mean(a[:, 6])))
            
            # Feature 12/13: Standard Absolute Deviation and Mean of flex 2
            features.append(round(np.std(a[:, 7])))
            features.append(round(np.mean(a[:, 7])))
            
            # Feature 14/15: Standard Absolute Deviation and Mean of flex 3
            features.append(round(np.std(a[:, 8])))
            features.append(round(np.mean(a[:, 8])))
            
            # Feature 16/17: Standard Absolute Deviation and Mean of flex 4
            features.append(round(np.std(a[:, 9])))
            features.append(round(np.mean(a[:, 9])))
            
            # Feature 18/19: Standard Absolute Deviation and Mean of flex 5
            features.append(round(np.std(a[:, 10])))
            features.append(round(np.mean(a[:, 10])))            
        
        # Right hand features
        if len(b) != 0 and len(b[0]) != 0:
            # Feature 20: Mean of DCT of Acceleration of X
            transformed_values_x = np.array(dct(b[:, 0]))
            features.append(round(np.mean(transformed_values_x), 3))
            
            # Feature 21: Mean of DCT of Acceleration of Y
            transformed_values_y = np.array(dct(b[:, 1]))
            features.append(round(np.mean(transformed_values_y), 3))
            
            # Feature 22: Mean of DCT of Acceleration of Z
            transformed_values_z = np.array(dct(b[:, 2]))
            features.append(round(np.mean(transformed_values_z), 3))
            
            # Feature 23/24: Mean Absolute Deviation and Mean of gyro in X
            features.append(round(mad(b[:, 3])))
            features.append(round(np.mean(b[:, 3])))
            
            # Feature 25/26: Mean Absolute Deviation and Mean of gyro in Y
            features.append(round(mad(b[:, 4])))
            features.append(round(np.mean(b[:, 4])))
            
            # Feature 27/28: Mean Absolute Deviation and Mean of gyro in Z
            features.append(round(mad(b[:, 5])))
            features.append(round(np.mean(b[:, 5])))
            
            # Feature 29/30: Standard Absolute Deviation and Mean of flex 1
            features.append(round(np.std(b[:, 6])))
            features.append(round(np.mean(b[:, 6])))
            
            # Feature 31/32: Standard Absolute Deviation and Mean of flex 2
            features.append(round(np.std(b[:, 7])))
            features.append(round(np.mean(b[:, 7])))
            
            # Feature 33/34: Standard Absolute Deviation and Mean of flex 3
            features.append(round(np.std(b[:, 8])))
            features.append(round(np.mean(b[:, 8])))
            
            # Feature 35/36: Standard Absolute Deviation and Mean of flex 4
            features.append(round(np.std(b[:, 9])))
            features.append(round(np.mean(b[:, 9])))
            
            # Feature 37/38: Standard Absolute Deviation and Mean of flex 5
            features.append(round(np.std(b[:, 10])))
            features.append(round(np.mean(b[:, 10])))
                
            if len(features) > 0:
                if train:
                    new_data.append({"label": f_data["label"], "user": f_data["user"], "features": features})
                else:
                    new_data.append({"features": features})
    
    return new_data

def process_data(data):
    final_data = []
    
    values = {"left": [], "right": []}

    for point in data:
        splits = point.split("|")
        left_array = splits[0].split(",")
        right_array = splits[1].split(",")
        
        left_value = [float(left_array[0]), float(left_array[1]), float(left_array[2]),
                      float(left_array[3]), float(left_array[4]), float(left_array[5]),
                      float(left_array[6]), float(left_array[7]),
                      float(left_array[8]), float(left_array[9]), float(left_array[10])]
                      
        right_value = [float(right_array[0]), float(right_array[1]), float(right_array[2]),
                      float(right_array[3]), float(right_array[4]), float(right_array[5]),
                      float(right_array[6]), float(right_array[7]),
                      float(right_array[8]), float(right_array[9]), float(right_array[10])]
                      
        values["left"].append(left_value)
        values["right"].append(right_value)
        
    final_data.append({"values": values})
    
    predict_data = feature_extraction(final_data)
    
    predict_df = pd.DataFrame(predict_data)
    X_pred = np.array(predict_df.features.tolist())
    
    clf_1 = joblib.load('ml-models/bayes.pkl')
    preds_nb = clf_1.predict(X_pred)
    
    cols = joblib.load('ml-models/col.pkl')
    
    return cols[preds_nb]

build_data = []    
sio = socketio.Server()
app = Flask(__name__)

@app.route('/')
def index():
    """Serve the client-side application."""
    return render_template('index.html')

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

@sio.on('sendSensorData')
def message(sid, data):
    global build_data   
    splits = data.split('\n')
    for line in splits:
        if line != "END":
            print 'Appended a line to the data.'
            build_data.append(line)
            sio.emit('receivedSensorData', data)
        else:
            value = process_data(build_data)
            build_data = []
            print 'Found END of the data.'
            sio.emit('predictedValue', value)

@sio.on('disconnect')
def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))    
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', port)), app)