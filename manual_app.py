import numpy as np
import os
import sklearn
from sklearn.externals import joblib
import pandas as pd
from scipy.fftpack import dct


def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def feature_extraction(data, train=False):
    new_data = []
    print(data)

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
            features.append(round(mad(a[:, 3]), 3))
            features.append(round(np.mean(a[:, 3]), 3))
            features.append(round(np.amax(a[:, 3]), 3))
            features.append(round(np.amin(a[:, 3]), 3))

            # Feature 6/7: Mean Absolute Deviation and Mean of gyro in Y
            features.append(round(mad(a[:, 4]), 3))
            features.append(round(np.mean(a[:, 4]), 3))
            features.append(round(np.amax(a[:, 4]), 3))
            features.append(round(np.amin(a[:, 4]), 3))

            # Feature 8/9: Mean Absolute Deviation and Mean of gyro in Z
            features.append(round(mad(a[:, 5]), 3))
            features.append(round(np.mean(a[:, 5]), 3))
            features.append(round(np.amax(a[:, 5]), 3))
            features.append(round(np.amin(a[:, 5]), 3))

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
                    new_data.append({"label": f_data["label"], "user": f_data["user"], "features": features[:26]})
                else:
                    new_data.append({"features": features[:26]})

    return new_data


def process_data(data):
    final_data = []

    values = {"left": [], "right": []}

    for point in data:

        left_array = point.split(",")

        left_value = [float(left_array[0]), float(left_array[1]), float(left_array[2]),
                      float(left_array[3]), float(left_array[4]), float(left_array[5]),
                      float(left_array[6]), float(left_array[7]),
                      float(left_array[8]), float(left_array[9]), float(left_array[10])]

        right_value = [float(0) for x in range(12)]

        values["left"].append(left_value)
        values["right"].append(right_value)

    final_data.append({"values": values})

    predict_data = feature_extraction(final_data)

    predict_df = pd.DataFrame(predict_data)
    X_pred = np.array(predict_df.features.tolist())

    print "Reached here"
    clf_1 = joblib.load('ml-models/svm_minus.pkl')
    preds_nb = clf_1.predict(X_pred)

    cols = joblib.load('ml-models/col_minus.pkl')

    print cols[preds_nb[0]]
    return cols[preds_nb[0]]


data = """7.81,-5.17,-5.17,0.03,0.03,0.03,474,417,432,333,379
8.14,-4.93,-4.93,0.02,0.02,0.02,474,417,432,333,379
7.47,-5.64,-5.64,-0.35,-0.35,-0.35,474,417,432,334,380
4.96,-6.59,-6.59,-0.43,-0.43,-0.43,475,416,432,334,380
4.96,-7.51,-7.51,-0.28,-0.28,-0.28,474,416,431,334,377
1.49,-8.43,-8.43,-0.51,-0.51,-0.51,471,415,429,334,373
-2.14,-8.74,-8.74,-0.26,-0.26,-0.26,468,412,428,334,372
-1.46,-9.08,-9.08,0.16,0.16,0.16,467,412,430,334,377
-1.80,-9.34,-9.34,0.02,0.02,0.02,467,412,430,334,377
-2.02,-9.38,-9.38,0.04,0.04,0.04,466,411,431,334,378
-1.64,-9.35,-9.35,0.01,0.01,0.01,466,411,432,334,379
END
-1.61,-9.32,-9.32,0.02,0.02,0.02,469,415,430,335,377
7.77,-5.15,-5.15,0.04,0.04,0.04,469,415,430,335,377
6.43,-5.54,-5.54,-0.18,-0.18,-0.18,469,415,430,336,377
8.11,-5.81,-5.81,-0.30,-0.30,-0.30,469,414,430,336,378
4.60,-7.42,-7.42,-0.46,-0.46,-0.46,467,413,429,335,379
2.84,-7.93,-7.93,-0.26,-0.26,-0.26,465,412,428,334,376
1.84,-9.53,-9.53,-0.34,-0.34,-0.34,461,409,428,334,375
-1.08,-8.70,-8.70,-0.28,-0.28,-0.28,459,408,428,334,376
-0.40,-9.75,-9.75,0.19,0.19,0.19,459,408,432,334,380
-0.84,-9.29,-9.29,0.04,0.04,0.04,460,410,440,334,392
-0.64,-9.28,-9.28,0.15,0.15,0.15,460,410,439,334,391
END"""

build_data = []

splits = data.split('\n')

for line in splits:
    if line != "END":
        build_data.append(line)
    else:
        value = process_data(build_data)
        build_data = []
        print 'Found END of the data.'
