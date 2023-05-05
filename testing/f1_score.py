from keras import backend as K


class F1:

    def __call__(self, y_actual, y_pred):
        tp = K.sum(K.round(K.clip(y_actual * y_pred, 0, 1)))
        fp = K.sum(K.round(K.clip(y_pred - y_actual, 0, 1)))
        fn = K.sum(K.round(K.clip(y_actual - y_pred, 0, 1)))
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
        f1_score = 2 * precision * recall / (precision + recall + K.epsilon())
        return f1_score
