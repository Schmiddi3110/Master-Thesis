import numpy as np
from art.estimators.classification.classifier import Classifier
import numpy as np
from art.estimators.classification.classifier import Classifier

class AeonClassifier(Classifier):
    def __init__(self, model, input_shape, nb_classes, clip_values=None, preprocessing_defences=None, postprocessing_defences=None, preprocessing=None):

        super().__init__(
            model=None, 
            clip_values=clip_values, 
            preprocessing_defences=preprocessing_defences, 
            postprocessing_defences=postprocessing_defences, 
            preprocessing=preprocessing
        )
        self._model = model  
        self.input_shape = input_shape
        self.nb_classes = nb_classes

    def predict(self, x, **kwargs):
        x, _ = self._apply_preprocessing(x, y=None, fit=False)

        x = np.array(x)
        if x.ndim == 2: 
            x = x.reshape((x.shape[0], *self.input_shape))

        predictions = self._model.predict_proba(x)

        predictions = self._apply_postprocessing(preds=predictions, fit=False)   


        return predictions

    def fit(self, x, y, **kwargs):

        raise NotImplementedError("Fit method not implemented for this model.")

    def loss_gradient(self, x, y, **kwargs):

        raise NotImplementedError("Gradient computation not implemented for this model.")

    def class_gradient(self, x, label=None, **kwargs):

        raise NotImplementedError("Class gradient computation not implemented for this model.")

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value):
        self._input_shape = value

    @property
    def nb_classes(self):
        return self._nb_classes

    @nb_classes.setter
    def nb_classes(self, value):
        self._nb_classes = value
