from art.attacks.extraction import CopycatCNN
import numpy as np
import torch

class CustomCopycatCNN(CopycatCNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _query_label(self, x: np.ndarray) -> np.ndarray:
        x_tensor = torch.FloatTensor(x).to(self.estimator._device)
        
        model_outputs = self.estimator._model(x_tensor)
                
        output = model_outputs[0].prediction_logits.detach().cpu().numpy().astype(np.float32)
        
        if not self.use_probability:
            output = np.argmax(output, axis=1)
            output = np.eye(self.estimator.nb_classes)[output]  
        
        return output

    def extract(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs):

        thieved_classifier = kwargs["thieved_classifier"]

        selected_x = self._select_data(x)


        selected_x = torch.tensor(selected_x) if isinstance(selected_x, np.ndarray) else selected_x
        selected_x = selected_x.permute(0, 2, 1).numpy()

        fake_labels = self._query_label(selected_x)
        selected_x = torch.tensor(selected_x)

        selected_x = selected_x.permute(0, 2, 1).numpy()  
        thieved_classifier.fit(  
            x=selected_x,
            y=fake_labels,
            batch_size=self.batch_size_fit,
            nb_epochs=self.nb_epochs,
        )

        return thieved_classifier  