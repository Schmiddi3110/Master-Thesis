from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import ExtractionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import to_categorical

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class CustomKnockoffNets(ExtractionAttack):
    attack_params = ExtractionAttack.attack_params + [
        "batch_size_fit",
        "batch_size_query",
        "nb_epochs",
        "nb_stolen",
        "sampling_strategy",
        "reward",
        "verbose",
        "use_probability",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        batch_size_fit: int = 1,
        batch_size_query: int = 1,
        nb_epochs: int = 10,
        nb_stolen: int = 1,
        sampling_strategy: str = "random",
        reward: str = "all",
        verbose: bool = True,
        use_probability: bool = False,
    ) -> None:

        super().__init__(estimator=classifier)

        self.batch_size_fit = batch_size_fit
        self.batch_size_query = batch_size_query
        self.nb_epochs = nb_epochs
        self.nb_stolen = nb_stolen
        self.sampling_strategy = sampling_strategy
        self.reward = reward
        self.verbose = verbose
        self.use_probability = use_probability
        self._check_params()

    def extract(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> "CLASSIFIER_TYPE":
        if self.sampling_strategy == "random" and y is not None:  
            logger.warning("This attack with random sampling strategy does not use the provided label y.")

        if self.sampling_strategy == "adaptive" and y is None:  
            raise ValueError("This attack with adaptive sampling strategy needs label y.")

        if x.shape[0] < self.nb_stolen: 
            logger.warning(
                "The size of the source input is smaller than the expected number of queries submitted "
                "to the victim classifier."
            )

        thieved_classifier = kwargs.get("thieved_classifier")
        if thieved_classifier is None or not isinstance(thieved_classifier, ClassifierMixin):  
            raise ValueError("A thieved classifier is needed.")

        if self.sampling_strategy == "random":
            thieved_classifier = self._random_extraction(x, thieved_classifier)  
        else:
            thieved_classifier = self._adaptive_extraction(x, y, thieved_classifier) 

        return thieved_classifier

    def _random_extraction(self, x: np.ndarray, thieved_classifier: "CLASSIFIER_TYPE") -> "CLASSIFIER_TYPE":
        selected_x = self._select_data(x)

        fake_labels = self._query_label(selected_x)

        thieved_classifier.fit(
            x=selected_x,
            y=fake_labels,
            batch_size=self.batch_size_fit,
            nb_epochs=self.nb_epochs,
            verbose=False,
        )

        return thieved_classifier

    def _select_data(self, x: np.ndarray) -> np.ndarray:
        nb_stolen = np.minimum(self.nb_stolen, x.shape[0])
        rnd_index = np.random.choice(x.shape[0], nb_stolen, replace=False)

        return x[rnd_index].astype(ART_NUMPY_DTYPE)

    def _query_label(self, x: np.ndarray) -> np.ndarray:
        labels = self.estimator.predict(x=x, batch_size=self.batch_size_query)
        if not self.use_probability:
            labels = np.argmax(labels, axis=1)
            labels = to_categorical(labels=labels, nb_classes=self.estimator.nb_classes)

        return labels

    def _adaptive_extraction(
        self, x: np.ndarray, y: np.ndarray, thieved_classifier: "CLASSIFIER_TYPE"
    ) -> "CLASSIFIER_TYPE":

        if len(y.shape) == 2:
            nb_actions = len(np.unique(np.argmax(y, axis=1)))
        elif len(y.shape) == 1:
            nb_actions = len(np.unique(y))
        else:
            raise ValueError("Target values `y` has a wrong shape.")

        if self.reward in ("div", "all"):
            self.y_avg = np.zeros(self.estimator.nb_classes)

        if self.reward == "all":
            self.reward_avg = np.zeros(3)
            self.reward_var = np.zeros(3)

        h_func = np.zeros(nb_actions)
        learning_rate = np.zeros(nb_actions)
        probs = np.ones(nb_actions) / nb_actions
        selected_x = []
        queried_labels = []

        avg_reward = 0.0
        for iteration in trange(1, self.nb_stolen + 1, desc="Knock-off nets", disable=not self.verbose):            
            action = np.random.choice(np.arange(0, nb_actions), p=probs)

            sampled_x = self._sample_data(x, y, action)
            selected_x.append(sampled_x)

            y_output = self.estimator.predict(x=np.array([sampled_x]), batch_size=self.batch_size_query)
            fake_label = np.argmax(y_output, axis=1)
            fake_label = to_categorical(labels=fake_label, nb_classes=self.estimator.nb_classes)
            queried_labels.append(fake_label[0])

            thieved_classifier.fit(
                x=np.array([sampled_x]),
                y=fake_label,
                batch_size=self.batch_size_fit,
                nb_epochs=1,
                verbose=False,
            )

            y_hat = thieved_classifier.predict(x=np.array([sampled_x]), batch_size=self.batch_size_query)

            reward = self._reward(y_output, y_hat, iteration)

            avg_reward = avg_reward + (1.0 / iteration) * (reward - avg_reward)
            learning_rate[action] += 1

            for i_action in range(nb_actions):
                if i_action != action:
                    h_func[i_action] = (h_func[i_action] - 1.0 / learning_rate[action] * (reward - avg_reward) * probs[i_action])
                else:
                    h_func[i_action] = h_func[i_action] + 1.0 / learning_rate[action] * (reward - avg_reward) * (1 - probs[i_action])

            aux_exp = np.exp(h_func)
            
            probs = aux_exp / np.sum(aux_exp)
            

        thieved_classifier.fit(
            x=np.array(selected_x),
            y=np.array(queried_labels),
            batch_size=self.batch_size_fit,
            nb_epochs=self.nb_epochs,
        )

        return thieved_classifier

    @staticmethod
    def _sample_data(x: np.ndarray, y: np.ndarray, action: int) -> np.ndarray:
        if len(y.shape) == 2:
            y_index = np.argmax(y, axis=1)
        else:
            y_index = y

        x_index = x[y_index == action]
        rnd_idx = np.random.choice(len(x_index))

        return x_index[rnd_idx]

    def _reward(self, y_output: np.ndarray, y_hat: np.ndarray, n: int) -> float | np.ndarray:

        if self.reward == "cert":
            return self._reward_cert(y_output)
        if self.reward == "div":
            return self._reward_div(y_output, n)
        if self.reward == "loss":
            return self._reward_loss(y_output, y_hat)
        return self._reward_all(y_output, y_hat, n)

    @staticmethod
    def _reward_cert(y_output: np.ndarray) -> float:

        largests = np.partition(y_output.flatten(), -2)[-2:]
        reward = largests[1] - largests[0]

        return reward

    def _reward_div(self, y_output: np.ndarray, n: int) -> float:

        self.y_avg = self.y_avg + (1.0 / n) * (y_output[0] - self.y_avg)

        reward = 0
        for k in range(self.estimator.nb_classes):
            reward += np.maximum(0, y_output[0][k] - self.y_avg[k])

        return reward

    def _reward_loss(self, y_output: np.ndarray, y_hat: np.ndarray) -> float:

        aux_exp = np.exp(y_output[0])
        probs_output = aux_exp / np.sum(aux_exp)

        aux_exp = np.exp(y_hat[0])
        probs_hat = aux_exp / np.sum(aux_exp)

        reward = 0
        for k in range(self.estimator.nb_classes):
            reward += -probs_output[k] * np.log(probs_hat[k])

        return reward

    def _reward_all(self, y_output: np.ndarray, y_hat: np.ndarray, n: int) -> float:

        reward_cert = self._reward_cert(y_output)
        reward_div = self._reward_div(y_output, n)
        reward_loss = self._reward_loss(y_output, y_hat)        
        reward = [reward_cert, reward_div, reward_loss]
        self.reward_avg = self.reward_avg + (1.0 / n) * (reward - self.reward_avg)
        self.reward_var = self.reward_var + (1.0 / n) * ((reward - self.reward_avg) ** 2 - self.reward_var)

        if n > 1:            
            reward = (reward - self.reward_avg) / np.sqrt(self.reward_var + 1e-7)
            
        else:
            reward = [max(min(r, 1), 0) for r in reward]

        return float(np.mean(reward))

    def _check_params(self) -> None:
        if not isinstance(self.batch_size_fit, int) or self.batch_size_fit <= 0:
            raise ValueError("The size of batches for fitting the thieved classifier must be a positive integer.")

        if not isinstance(self.batch_size_query, int) or self.batch_size_query <= 0:
            raise ValueError("The size of batches for querying the victim classifier must be a positive integer.")

        if not isinstance(self.nb_epochs, int) or self.nb_epochs <= 0:
            raise ValueError("The number of epochs must be a positive integer.")

        if not isinstance(self.nb_stolen, int) or self.nb_stolen <= 0:
            raise ValueError("The number of queries submitted to the victim classifier must be a positive integer.")

        if self.sampling_strategy not in ["random", "adaptive"]:
            raise ValueError("Sampling strategy must be either `random` or `adaptive`.")

        if self.reward not in ["cert", "div", "loss", "all"]:
            raise ValueError("Reward type must be in ['cert', 'div', 'loss', 'all'].")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
        if not isinstance(self.use_probability, bool):
            raise ValueError("The argument `use_probability` has to be of type bool.")