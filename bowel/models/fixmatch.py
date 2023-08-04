from bowel.models.semisupervised import Semisupervised
from bowel.utils.train_utils import get_score
from bowel.utils.audio_utils import mixup

import numpy as np

class FixMatch(Semisupervised):
    def train(self, X, y, weak_X_unannotated, strong_X_unannotated, X_val=None, y_val=None):
        self.model.train(X, y, X_val, y_val)
        y_unannotated = np.squeeze(self.predict(weak_X_unannotated))
        confident_args = np.where(y_unannotated < 0.5, 1 - y_unannotated, y_unannotated).min(axis=1)>0.55
        confident_X_unannotated = strong_X_unannotated[confident_args]
        confident_y_unannotated = np.flip((y_unannotated[confident_args] > 0.5).astype(int), axis=1)
        X = np.concatenate((X, confident_X_unannotated), axis=0)
        y = np.concatenate((y, confident_y_unannotated), axis=0)
        self.model = self.build_model()
        self.model.train(X, y, X_val, y_val)
        train_metrics = get_score(y, self.predict(X))
        valid_metrics = get_score(y_val, self.predict(X_val))
        print('Train:')
        for j in train_metrics:
            print(f'{j}: {train_metrics[j]}')
        print('Valid:')
        for j in valid_metrics:
            print(f'{j}: {valid_metrics[j]}') 
        return self.model

