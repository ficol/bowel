from bowel.models.semisupervised import Semisupervised
from bowel.utils.train_utils import get_score

import numpy as np

class CurriculumLabel(Semisupervised):
    def train(self, X, y, X_unannotated, X_val=None, y_val=None):
        self.model.train(X, y, X_val, y_val)
        with open('models/curriculummodel.txt', 'w') as f:
            for i in range(self.config['parts']):
                train_metrics = get_score(y, self.predict(X))
                valid_metrics = get_score(y_val, self.predict(X_val))
                f.write(f'{i}\nTrain:\n')
                for j in train_metrics:
                    f.write(f'{j}: {train_metrics[j]}\n')
                f.write('Valid:\n')
                for j in valid_metrics:
                    f.write(f'{j}: {valid_metrics[j]}\n')
                f.write('\n')
                self.save(f'models/curriculummodel_{i}.h5')
                y_unannotated = np.squeeze(self.predict(X_unannotated))
                # confident_args = np.argpartition(np.where(y_unannotated < 0.5, 1 - y_unannotated, y_unannotated).mean(axis=1), -part)[-part:]
                confident_args = np.where(y_unannotated < 0.5, 1 - y_unannotated, y_unannotated).min(axis=1)>0.6
                confident_X_unannotated = X_unannotated[confident_args]
                confident_y_unannotated = (y_unannotated[confident_args] > 0.5).astype(int)
                X_unannotated = np.delete(X_unannotated, confident_args, axis=0)
                y_unannotated = np.delete(y_unannotated, confident_args, axis=0)
                X = np.concatenate((X, confident_X_unannotated), axis=0)
                y = np.concatenate((y, confident_y_unannotated), axis=0)
                self.model = self.build_model()
                self.model.train(X, y, X_val, y_val)
            train_metrics = get_score(y, self.predict(X))
            valid_metrics = get_score(y_val, self.predict(X_val))
            f.write(f'{i}\nTrain:\n')
            for j in train_metrics:
                f.write(f'{j}: {train_metrics[j]}\n')
            f.write('Valid:\n')
            for j in valid_metrics:
                f.write(f'{j}: {valid_metrics[j]}\n')
            f.write('\n')
        return self.model

