import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

class JaundiceClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.trained = False

    @staticmethod
    def load_images(jaundice_dir, normal_dir):
        images = []
        labels = []
        directories = [(jaundice_dir, 'Jaundice'), (normal_dir, 'Normal')]
        for dir_path, label in directories:
            filenames = os.listdir(dir_path)
            for filename in filenames:
                img_path = os.path.join(dir_path, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image = cv2.resize(image, (64, 64))
                    images.append(image)
                    labels.append(label)
        images = np.array(images)
        labels = np.array(labels)
        return images, labels

    @staticmethod
    def encode_labels(labels):
        return np.where(labels == 'Jaundice', 1, 0)

    def train(self, images, labels):
        n_samples = images.shape[0]
        images_flattened = images.reshape(n_samples, -1)
        encoded_labels = self.encode_labels(labels)
        X_train, X_test, y_train, y_test = train_test_split(images_flattened, encoded_labels, test_size=0.2, random_state=42)
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X_train, y_train)
        self.trained = True
        return X_test, y_test

    def predict(self, img_array):
        if not self.trained:
            raise Exception("Model not trained.")
        return self.model.predict(img_array)

    def predict_proba(self, img_array):
        if not self.trained:
            raise Exception("Model not trained.")
        return self.model.predict_proba(img_array)

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('L')
        img = img.resize((64, 64))
        img_array = np.array(img)
        img_array = img_array.flatten()
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def plot_metrics(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Jaundice']))
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='bone', xticklabels=['Normal', 'Jaundice'], yticklabels=['Normal', 'Jaundice'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': np.arange(X_test.shape[1]),
            'Importance': importances
        })
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        print("Top features:")
        print(feature_importance_df.head(10))
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
        plt.title('Top 20 Feature Importances for XGBoost Model')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.show()


def create_app(classifier: JaundiceClassifier):
    app = Flask(__name__)

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'healthy'}), 200

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            filename = secure_filename(file.filename)
            temp_filepath = os.path.join('tmp', filename)
            os.makedirs('tmp', exist_ok=True)
            file.save(temp_filepath)
            try:
                preprocessed_img = classifier.preprocess_image(temp_filepath)
                prediction = classifier.predict(preprocessed_img)
                predicted_class = 'jaundice' if prediction.flatten()[0] == 1 else 'normal'
                return jsonify({'prediction': predicted_class})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            finally:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
        else:
            return jsonify({'error': 'Something went wrong with the file upload'}), 500
    return app

if __name__ == "__main__":
    jaundice_dir = 'data/jaundice'
    normal_dir = 'data/normal'
    images, labels = JaundiceClassifier.load_images(jaundice_dir, normal_dir)
    classifier = JaundiceClassifier()
    X_test, y_test = classifier.train(images, labels)
    classifier.plot_metrics(X_test, y_test)
    app = create_app(classifier)
    port = 5000
    app.run(host='0.0.0.0', port=port) 