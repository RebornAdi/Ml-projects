import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from config import NUM_GESTURES, TEST_SIZE, RANDOM_STATE

def load_data():
    X, y = [], []
    for gesture_idx in range(NUM_GESTURES):
        gesture_dir = os.path.join('data', str(gesture_idx))
        for file in os.listdir(gesture_dir):
            data = np.load(os.path.join(gesture_dir, file))
            X.append(data)
            y.append(gesture_idx)
    return np.array(X), np.array(y)

def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Create and train model
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print(f'Training accuracy: {model.score(X_train, y_train):.2f}')
    print(f'Test accuracy: {model.score(X_test, y_test):.2f}')
    
    return model

def main():
    # Load data
    X, y = load_data()
    print(f"Loaded {len(X)} samples")
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/gesture_model.pkl')
    print("Model saved to models/gesture_model.pkl")

if __name__ == "__main__":
    main()