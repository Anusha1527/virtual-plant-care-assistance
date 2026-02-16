import os
import cv2
import mahotas
import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ---------------- PATHS (MATCH YOUR PROJECT)
train_path = "image_classification/dataset/train"
h5_train_data = "image_classification/output/train_data.h5"
h5_train_labels = "image_classification/output/train_labels.h5"

fixed_size = (500, 500)
bins = 8

# ---------------- FEATURE FUNCTIONS

def rgb_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def bgr_hsv(rgb_img):
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

def img_segmentation(rgb_img, hsv_img):
    lower_green = np.array([25,0,20])
    upper_green = np.array([100,255,255])
    healthy = cv2.inRange(hsv_img, lower_green, upper_green)

    lower_brown = np.array([10,0,10])
    upper_brown = np.array([30,255,255])
    disease = cv2.inRange(hsv_img, lower_brown, upper_brown)

    final = healthy + disease
    return cv2.bitwise_and(rgb_img, rgb_img, mask=final)

def fd_hu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(gray)).flatten()

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return mahotas.features.haralick(gray).mean(axis=0)

def fd_hist(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[bins,bins,bins],[0,256,0,256,0,256])
    cv2.normalize(hist,hist)
    return hist.flatten()

# ---------------- FEATURE EXTRACTION

labels = []
features = []

print("[INFO] Extracting features...")

for cls in os.listdir(train_path):
    folder = os.path.join(train_path, cls)

    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file))
        if img is None:
            continue

        img = cv2.resize(img, fixed_size)

        rgb = rgb_bgr(img)
        hsv = bgr_hsv(rgb)
        seg = img_segmentation(rgb,hsv)

        f = np.hstack([fd_hist(seg), fd_haralick(seg), fd_hu(seg)])

        labels.append(cls)
        features.append(f)

    print("Processed:", cls)

features = np.array(features)
labels = np.array(labels)

print("Feature shape:", features.shape)

# ---------------- ENCODE + SCALE

le = LabelEncoder()
target = le.fit_transform(labels)

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# ---------------- SAVE HDF5

os.makedirs("image_classification/output", exist_ok=True)

with h5py.File(h5_train_data,'w') as f:
    f.create_dataset("dataset_1", data=features)

with h5py.File(h5_train_labels,'w') as f:
    f.create_dataset("dataset_1", data=target)

print("[INFO] Features saved")

# ---------------- TRAINING

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=9)

models = [
    ('LR', LogisticRegression(max_iter=1000)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('DT', DecisionTreeClassifier()),
    ('RF', RandomForestClassifier(n_estimators=100)),
    ('NB', GaussianNB()),
    ('SVM', SVC())
]

print("\n[INFO] Cross validation:")

for name, model in models:
    kfold = KFold(n_splits=5, shuffle=True, random_state=9)
    score = cross_val_score(model, X_train, y_train, cv=kfold).mean()
    print(name, ":", score)

# ---------------- FINAL RANDOM FOREST

clf = RandomForestClassifier(n_estimators=100, random_state=9)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

print("Final Accuracy:", accuracy_score(y_test, pred))