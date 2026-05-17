import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft

# =========================================================
# DATASET GENERATOR
# =========================================================

def generate_shape(shape_type, size=200, rotation=0, scale=1.0, tx=0, ty=0):

    img = np.zeros((size, size), dtype=np.uint8)

    center = (size // 2 + tx, size // 2 + ty)

    if shape_type == "circle":
        radius = int(40 * scale)
        cv2.circle(img, center, radius, 255, -1)

    elif shape_type == "square":

        side = int(80 * scale)

        pts = np.array([
            [-side//2, -side//2],
            [ side//2, -side//2],
            [ side//2,  side//2],
            [-side//2,  side//2]
        ])

        theta = np.radians(rotation)

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        pts = np.dot(pts, R).astype(int)

        pts[:, 0] += center[0]
        pts[:, 1] += center[1]

        cv2.fillPoly(img, [pts], 255)

    elif shape_type == "triangle":

        side = int(100 * scale)

        pts = np.array([
            [0, -side//2],
            [-side//2, side//2],
            [side//2, side//2]
        ])

        theta = np.radians(rotation)

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        pts = np.dot(pts, R).astype(int)

        pts[:, 0] += center[0]
        pts[:, 1] += center[1]

        cv2.fillPoly(img, [pts], 255)

    return img

# =========================================================
# CHAIN CODE
# =========================================================

def freeman_chain_code_8dir(points):

    directions = [
        (1,0),
        (1,1),
        (0,1),
        (-1,1),
        (-1,0),
        (-1,-1),
        (0,-1),
        (1,-1)
    ]

    code = []

    for i in range(len(points)-1):

        dx = np.sign(points[i+1][0] - points[i][0])
        dy = np.sign(points[i+1][1] - points[i][1])

        for idx, (ddx, ddy) in enumerate(directions):

            if dx == ddx and dy == ddy:
                code.append(idx)

    return code


def freeman_chain_code_4dir(points):

    directions = [
        (1,0),
        (0,1),
        (-1,0),
        (0,-1)
    ]

    code = []

    for i in range(len(points)-1):

        dx = np.sign(points[i+1][0] - points[i][0])
        dy = np.sign(points[i+1][1] - points[i][1])

        for idx, (ddx, ddy) in enumerate(directions):

            if dx == ddx and dy == ddy:
                code.append(idx)

    return code


def normalize_chain_code(chain):

    norm = []

    for i in range(len(chain)-1):
        diff = (chain[i+1] - chain[i]) % 8
        norm.append(diff)

    return norm

# =========================================================
# FOURIER DESCRIPTOR
# =========================================================

def fourier_descriptor(contour, num_coeff=20):

    contour = contour[:,0,:]

    complex_contour = contour[:,0] + 1j * contour[:,1]

    fd = fft(complex_contour)

    fd = np.abs(fd)

    # normalization
    fd = fd / (fd[1] + 1e-6)

    return fd[:num_coeff]


def reconstruct_shape(fd_complex, keep):

    fd = fd_complex.copy()

    fd[keep:-keep] = 0

    recon = np.fft.ifft(fd)

    return recon.real, recon.imag

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(img):

    contours, _ = cv2.findContours(
        img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    contour = contours[0]

    # ----------------------------------------
    # REGION PROPERTIES
    # ----------------------------------------

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(contour, True)

    M = cv2.moments(contour)

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = w / h

    rect_area = w * h

    extent = area / rect_area

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    solidity = area / hull_area

    compactness = (perimeter ** 2) / (4 * np.pi * area)

    # ----------------------------------------
    # MOMENTS
    # ----------------------------------------

    hu = cv2.HuMoments(M).flatten()

    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    mu20 = M["mu20"]
    mu02 = M["mu02"]
    mu11 = M["mu11"]

    # ----------------------------------------
    # FOURIER
    # ----------------------------------------

    fd = fourier_descriptor(contour, 10)

    # ----------------------------------------
    # FEATURE VECTOR
    # ----------------------------------------

    features = [

        area,
        perimeter,
        aspect_ratio,
        extent,
        solidity,
        compactness,

        mu20,
        mu02,
        mu11,

        hu[0],
        hu[1],
        hu[2]

    ]

    features.extend(fd.tolist())

    return features, contour, hu

# =========================================================
# DATASET CREATION
# =========================================================

dataset = []
labels = []

shape_classes = ["circle", "square", "triangle"]

for label, shape_name in enumerate(shape_classes):

    for i in range(8):

        rotation = np.random.randint(0, 180)

        scale = np.random.uniform(0.7, 1.2)

        tx = np.random.randint(-20, 20)

        ty = np.random.randint(-20, 20)

        img = generate_shape(
            shape_name,
            rotation=rotation,
            scale=scale,
            tx=tx,
            ty=ty
        )

        features, contour, hu = extract_features(img)

        dataset.append(features)

        labels.append(label)

# =========================================================
# KNN CLASSIFICATION
# =========================================================

X = np.array(dataset)

y = np.array(labels)

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\n=================================================")
print("KNN CLASSIFICATION")
print("=================================================")

print(f"\nAccuracy : {acc*100:.2f}%")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred,
    target_names=shape_classes
))

# =========================================================
# VISUALIZATION
# =========================================================

fig, axes = plt.subplots(3, 4, figsize=(12, 10))

idx = 0

for r in range(3):

    for c in range(4):

        if idx >= len(shape_classes):
            break

        img = generate_shape(
            shape_classes[idx],
            rotation=np.random.randint(0,180),
            scale=np.random.uniform(0.8,1.2)
        )

        features, contour, hu = extract_features(img)

        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(display, [contour], -1, (0,255,0), 2)

        hull = cv2.convexHull(contour)

        cv2.drawContours(display, [hull], -1, (255,0,0), 2)

        epsilon = 0.02 * cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(contour, epsilon, True)

        cv2.drawContours(display, [approx], -1, (0,0,255), 2)

        axes[r,c].imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

        axes[r,c].set_title(shape_classes[idx])

        axes[r,c].axis("off")

        idx += 1

        if idx >= len(shape_classes):
            break

plt.tight_layout()
plt.show()

# =========================================================
# CHAIN CODE DEMONSTRATION
# =========================================================

print("\n=================================================")
print("CHAIN CODE DEMONSTRATION")
print("=================================================")

sample_img = generate_shape("square", rotation=45)

contours, _ = cv2.findContours(
    sample_img,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
)

contour = contours[0][:,0,:]

chain8 = freeman_chain_code_8dir(contour)

chain4 = freeman_chain_code_4dir(contour)

norm_chain = normalize_chain_code(chain8)

print("\nFirst 30 Chain Code 8-direction:")
print(chain8[:30])

print("\nFirst 30 Chain Code 4-direction:")
print(chain4[:30])

print("\nFirst 30 Normalized Chain Code:")
print(norm_chain[:30])

# =========================================================
# FOURIER RECONSTRUCTION
# =========================================================

img = generate_shape("triangle", rotation=30)

contours, _ = cv2.findContours(
    img,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
)

contour = contours[0][:,0,:]

z = contour[:,0] + 1j * contour[:,1]

fd_complex = np.fft.fft(z)

fig, axes = plt.subplots(1,4, figsize=(15,4))

axes[0].imshow(img, cmap='gray')
axes[0].set_title("Original")

for i, coeff in enumerate([5,10,20]):

    x_rec, y_rec = reconstruct_shape(fd_complex, coeff)

    recon = np.zeros((200,200), dtype=np.uint8)

    pts = np.column_stack([
        x_rec.astype(int),
        y_rec.astype(int)
    ])

    cv2.polylines(recon, [pts], True, 255, 1)

    axes[i+1].imshow(recon, cmap='gray')

    axes[i+1].set_title(f"{coeff} Descriptors")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()