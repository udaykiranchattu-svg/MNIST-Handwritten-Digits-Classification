# MNIST Handwritten Digits Classification

A deep learning project that uses **TensorFlow and Keras** to recognize handwritten digits from **0 to 9** using the MNIST dataset.

The project builds a simple neural network, trains it on MNIST handwritten digit images, evaluates its performance, visualizes training results, predicts test images, and performs an experiment by changing the number of neurons in the hidden layer.

## 📌 Project Overview

Handwritten digit recognition is a common beginner-level computer vision and deep learning problem.

In this project, a neural network is trained to classify grayscale handwritten digit images into one of ten classes:

**0, 1, 2, 3, 4, 5, 6, 7, 8, 9**

The MNIST dataset contains images of handwritten digits with an image size of **28 × 28 pixels**.

## 🎯 Objectives

* Load and explore the MNIST dataset.
* Display sample handwritten digit images.
* Normalize image pixel values.
* Build a neural network using TensorFlow/Keras.
* Train the model to classify digits from 0–9.
* Evaluate the model using test accuracy.
* Visualize training and validation accuracy.
* Visualize training and validation loss.
* Compare actual and predicted labels.
* Predict user-uploaded handwritten images.
* Experiment with a different number of hidden-layer neurons.

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **NumPy**
* **Matplotlib**
* **Pillow (PIL)**
* **Google Colab**

## 📂 Dataset

The project uses the built-in **MNIST handwritten digits dataset** provided by TensorFlow/Keras.

The dataset contains:

* Training images
* Training labels
* Testing images
* Testing labels
* 10 digit classes
* Image size: **28 × 28 pixels**

The dataset is loaded using:

```python
tf.keras.datasets.mnist.load_data()
```

## 🧠 Neural Network Architecture

The original model uses the following architecture:

```text
Input: 28 × 28 image
        ↓
Flatten
        ↓
Dense Layer - 128 neurons
        ↓
ReLU Activation
        ↓
Dropout - 20%
        ↓
Dense Layer - 10 neurons
        ↓
Softmax
        ↓
Predicted Digit
```

### Model Code

```python
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(10, activation="softmax")
])
```

The model is compiled using:

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

## 🔄 Data Preprocessing

The MNIST images contain pixel values between **0 and 255**.

The images are normalized to a range between **0 and 1**:

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```

This helps the neural network train more effectively.

## 🏋️ Model Training

The original model is trained with:

* **Epochs:** 10
* **Batch Size:** 32
* **Validation Split:** 10%
* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy

```python
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)
```

## 📊 Model Evaluation

After training, the model is evaluated using the MNIST test dataset.

```python
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)
```

The project displays:

* Test Loss
* Test Accuracy
* Test Accuracy Percentage

```python
print("Test Accuracy:", test_accuracy)
print("Test Accuracy Percentage:", test_accuracy * 100)
```

> **Note:** The exact accuracy depends on the execution environment and training run. Run the notebook to obtain the actual result for your experiment.

## 📈 Training Visualization

The project generates graphs for:

### Training and Validation Accuracy

The accuracy curves help visualize how the model performs during training.

### Training and Validation Loss

The loss curves show how the model's error changes during training.

These graphs are generated using Matplotlib.

## 🔢 Digit Prediction

The trained model predicts the digits in the MNIST test dataset.

```python
predictions = model.predict(x_test, verbose=0)

predicted_labels = np.argmax(predictions, axis=1)
```

The project displays:

```text
Actual labels
Predicted labels
```

It also visually displays the actual and predicted values for sample images.

## ✍️ Handwritten Image Prediction

The project also allows users to upload their own handwritten digit images.

The uploaded images are preprocessed before prediction.

### Image Preprocessing Steps

1. Convert image to grayscale.
2. Invert the image.
3. Crop unnecessary background.
4. Resize the digit while maintaining proportions.
5. Create a 28 × 28 canvas.
6. Center the digit.
7. Normalize pixel values.
8. Send the processed image to the neural network.

The preprocessing function is:

```python
def preprocess_image(file_data):
    img = Image.open(
        io.BytesIO(file_data)
    ).convert("L")

    img = ImageOps.invert(img)

    bbox = img.getbbox()

    if bbox:
        img = img.crop(bbox)

    img.thumbnail((20, 20))

    canvas = Image.new("L", (28, 28), 0)

    x = (28 - img.width) // 2
    y = (28 - img.height) // 2

    canvas.paste(img, (x, y))

    image_array = np.array(canvas)
    image_array = image_array.astype("float32") / 255.0

    return image_array
```

The actual digit is taken from the uploaded filename.

For example:

```text
5.png
```

means the actual digit is considered to be **5**.

The project then compares the actual digit with the predicted digit and displays whether the prediction is **Correct** or **Wrong**.

## 🧪 Experiment

An additional experiment is performed by changing the number of neurons in the hidden layer.

### Original Model

```text
128 neurons
```

### Experimental Model

```text
64 neurons
```

The experimental model is:

```python
experiment_model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(10, activation="softmax")
])
```

Both models are trained for **10 epochs** with a batch size of **32**.

Their test accuracies are then compared.

## 📊 Model Comparison

The project compares:

| Model              | Hidden Neurons |
| ------------------ | -------------: |
| Original Model     |            128 |
| Experimental Model |             64 |

The notebook calculates the difference in accuracy:

```python
difference = (
    experiment_accuracy - test_accuracy
) * 100
```

A validation-accuracy graph is also generated to compare both models.

## 📁 Project Structure

```text
MNIST-Handwritten-Digits-Classification/
│
├── main.ipynb
└── README.md
```

## ▶️ How to Run

### Option 1: Google Colab

1. Open the repository.
2. Open `main.ipynb`.
3. Upload the notebook to Google Colab.
4. Run the cells from top to bottom.
5. When prompted, upload handwritten digit images.

### Option 2: Local Python Environment

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib pillow
```

Then run the notebook using Jupyter Notebook or JupyterLab.

```bash
jupyter notebook
```

## 📦 Requirements

```text
Python
TensorFlow
NumPy
Matplotlib
Pillow
Jupyter Notebook / Google Colab
```

## 💡 Key Learning Outcomes

Through this project, you can learn:

* Basics of image classification.
* Working with the MNIST dataset.
* Data normalization.
* Neural network architecture.
* Dense layers.
* ReLU activation.
* Softmax activation.
* Dropout.
* Model compilation.
* Model training.
* Model evaluation.
* Accuracy and loss visualization.
* Image preprocessing.
* Handwritten digit prediction.
* Basic model experimentation.

## 🔮 Future Improvements

The project can be further improved by:

* Using Convolutional Neural Networks (CNNs).
* Adding a confusion matrix.
* Calculating precision, recall, and F1-score.
* Adding more handwritten image tests.
* Creating a web interface using Streamlit or Flask.
* Saving and loading the trained model.
* Comparing different activation functions.
* Experimenting with different numbers of layers and epochs.

## 👨‍💻 Author

**Chattu Uday Kiran**

B.Tech – Computer Science and Engineering
Rai Technology University, Bengaluru

## 🔗 GitHub Repository

[MNIST Handwritten Digits Classification – GitHub](https://github.com/udaykiranchattu-svg/MNIST-Handwritten-Digits-Classification?utm_source=chatgpt.com)

## 📜 License

This project is created for **educational and academic purposes**.
