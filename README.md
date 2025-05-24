This is a simple and powerful web application that allows users to upload brain MRI scans and receive instant predictions on the type of brain tumor using deep learning models built with PyTorch and TensorFlow.

The app uses Flask for the backend and provides a clean user interface for uploading images and viewing predictions.

```
brain-tumor-classifier/
├── main.py                     # Flask application
├── models/
│   ├── cnn.py                  # PyTorch CNN model and Tensorflow model definition
├── templates/
│   └── index.html              # Web frontend
├── static/
│   └── traitementcancerducerveaucerebrale.jpg          # Background image
├── my_thiemokho_model.torch    # Pre-trained PyTorch model
├── my_thiemokho_model.keras    # Pre-trained TensorFlow model
└── README.md
```
✅ Upload brain MRI images (JPG, PNG, etc.)

✅ Choose between PyTorch and TensorFlow models

✅ Get classification results: Glioma, Meningioma, No Tumor, Pituitary

✅ Real-time progress bar during prediction

✅ Background image & styled frontend

✅ Flash error handling
