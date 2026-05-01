import coremltools as ct
import tensorflow as tf
from preprocess import CLASSES

print(f"TF Version: {tf.__version__}")
model_dir = "../emotion_saved_model"

model = tf.saved_model.load(model_dir)
# Extract the default concrete function
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

image_input = ct.ImageType(
    name="keras_tensor", # The name of the input tensor in the concrete function
    shape=(1, 64, 64, 3),
    scale=1.0 / 255.0,
    color_layout=ct.colorlayout.RGB,
    channel_first=False,
)

classifier = ct.ClassifierConfig(CLASSES)

print("Converting...")
try:
    mlmodel = ct.convert(
        [concrete_func],
        source="tensorflow",
        inputs=[image_input],
        classifier_config=classifier,
        convert_to="neuralnetwork"
    )
except Exception as e:
    print("neuralnetwork failed:", e)
    mlmodel = ct.convert(
        [concrete_func],
        source="tensorflow",
        inputs=[image_input],
        classifier_config=classifier,
        convert_to="neuralnetwork"
    )

mlmodel.save("../EmotionClassifier.mlmodel")
print("Saved to EmotionClassifier.mlmodel")
