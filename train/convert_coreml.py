"""
Convert emotion_model_clcm_v3.h5 → EmotionClassifier.mlmodel
"""
import tensorflow as tf
import coremltools as ct

MODEL_PATH  = '/Users/ririputri/porto/train/emotion_model_clcm_v3.h5'
OUTPUT_PATH = '/Users/ririputri/porto/EmotionClassifier.mlpackage'
CLASS_LABELS = ['angry', 'happy', 'neutral', 'sad', 'surprise']

def convert():
    print("Loading model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    # STEP 1: Verifikasi input name
    input_name = model.input_names[0]
    print(f"Input name  : '{input_name}'")
    print(f"Input shape : {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # STEP 2: Test inference
    import numpy as np
    dummy = np.zeros((1, 96, 96, 3), dtype=np.float32)
    result = model.predict(dummy, verbose=0)
    assert result.shape == (1, 5), f"Output shape salah: {result.shape}"
    print(f"Test inference OK: {result}")

    # STEP 3: Convert
    print("\nConverting to Core ML...")
    image_input = ct.ImageType(
        name=input_name,
        shape=(1, 96, 96, 3),
        color_layout=ct.colorlayout.RGB,
        scale=1.0 / 255.0,
        bias=[0.0, 0.0, 0.0]
    )

    mlmodel = ct.convert(
        model,
        inputs=[image_input],
        classifier_config=ct.ClassifierConfig(CLASS_LABELS),
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL
    )

    # STEP 4: Metadata
    mlmodel.short_description = (
        "EmotionLens CLCM v3 — 5 emotions. "
        "MobileNetV2 alpha=0.75, 96x96 RGB, FER-2013. "
        "Ref: Gursesli et al., IEEE Access 2024."
    )
    mlmodel.author  = "EmotionLens — Apple Developer Academy Portfolio"
    mlmodel.version = "3.0"

    # STEP 5: Verifikasi spec
    spec = mlmodel.get_spec()
    print("\n── Core ML Spec ──")
    for inp in spec.description.input:
        print(f"  Input  → '{inp.name}'")
    for out in spec.description.output:
        print(f"  Output → '{out.name}'")

    # STEP 6: Save
    mlmodel.save(OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")
    print("\nNext steps:")
    print("1. Buka Xcode project EmotionLens")
    print("2. Drag EmotionClassifier.mlmodel ke Project Navigator")
    print("3. Centang 'Add to target: EmotionLens'")
    print("4. Klik mlmodel → tab Predictions → verifikasi input 96x96 RGB")
    print("5. Cmd+Shift+K → Cmd+B → Run di iPhone")

if __name__ == '__main__':
    convert()
