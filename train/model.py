import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout

def build_clcm_model(num_classes=5, input_shape=(96, 96, 3)):
    base = MobileNetV2(
        input_shape=input_shape,
        alpha=0.75,
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D(name='gap')(x)
    x = Dense(256, activation='relu', name='dense_256')(x)
    x = BatchNormalization(name='bn_256')(x)
    x = Dropout(0.4, name='dropout_256')(x)
    x = Dense(128, activation='relu', name='dense_128')(x)
    x = Dropout(0.3, name='dropout_128')(x)
    output = Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=base.input, outputs=output, name='CLCM_EmotionLens')
    print(f"Total params: {model.count_params():,}")
    print(f"Input name  : {model.input_names}")
    return model

if __name__ == '__main__':
    build_clcm_model()
