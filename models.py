"""
Deep Learning Models for Action Classification
Implements multiple architectures for comparison
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import (
    ResNet50, VGG16, EfficientNetB0, MobileNetV2, InceptionV3
)
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten,
    GlobalAveragePooling2D, GlobalAveragePooling1D, BatchNormalization, Activation,
    MultiHeadAttention, LayerNormalization, Add
)
import config


class ActionClassificationModels:
    """
    Collection of deep learning models for action classification
    """
    
    def __init__(self, input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS), 
                 num_classes=config.NUM_CLASSES):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_custom_cnn(self, model_name="Custom_CNN"):
        """
        Build a custom CNN architecture from scratch
        """
        model = models.Sequential(name=model_name)
        
        # Block 1
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block 2
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block 3
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Block 4
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Dense layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def build_resnet50(self, trainable_layers=50):
        """
        Build ResNet50 with transfer learning
        """
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers (fine-tuning option)
        if trainable_layers == 0:
            base_model.trainable = False
        else:
            # Make only the last n layers trainable
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='ResNet50_Transfer')
        
        return model
    
    def build_vgg16(self, trainable_layers=5):
        """
        Build VGG16 with transfer learning
        """
        # Load pre-trained VGG16
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        if trainable_layers == 0:
            base_model.trainable = False
        else:
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='VGG16_Transfer')
        
        return model
    
    def build_efficientnet(self, trainable_layers=50):
        """
        Build EfficientNetB0 with transfer learning
        """
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        if trainable_layers == 0:
            base_model.trainable = False
        else:
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='EfficientNetB0_Transfer')
        
        return model
    
    def build_mobilenet(self, trainable_layers=30):
        """
        Build MobileNetV2 with transfer learning (lightweight model)
        """
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        if trainable_layers == 0:
            base_model.trainable = False
        else:
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='MobileNetV2_Transfer')
        
        return model
    
    def build_inception_v3(self, trainable_layers=50):
        """
        Build InceptionV3 with transfer learning
        """
        # Load pre-trained InceptionV3
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        if trainable_layers == 0:
            base_model.trainable = False
        else:
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        # Build model
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='InceptionV3_Transfer')
        
        return model
    
    def build_vision_transformer(self, patch_size=16, num_heads=8, 
                                 transformer_layers=4, mlp_dim=512):
        """
        Build a simplified Vision Transformer (ViT) model
        """
        inputs = keras.Input(shape=self.input_shape)
        
        # Create patches
        num_patches = (self.input_shape[0] // patch_size) * (self.input_shape[1] // patch_size)
        patch_dim = patch_size * patch_size * self.input_shape[2]
        
        # Extract patches
        patches = layers.Reshape((num_patches, patch_dim))(
            layers.Conv2D(patch_dim, patch_size, strides=patch_size, padding='valid')(inputs)
        )
        
        # Patch embedding
        projection_dim = 256
        x = layers.Dense(projection_dim)(patches)
        
        # Add position embedding
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )(positions)
        x = x + position_embedding
        
        # Transformer blocks
        for _ in range(transformer_layers):
            # Layer normalization 1
            x1 = LayerNormalization(epsilon=1e-6)(x)
            
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=projection_dim // num_heads,
                dropout=0.1
            )(x1, x1)
            
            # Skip connection 1
            x2 = Add()([attention_output, x])
            
            # Layer normalization 2
            x3 = LayerNormalization(epsilon=1e-6)(x2)
            
            # MLP
            mlp_output = layers.Dense(mlp_dim, activation='gelu')(x3)
            mlp_output = layers.Dropout(0.1)(mlp_output)
            mlp_output = layers.Dense(projection_dim)(mlp_output)
            mlp_output = layers.Dropout(0.1)(mlp_output)
            
            # Skip connection 2
            x = Add()([mlp_output, x2])
        
        # Classification head
        x = LayerNormalization(epsilon=1e-6)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.3)(x)
        x = Dense(mlp_dim, activation='gelu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='Vision_Transformer')
        
        return model
    
    def compile_model(self, model, learning_rate=config.LEARNING_RATE):
        """
        Compile model with optimizer, loss, and metrics
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def get_model_summary(self, model):
        """
        Get detailed model summary
        """
        print(f"\n{'='*80}")
        print(f"Model: {model.name}")
        print(f"{'='*80}")
        model.summary()
        
        # Count parameters
        trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        
        print(f"\n{'='*80}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print(f"{'='*80}\n")


def main():
    """Test model building"""
    model_builder = ActionClassificationModels()
    
    # Build and compile different models
    print("Building Custom CNN...")
    custom_cnn = model_builder.build_custom_cnn()
    custom_cnn = model_builder.compile_model(custom_cnn)
    model_builder.get_model_summary(custom_cnn)
    
    print("\nBuilding ResNet50...")
    resnet = model_builder.build_resnet50(trainable_layers=0)
    resnet = model_builder.compile_model(resnet)
    model_builder.get_model_summary(resnet)


if __name__ == "__main__":
    main()
