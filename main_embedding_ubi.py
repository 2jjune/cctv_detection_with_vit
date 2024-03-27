from tensorflow_docs.vis import embed
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
from Swin_Transformer_TF.swintransformer import SwinTransformer
from focal_loss import BinaryFocalLoss
from vit_keras import vit
import tfimm
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")


MAX_SEQ_LENGTH = 150
NUM_FEATURES = 768
IMG_SIZE = 800

EPOCHS = 300
base_dir = 'D:/Jiwoon/dataset/UBI_FIGHTS/vit/'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# center_crop_layer = layers.CenterCrop(IMG_SIZE, IMG_SIZE)
# def crop_center(frame):
#     cropped = center_crop_layer(frame[None, ...])
#     cropped = cropped.numpy().squeeze()
#     return cropped
def augment_video(frames):
    # Convert frames to tensors
    frames_tensor = tf.convert_to_tensor(frames)
    seed = (tf.random.uniform([], 0, 1000000, dtype=tf.int32), tf.random.uniform([], 0, 1000000, dtype=tf.int32))

    # Define the data augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        layers.Rescaling(scale=1. / 255.),
        layers.RandomFlip('horizontal', seed=seed),
        layers.RandomBrightness(factor=0.2, seed=seed),
        layers.RandomContrast(factor=0.2, seed=seed),
    ])

    # Apply augmentation
    augmented_frames = data_augmentation(frames_tensor)

    return augmented_frames.numpy()

def extend_frames(frames, target_length):
    if not frames:
        return [], []
    extended_frames = frames.copy()

    while len(extended_frames) < target_length:
        # 현재 frames의 마지막부터 처음까지 역순으로 반복하는 index를 계산합니다.
        # 예를 들어, frames가 [1, 2, 3, 4, 5] 라면 [4, 3, 2, 1] 순서로 index를 추가합니다.
        for idx in range(len(frames) - 2, -1, -1):  # Reflect from the end
            extended_frames.append(frames[idx])
            if len(extended_frames) == target_length:
                break
        if len(extended_frames) < target_length:
            for idx in range(1, len(frames)):  # Reflect from the start
                extended_frames.append(frames[idx])
                if len(extended_frames) == target_length:
                    break
    return extended_frames

def load_video(path, max_frames=MAX_SEQ_LENGTH, annotations_dir="D:/Jiwoon/dataset/UBI_FIGHTS/annotation"):
    cap = cv2.VideoCapture(path)
    frames = []
    frame_count = 0
    try:
        if 'nonfight' in path:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % 4 == 0:
                    frame = cv2.resize(frame,(IMG_SIZE,IMG_SIZE))
                    # frame = crop_center(frame)
                    frame = frame[:, :, [2, 1, 0]]
                    frames.append(frame)

                if len(frames) == max_frames:
                    break
                frame_count += 1

            if len(frames)<max_frames:
                frames = extend_frames(frames,max_frames)
            # frames = augment_video(frames)
        elif 'fight' in path:
            filename = os.path.basename(path).replace('.mp4', '.csv')
            csv_path = os.path.join(annotations_dir, filename)
            annotations = pd.read_csv(csv_path, header=None)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if annotations.iloc[frame_count, 0] == 1:
                    if frame_count % 2 == 0:
                        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                        # frame = crop_center(frame)
                        frame = frame[:, :, [2, 1, 0]]
                        frames.append(frame)

                    if len(frames) == max_frames:
                        break
                    frame_count += 1

            if len(frames) < max_frames:
                frames = extend_frames(frames, max_frames)
            # frames = augment_video(frames)
    finally:
        cap.release()
    frames = np.stack(frames) / 255.0
    return np.array(frames)


def build_feature_extractor():
    '''['ConvNeXtBase', 'ConvNeXtLarge', 'ConvNeXtSmall', 'ConvNeXtTiny', 'ConvNeXtXLarge',
    'DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
    'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2B0',
     'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M', 'EfficientNetV2S',
     'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2', 'MobileNetV3Large', 'MobileNetV3Small', 'NASNetLarge',
     'NASNetMobile', 'RegNetX002', 'RegNetX004', 'RegNetX006', 'RegNetX008', 'RegNetX016', 'RegNetX032', 'RegNetX040',
     'RegNetX064', 'RegNetX080', 'RegNetX120', 'RegNetX160', 'RegNetX320', 'RegNetY002', 'RegNetY004', 'RegNetY006',
     'RegNetY008', 'RegNetY016', 'RegNetY032', 'RegNetY040', 'RegNetY064', 'RegNetY080', 'RegNetY120', 'RegNetY160',
     'RegNetY320', 'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'ResNet50', 'ResNet50V2', 'ResNetRS101',
     'ResNetRS152', 'ResNetRS200', 'ResNetRS270', 'ResNetRS350', 'ResNetRS420', 'ResNetRS50', 'VGG16', 'VGG19',
     'Xception', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__',
     '__path__', '__spec__', '_sys', 'convnext', 'densenet', 'efficientnet', 'efficientnet_v2', 'imagenet_utils',
     'inception_resnet_v2', 'inception_v3', 'mobilenet', 'mobilenet_v2', 'mobilenet_v3', 'nasnet', 'regnet',
      'resnet', 'resnet50', 'resnet_rs', 'resnet_v2', 'vgg16', 'vgg19', 'xception']'''
    # feature_extractor = SwinTransformer('swin_tiny_224', include_top=False, pretrained=True)
    # print(123123123123123123123123123,tfimm.list_models(pretrained='timm'))
    # feature_extractor = tfimm.create_model("convnext_large", pretrained="timm", nb_classes=0)
    feature_extractor = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    # feature_extractor = vit.vit_b16(
    #     image_size=IMG_SIZE,
    #     activation='sigmoid',
    #     pretrained=True,
    #     include_top=False,
    #     pretrained_top=False,
    #     classes=1)
    feature_extractor.summary()
    # feature_extractor = keras.applications.DenseNet121(
    #     weights="imagenet",
    #     include_top=False,
    #     pooling="avg",
    #     input_shape=(IMG_SIZE, IMG_SIZE, 3),
    # )
    # preprocess_input = keras.applications.densenet.preprocess_input
    # preprocess_input = keras.applications.resnet50.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    # preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(inputs)
    print('before',99999999999999999999999999,outputs.shape)
    # global NUM_FEATURES
    # NUM_FEATURES = outputs.shape[-1]
    if outputs.shape[-1] != NUM_FEATURES:
        outputs = layers.Dense(NUM_FEATURES, activation='linear', trainable=False)(outputs)
    print("NUM_FEATURESNUM_FEATURESNUM_FEATURES", NUM_FEATURES)

    print(99999999999999999999999999,outputs.shape)

    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()


# Label preprocessing with StringLookup.
# label_processor = keras.layers.StringLookup(
#     num_oov_indices=0, vocabulary=np.unique(train_df["tag"]), mask_token=None
# )
# print(label_processor.get_vocabulary())


def prepare_all_videos(root_dir):
    # num_samples = len(df)
    # video_paths = df["video_name"].values.tolist()
    # labels = df["tag"].values
    # labels = label_processor(labels[..., None]).numpy()
    #
    # # `frame_features` are what we will feed to our sequence model.
    # frame_features = np.zeros(
    #     shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    # )
    video_paths = []
    labels = []
    classes = ["fight", "nonfight"]
    for class_name in classes:
        class_dir = os.path.join(root_dir, class_name)
        for video_name in os.listdir(class_dir):
            video_paths.append(os.path.join(class_dir, video_name))
            labels.append(class_name)
    num_samples = len(video_paths)
    # labels = label_processor(np.array(labels)[..., None]).numpy()
    labels = [0 if label == "fight" else 1 for label in labels]
    labels = np.array(labels)

    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))

        # Pad shorter videos.
        # if len(frames) < MAX_SEQ_LENGTH:
        #     diff = MAX_SEQ_LENGTH - len(frames)
        #     padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
        #     frames = np.concatenate(frames, padding)

        frames = frames[None, ...]

        # Initialize placeholder to store the features of the current video.
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                if np.mean(batch[j, :]) > 0.0:
                    temp_frame_features[i, j, :] = feature_extractor.predict(
                        batch[None, j, :], verbose=0
                    )

                else:
                    temp_frame_features[i, j, :] = 0.0
                if "train" in root_dir:
                    progress_str = f"Training: Processing frame {j + 1}/{length} of video {idx + 1}/{num_samples}"
                    print(progress_str, end='\r')
                else:
                    progress_str = f"Val: Processing frame {j + 1}/{length} of video {idx + 1}/{num_samples}"
                    print(progress_str, end='\r')
                # print(progress_str + ' ' * (100 - len(progress_str)), end='')
                # print(progress_str, end='\n')
                # progress_str = f"\rProcessing frame {j + 1}/{length} of video {idx + 1}/{num_samples}"
                # print(progress_str + ' ' * (100 - len(progress_str)), end='')
                # print(f"\rProcessing frame {j + 1}/{length} of video {idx + 1}/{num_samples}", end='')
            if "train" in root_dir and idx == len(video_paths) - 1:
                print()
        frame_features[idx,] = temp_frame_features.squeeze()

    return frame_features, labels


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask


class RelativePositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, embed_dim):
        super(RelativePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.rel_embeddings = self.add_weight("rel_embeddings", shape=[sequence_length, embed_dim])

    def call(self, x):
        rel_positions = self.rel_embeddings
        # 상대적 위치 정보를 입력 텐서의 차원과 일치하게 만듭니다.
        rel_positions = tf.expand_dims(rel_positions, axis=0)  # Shape: (1, sequence_length, embed_dim)
        # 상대적 위치 정보를 각 샘플에 대해 복제합니다.
        rel_positions = tf.repeat(rel_positions, repeats=tf.shape(x)[0], axis=0)
        return x + rel_positions[:, :tf.shape(x)[1], :self.embed_dim]  # Truncate to match the feature dimension


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.dropout_layer = layers.Dropout(0.23)
        self.attention = layers.MultiHeadAttention(
            # num_heads=num_heads, key_dim=embed_dim, dropout=0.3
            num_heads = num_heads, key_dim = embed_dim, dropout = 0.23
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation='relu'),#tf.nn.gelu
             layers.Dropout(0.23),
             layers.Dense(1024, activation='relu'),
             layers.Dropout(0.3),
             layers.Dense(2048, activation='relu'),
             layers.Dropout(0.3),
             layers.Dense(embed_dim, activation='relu'),
             ]
        )
        self.dense_proj2 = keras.Sequential(
            [layers.Dense(3072, activation='relu')]
        )
        self.dense_proj3 = keras.Sequential(
            [layers.Dense(embed_dim, activation='relu')]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        # attention_output = self.attention(inputs, inputs, attention_mask=mask)
        # proj_input = self.layernorm_1(inputs + attention_output)
        # attention_output = self.dropout_layer(proj_input)
        #
        # output = self.layernorm_2(attention_output)
        # output = self.dense_proj2(output)
        # output = self.dropout_layer(output)
        # output = self.dense_proj3(output)
        # output = self.dropout_layer(output)
        # return output

        inputs = self.layernorm_1(inputs)
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        # proj_input = self.layernorm_1(inputs + attention_output)
        proj_input = self.dropout_layer(attention_output)
        proj_output = self.dense_proj(proj_input)
        output = self.layernorm_2(proj_input + proj_output)
        output = self.dense_proj2(output)
        output = self.dropout_layer(output)
        return self.dense_proj3(output)


class TemporalPositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, feature_dim, time_dim, **kwargs):
        super(TemporalPositionalEmbedding, self).__init__(**kwargs)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=feature_dim)
        self.time_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=time_dim)
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.time_dim = time_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)

        # Positional Embedding
        pos_embeddings = self.position_embeddings(positions)
        print('pos_embeddings.shape',pos_embeddings.shape)
        # Time Embedding
        time_embeddings = self.time_embeddings(positions)
        print('time_embeddings.shape',time_embeddings.shape)

        return inputs + pos_embeddings + time_embeddings

class Attention(tf.keras.layers.Layer):
    def __init__(self, name="attention_layer", **kwargs):
        super(Attention, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer="random_normal",
                                 name="attention_W",
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer="zeros",
                                 name="attention_b",
                                 trainable=True)

    def call(self, x):
        q = tf.nn.tanh(tf.linalg.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(q, axis=1)
        output = a * x
        return output

class BiLSTMWithAttention(tf.keras.layers.Layer):
    def __init__(self, units, name="bilstm_attention_layer", **kwargs):
        super(BiLSTMWithAttention, self).__init__(name=name, **kwargs)
        self.units = units
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.units, return_sequences=True, name="bilstm_layer", dropout=0.21),
            name="bidirectional_layer"
        )
        self.attention = Attention(name="attention_layer")

    def call(self, inputs):
        x = self.bilstm(inputs)
        x = self.attention(x)
        return x
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value = inputs['query'], inputs['key'], inputs['value']
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        matmul_qk = tf.matmul(query, key, transpose_b=True)
        attention_weights = tf.nn.softmax(matmul_qk, axis=-1)
        # attention_weights = tf.nn.sigmoid(matmul_qk)
        output = tf.matmul(attention_weights, value)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output

class BiLSTMWithMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, units, num_heads, d_model):
        super(BiLSTMWithMultiHeadAttention, self).__init__()
        self.units = units
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.units, return_sequences=True)
        )
        self.multi_head_attention = MultiHeadAttention(num_heads, d_model)
        self.batchnom = layers.BatchNormalization()
    def call(self, inputs):
        x = self.bilstm(inputs)
        x = self.multi_head_attention({
            'query': x,
            'key': x,
            'value': x
        })
        return x

def get_compiled_model():
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = 1024
    num_heads = 8
    classes = 1
    TIME_DIM = 64

    print("NUM_FEATURESNUM_FEATURESNUM_FEATURES", NUM_FEATURES)
    # inputs = keras.Input(shape=(None, None))
    inputs = keras.Input(shape=(None, NUM_FEATURES))
    # x = BiLSTMWithAttention(512, name="bilstm_attention_layer")(inputs)
    x = BiLSTMWithMultiHeadAttention(1024, 16, NUM_FEATURES)(inputs)
    # x = layers.Bidirectional(layers.LSTM(256, dropout=0.2, return_sequences=True))(inputs)
    relative_positional_encoding = RelativePositionalEncoding(sequence_length, x.shape[-1])
    x = relative_positional_encoding(x)

    # x = PositionalEmbedding(
    #     sequence_length, x.shape[-1], name="frame_position_embedding"
    # )(x)
    # x = PositionalEmbedding(
    #     sequence_length, embed_dim, name="frame_position_embedding"
    # )(x)
    x = TransformerEncoder(x.shape[-1], dense_dim, num_heads, name="transformer_layer")(x)
    # x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.42)(x)
    outputs = layers.Dense(classes, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    learning_rate = 1e-4
    weight_decay = 0.0001
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimizer,
        # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        loss=BinaryFocalLoss(gamma=2),
        metrics=[
            # keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            # keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    # model.compile(
    #     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    # )
    return model

def plot_metrics(history, metric, filepath):
    # Plot training & validation metric values
    plt.figure(figsize=(10, 6))
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'Model {metric}')
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    plt.savefig(os.path.join(filepath, f"{metric}.png"))
    plt.close()

def plot_roc_curve(labels, predictions, filepath):
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(filepath, "roc_auc_curve.png"))
    plt.close()

def run_experiment():
    filepath = "./tmp/video_classifier"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath+'/ubi_lstm_transformer.h5', monitor="val_auc", save_weights_only=True, save_best_only=True, verbose=1, mode='max'
        # 'C:/Users/user/PycharmProjects/cctv_transformer_demo/ubi_mobile_lstm_transformer.h5', monitor = "val_auc", save_weights_only = True, save_best_only = True, verbose = 1, mode = 'max'
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=30, restore_best_weights=True, verbose=1, mode='max'
    )
    # Reduce learning rate on plateau callback
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc', factor=0.1, patience=15, min_lr=1e-6, verbose=1, mode='max'
    )

    model = get_compiled_model()
    history = model.fit(
        train_frame_features,
        train_labels,
        # validation_split=0.15,
        validation_data=(val_frame_features, val_labels),
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping, reduce_lr],
    )
    plot_metrics(history, 'accuracy', filepath)
    plot_metrics(history, 'loss', filepath)
    plot_metrics(history, 'auc', filepath)
    predictions = model.predict(val_frame_features)
    print(val_labels.shape)
    print(predictions.shape)
    misclassified_data = [
        (i, round(pred[0]), true) for i, (true, pred) in enumerate(zip(val_labels, predictions))
        if round(pred[0]) != true
    ]

    misclassified_df = pd.DataFrame(
        misclassified_data, columns=["Misclassified Indices", "Predicted Value", "True Value"]
    )
    misclassified_df.to_csv(filepath + "/dilated_misclassified_indices.csv", index=False)

    model.load_weights(filepath+'/ubi_lstm_transformer.h5')
    # model.load_weights('C:/Users/user/PycharmProjects/cctv_transformer_demo/ubi_mobile_lstm_transformer.h5')
    _, accuracy, auc = model.evaluate(val_frame_features, val_labels)
    # 예측값 얻기
    predictions = model.predict(val_frame_features)
    # ROC-AUC 곡선 그리기
    plot_roc_curve(val_labels, predictions, filepath)

    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test AUC: {round(auc, 4)}")
    return model

print('/' * 100)
print(dir(vit))
train_frame_features, train_labels = prepare_all_videos(train_dir)
np.save('./150_mobilev2_ubi_train_frame_features.npy', train_frame_features)
np.save('./150_mobilev2_ubi_train_labels.npy', train_labels)
val_frame_features, val_labels = prepare_all_videos(val_dir)
np.save('./150_mobilev2_ubi_val_frame_features.npy', val_frame_features)
np.save('./150_mobilev2_ubi_val_labels.npy', val_labels)

train_frame_features = np.load('./vit_ubi_train_frame_features.npy')
train_labels = np.load('./vit_ubi_train_labels.npy')
val_frame_features = np.load('./vit_ubi_val_frame_features.npy')
val_labels = np.load('./vit_ubi_val_labels.npy')
print()
print('features and shape')
print(train_frame_features.shape, train_labels.shape)

trained_model = run_experiment()



