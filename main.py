import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


def preprocess_images(folder_0, folder_1, divisor):
    """
    두 폴더의 이미지를 처리하여 크기를 조정하고 RGB 값을 추출한 후 numpy 배열로 반환합니다.

    Args:
        folder_0 (str): 레이블 0인 이미지가 있는 폴더 경로
        folder_1 (str): 레이블 1인 이미지가 있는 폴더 경로
        divisor (int): 이미지 크기를 나눌 값

    Returns:
        tuple: (X, y) - X는 이미지 데이터 배열, y는 레이블 배열
    """
    X, y = [], []

    def process_folder(folder, label):
        files = [f for f in os.listdir(folder) if f.endswith(".png")]
        total_files = len(files)
        print(f"\n[{folder}] 처리 시작 - 총 {total_files}개 이미지")

        for idx, filename in enumerate(files, 1):
            if idx % 100 == 0:  # 100개마다 진행상황 출력
                print(f"진행률: {idx}/{total_files} ({idx / total_files * 100:.1f}%)")

            img_path = os.path.join(folder, filename)
            img = load_img(img_path)
            width, height = img.size
            img = img.resize((width // divisor, height // divisor))
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            y.append(label)

        print(f"[{folder}] 처리 완료")

    process_folder(folder_0, 0)  # 레이블 0 이미지 처리
    process_folder(folder_1, 1)  # 레이블 1 이미지 처리

    return np.array(X), np.array(y)


def balance_data(X, y):
    """
    데이터 불균형을 해결하기 위해 SMOTE를 사용하여 오버샘플링을 수행합니다.

    Args:
        X (numpy.ndarray): 이미지 데이터
        y (numpy.ndarray): 레이블 데이터

    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("\nSMOTE 오버샘플링 시작...")
    total_samples = len(X)
    print(f"처리할 샘플 수: {total_samples}")

    X_flat = X.reshape(len(X), -1)  # 2D 배열로 변환
    smote = SMOTE(sampling_strategy="auto", random_state=42, n_jobs=-1)  # n_jobs=-1로 모든 CPU 코어 사용

    with tqdm(total=100, desc="SMOTE 처리 중") as pbar:
        X_resampled, y_resampled = smote.fit_resample(X_flat, y)
        pbar.update(100)

    print(f"SMOTE 처리 완료. 생성된 샘플 수: {len(X_resampled)}")

    return X_resampled.reshape(-1, X.shape[1], X.shape[2], X.shape[3]), y_resampled


def build_model(input_shape):
    """
    개선된 CNN 모델: Batch Normalization 추가, Dropout 감소

    Args:
        input_shape (tuple): 입력 이미지의 형태

    Returns:
        model: 컴파일된 Keras 모델
    """
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.2),  # 드롭아웃 추가
            Conv2D(64, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),  # 드롭아웃 추가
            Flatten(),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.4),  # 드롭아웃 증가
            Dense(2, activation="softmax"),
        ]
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

    return model


# 사용 예시
if __name__ == "__main__":
    start_time = time.time()
    print("트립틱 이미지 분류 모델 학습을 시작합니다...")

    folder_0 = input("레이블 0 이미지가 있는 폴더 경로를 입력하세요: ").strip('"')
    folder_1 = input("레이블 1 이미지가 있는 폴더 경로를 입력하세요: ").strip('"')
    divisor = int(input("이미지 크기를 나눌 값을 입력하세요 (예: 2는 크기를 1/2로 줄임): "))

    print("\n이미지 전처리 시작...")
    preprocess_start = time.time()
    X, y = preprocess_images(folder_0, folder_1, divisor)
    y = to_categorical(y, num_classes=2)
    preprocess_time = time.time() - preprocess_start
    print(f"전처리 완료 - 총 {len(X)}개 이미지 (소요시간: {preprocess_time:.2f}초)")

    # 데이터 균형 조정 (SMOTE 사용)
    print("\n데이터 균형 조정 중...")
    X, y = balance_data(X, np.argmax(y, axis=1))  # SMOTE는 one-hot encoding이 아니라 정수형 레이블을 필요로 함
    y = to_categorical(y, num_classes=2)  # 다시 one-hot encoding으로 변환

    # 클래스 가중치 계산
    print("\n클래스 가중치 계산 중...")
    class_weights = compute_class_weight("balanced", classes=np.unique(np.argmax(y, axis=1)), y=np.argmax(y, axis=1))
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"계산된 클래스 가중치: {class_weights_dict}")

    # 모델 구축
    model = build_model(input_shape=X.shape[1:])

    print("\n모델 학습 시작...")
    training_start = time.time()
    model.fit(
        X,
        y,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        class_weight=class_weights_dict,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )
    training_time = time.time() - training_start
    print(f"모델 학습 완료 (소요시간: {training_time:.2f}초)")

    # 모델 저장 경로 설정
    model_save_path = f"trained_model_{datetime.now().strftime('%Y%m%d')}_{len(os.listdir()) + 1:03d}.keras"

    print("\n모델 저장 중...")
    model.save(model_save_path)
    print(f"모델이 다음 경로에 저장되었습니다: {model_save_path}")

    total_time = time.time() - start_time
    print(f"\n전체 작업 완료 (총 소요시간: {total_time:.2f}초)")
