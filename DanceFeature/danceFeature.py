import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

# MoveNet Lightning 모델 로드
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")


# 관절 연결 정의
connections = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

def preprocess_frame(frame):
    """
    Thunder 모델 입력 형식에 맞게 전처리합니다.
    """
    input_size = (256, 256)  # Thunder 모델 입력 크기
    resized = cv2.resize(frame, input_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0).astype(np.int32)  # int32로 변환



def draw_keypoints_and_connections(frame, keypoints, connections, threshold=0.1):
    """
    키포인트와 관절 연결선을 프레임 위에 시각화합니다.
    """
    height, width, _ = frame.shape
    keypoints_xy = []

    # 키포인트 좌표 변환 및 시각화
    for kp in keypoints:
        y, x, confidence = kp
        if confidence > threshold:  # 신뢰도 조건 확인
            # 정규화된 좌표를 픽셀 좌표로 변환
            x = int(x * width)
            y = int(y * height)
            keypoints_xy.append((x, y))
            
            # 초록색 원으로 키포인트 표시
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        else:
            keypoints_xy.append(None)

    # 관절 연결선 그리기
    for connection in connections:
        start_idx, end_idx = connection
        if keypoints_xy[start_idx] and keypoints_xy[end_idx]:
            # 파란색 선으로 연결선 표시
            cv2.line(frame, keypoints_xy[start_idx], keypoints_xy[end_idx], (255, 0, 0), 2)

def calculate_confidence_mean(keypoints):
    """
    키포인트 신뢰도의 평균을 계산합니다.
    """
    confidences = [kp[2] for kp in keypoints]
    mean_confidence = sum(confidences) / len(confidences)
    return mean_confidence


# 동영상 경로 및 저장 디렉토리 설정
video_path = "test2.mp4"  # 동영상 파일 경로
output_dir = "output_images2"  # 저장 디렉토리
os.makedirs(output_dir, exist_ok=True)

# 동영상 읽기
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("동영상을 열 수 없습니다.")
    exit()

# 동영상 정보 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 5)  # 5초에 해당하는 프레임 간격
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 5초마다 프레임 처리
    if frame_count % frame_interval == 0:
        # 모델 입력 전처리
        input_image = preprocess_frame(frame)

        # MoveNet 모델 실행
        outputs = movenet.signatures['serving_default'](tf.constant(input_image, dtype=tf.int32))
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]  # [1, 1, 17, 3] -> [17, 3]

        # 신뢰도 평균 계산
        confidence_mean = calculate_confidence_mean(keypoints)
        confidence_mean_str = f"{confidence_mean:.2f}"  # 소수점 2자리까지 표시

        # 원본 프레임 저장
        original_image_path = os.path.join(output_dir, f"frame_{frame_count}_conf_{confidence_mean_str}.jpg")
        cv2.imwrite(original_image_path, frame)

        # 키포인트가 시각화된 프레임 저장
        annotated_frame = frame.copy()
        draw_keypoints_and_connections(annotated_frame, keypoints, connections)
        
        # 저장된 이미지 경로
        annotated_image_path = os.path.join(output_dir, f"keypoints_{frame_count}_conf_{confidence_mean_str}.jpg")
        cv2.imwrite(annotated_image_path, annotated_frame)

        print(f"Saved frame and keypoints at frame {frame_count} with mean confidence {confidence_mean_str}")

    frame_count += 1

cap.release()
print("동영상 처리가 완료되었습니다.")
