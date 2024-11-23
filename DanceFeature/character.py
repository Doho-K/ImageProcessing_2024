import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

# MoveNet Thunder 모델 로드
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

# 관절 연결 정의
connections = [
    # 머리
    (0, 1), (0, 2), (1, 3), (2, 4),
    # 팔
    (5, 7), (7, 9), (6, 8), (8, 10),
    # 몸통
    (5, 6), (5, 11), (6, 12), (11, 12),
    # 다리
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def preprocess_frame(frame):
    input_size = (256, 256)  # Thunder 모델 입력 크기
    resized = cv2.resize(frame, input_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0).astype(np.int32)

def draw_human_character(frame, keypoints, connections, threshold=0.1):
    """
    주어진 connections 정의를 사용해 사람 같은 캐릭터를 생성합니다.
    """
    height, width, _ = frame.shape
    keypoints_xy = []

    # 키포인트 좌표 변환 및 필터링
    for kp in keypoints:
        y, x, confidence = kp
        if confidence > threshold:
            x = int(x * width)
            y = int(y * height)
            keypoints_xy.append((x, y))
        else:
            keypoints_xy.append(None)

    # 머리 (큰 원)
    if keypoints_xy[0]:
        cv2.circle(frame, keypoints_xy[0], 30, (0, 255, 255), -1)  # 노란색 머리

    # 몸통 (타원)
    if keypoints_xy[5] and keypoints_xy[6] and keypoints_xy[11] and keypoints_xy[12]:
        # 어깨와 엉덩이 좌표 계산
        shoulder_x = (keypoints_xy[5][0] + keypoints_xy[6][0]) // 2
        shoulder_y = (keypoints_xy[5][1] + keypoints_xy[6][1]) // 2
        hip_x = (keypoints_xy[11][0] + keypoints_xy[12][0]) // 2
        hip_y = (keypoints_xy[11][1] + keypoints_xy[12][1]) // 2

        # 몸통 중심과 크기 계산
        center_x = (shoulder_x + hip_x) // 2
        center_y = (shoulder_y + hip_y) // 2
        width_body = abs(keypoints_xy[5][0] - keypoints_xy[6][0]) + 40  # 어깨 너비 기준
        height_body = abs(shoulder_y - hip_y) + 50  # 어깨와 엉덩이 거리 기준

        # 타원으로 몸통 그리기
        cv2.ellipse(frame, (center_x, center_y), (width_body // 2, height_body // 2), 0, 0, 360, (0, 0, 255), -1)

    # 팔 (굵은 선)
    for i, j in [(5, 7), (7, 9), (6, 8), (8, 10)]:
        if keypoints_xy[i] and keypoints_xy[j]:
            start_x, start_y = keypoints_xy[i]
            end_x, end_y = keypoints_xy[j]
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 20)

    # 다리 (굵은 선)
    for i, j in [(11, 13), (13, 15), (12, 14), (14, 16)]:
        if keypoints_xy[i] and keypoints_xy[j]:
            start_x, start_y = keypoints_xy[i]
            end_x, end_y = keypoints_xy[j]
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 20)

    # 손과 발 (작은 원)
    for i in [9, 10, 15, 16]:
        if keypoints_xy[i]:
            cv2.circle(frame, keypoints_xy[i], 10, (128, 0, 128), -1)


# 동영상 경로 및 저장 디렉토리 설정
video_path = "test2.mp4"  # 동영상 파일 경로
output_dir = "output_images2"  # 저장 디렉토리
os.makedirs(output_dir, exist_ok=True)

# 동영상 읽기
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("동영상을 열 수 없습니다.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 1)  # 5초마다 처리
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_interval == 0:
        input_image = preprocess_frame(frame)
        outputs = movenet.signatures['serving_default'](tf.constant(input_image, dtype=tf.int32))
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
        
        # 캐릭터 이미지 저장
        character_frame = np.ones_like(frame) * 255  # 빈 캔버스
        draw_human_character(character_frame, keypoints, connections)
        character_image_path = os.path.join(output_dir, f"character_{frame_count}.jpg")
        cv2.imwrite(character_image_path, character_frame)

        print(f"Saved character image at frame {frame_count}")

    frame_count += 1

cap.release()
print("동영상 처리가 완료되었습니다.")
