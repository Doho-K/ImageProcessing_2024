import cv2
import numpy as np
import tensorflow as tf
import random
import math

# MoveNet 모델 로드 (TFLite Interpreter)
model_path = "4.tflite"  # 경로를 본인의 MoveNet TFLite 모델 파일로 변경하세요.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 크기 및 타입 확인
input_shape = input_details[0]['shape']  # [1, 192, 192, 3] 예상
input_dtype = input_details[0]['dtype']  # np.uint8 또는 np.float32
input_size = (input_shape[1], input_shape[2])

# 신체 연결 정의
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

# 파티클 클래스 정의
class Particle:
    def __init__(self, x, y, size, speed):
        self.x = x
        self.y = y
        self.size = size
        self.speed = speed
        self.color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))  # 밝은 색상

    def move(self, height, keypoints, connections, avoidance_radius=50):
        """
        입자를 아래로 이동시키되, 키포인트 또는 연결된 선 근처에서 회피 행동 수행.
        """
        for (start_idx, end_idx) in connections:
            if keypoints[start_idx][2] < 0.1 or keypoints[end_idx][2] < 0.1:  # 신뢰도 낮은 점 무시
                continue

            # 선분의 시작점과 끝점
            x1, y1 = int(keypoints[start_idx][1] * width), int(keypoints[start_idx][0] * height)
            x2, y2 = int(keypoints[end_idx][1] * width), int(keypoints[end_idx][0] * height)

            # 점과 선분 사이의 거리 계산
            distance = self.point_line_distance(self.x, self.y, x1, y1, x2, y2)
            if distance < avoidance_radius:
                # 회피 행동: 랜덤하게 좌우로 퍼지기
                self.x += random.choice([-10, 10])
                self.y -= random.choice([5, 10])  # 위로 살짝 이동
                return

        self.y += self.speed
        if self.y > height:  # 화면을 벗어나면 다시 위로 이동
            self.y = 0
            self.x = random.randint(0, width)

    def point_line_distance(self, px, py, x1, y1, x2, y2):
        """
        점(px, py)와 선(x1, y1)-(x2, y2) 사이의 최소 거리 계산.
        """
        if x1 == x2 and y1 == y2:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)  # 선분이 점인 경우
        u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
        u = max(min(u, 1), 0)  # u를 [0, 1] 범위로 제한
        closest_x = x1 + u * (x2 - x1)
        closest_y = y1 + u * (y2 - y1)
        return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, self.color, -1)

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 화면 크기 가져오기
ret, frame = cap.read()
if not ret:
    print("웹캠에서 프레임을 읽을 수 없습니다.")
    exit()

height, width, _ = frame.shape

# 파티클 리스트 생성
num_particles = 150
particles = [Particle(random.randint(0, width), random.randint(0, height // 2), random.randint(2, 5), random.randint(1, 3)) for _ in range(num_particles)]

def preprocess(frame, input_size, input_dtype):
    """
    이미지를 MoveNet의 입력 형식에 맞게 전처리합니다.
    """
    image = cv2.resize(frame, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if input_dtype == np.float32:
        image = image.astype(np.float32) / 255.0
    elif input_dtype == np.uint8:
        image = image.astype(np.uint8)
    image = np.expand_dims(image, axis=0)
    return image

def draw_keypoints_and_connections(frame, keypoints, connections, confidence_threshold=0.3):
    """
    키포인트와 연결선을 이미지 위에 그립니다.
    """
    h, w, _ = frame.shape

    # 연결선 그리기
    for (start_idx, end_idx) in connections:
        if keypoints[start_idx][2] > confidence_threshold and keypoints[end_idx][2] > confidence_threshold:
            x1, y1 = int(keypoints[start_idx][1] * w), int(keypoints[start_idx][0] * h)
            x2, y2 = int(keypoints[end_idx][1] * w), int(keypoints[end_idx][0] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 키포인트 그리기
    for kp in keypoints:
        y, x, confidence = kp
        if confidence > confidence_threshold:
            cv2.circle(frame, (int(x * w), int(y * h)), 5, (0, 255, 0), -1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 프레임을 읽을 수 없습니다.")
        break

    # MoveNet 입력 전처리
    input_image = preprocess(frame, input_size, input_dtype)
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    # 모델 출력
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    keypoints = keypoints[0, 0, :, :]  # [1, 1, 17, 3] -> [17, 3]

    # 파티클 업데이트 및 그리기
    for particle in particles:
        particle.move(height, keypoints, connections, avoidance_radius=50)  # 키포인트와 연결선 회피
        particle.draw(frame)

    # 키포인트와 연결선 그리기
    draw_keypoints_and_connections(frame, keypoints, connections, confidence_threshold=0.3)

    # 결과 출력
    cv2.imshow("MoveNet with Particles and Skeleton", frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
