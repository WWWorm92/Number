import os
import re
import cv2
import pytesseract
import numpy as np

# Установите путь к файлу tesseract.exe в переменной PATH_TO_TESSERACT
PATH_TO_TESSERACT = r"/usr/bin/tesseract"

# Загрузите Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSERACT

# Загрузите классификатор Хаара для распознавания номеров автомобилей
car_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_russian_plate_number.xml")

# Загрузите видео с камеры видеонаблюдения
video = cv2.VideoCapture("rtsp://admin:1qazxsw2@192.168.1.4/doc/page/preview.asp")
#video = cv2.VideoCapture(0)

# Создайте папку для сохранения изображений с номерами автомобилей
if not os.path.exists("img"):
    os.makedirs("img")

while True:
    # Считайте кадр из видео
    ret, frame = video.read()
    if not ret:
        break

    # Преобразуйте изображение в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружьте номера автомобилей на кадре
    cars = car_cascade.detectMultiScale(gray, 1.1, 4)

    # Распознайте номера автомобилей с помощью Tesseract OCR
    for (x, y, w, h) in cars:
        car_image = gray[y:y + h, x:x + w]

        # Выпрямите номерной знак на изображении
        rect = cv2.minAreaRect(cv2.findNonZero(car_image))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(car_image, M, (width, height))

        # Распознайте номер на выпрямленном изображении
        car_number = pytesseract.image_to_string(warped, lang="rus", config="--psm 7")
        car_number = re.sub(r'\W+', '', car_number)

        # Сохраните изображение с номером в папку img
        if car_number:
            cv2.imwrite("img/{}.png".format(car_number), warped)

    # Нарисуйте обнаруженные номера на кадре
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Отобразите кадр с обнаруженными номерами автомобилей
    cv2.imshow("Car Detection", frame)

    # Нажмите клавишу "q", чтобы остановить выполнение программы
    if cv2.waitKey(1) == ord('q'):
        break

# Освободите ресурсы
video.release()
cv2.destroyAllWindows()