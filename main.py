import re

import cv2
import pytesseract
import os

# Установите путь к файлу tesseract.exe в переменной PATH_TO_TESSERACT
PATH_TO_TESSERACT = r"/usr/bin/tesseract"

# Загрузите Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = PATH_TO_TESSERACT

# Создайте папку "img", если ее нет
if not os.path.exists('img'):
    os.makedirs('img')

# Загрузите видео
#video = cv2.VideoCapture("rtsp://admin:1qazxsw2@192.168.1.4/doc/page/preview.asp")
video = cv2.VideoCapture(0)

# Загрузите классификатор Хаара для распознавания номеров автомобилей
car_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_russian_plate_number.xml")

count_num = 0

# Читайте кадры из видео
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Преобразуйте изображение в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружьте номера автомобилей на кадре
    cars = car_cascade.detectMultiScale(gray, 1.1, 4)

    # Читайте каждый найденный номер автомобиля
    for (x, y, w, h) in cars:
        # Обрежьте изображение, чтобы получить только номер
        car_img = gray[y:y + h, x:x + w]
        custom_config = r'--oem 3 --psm 6 -l eng+rus -c tessedit_char_whitelist=АВСЕНКМОРТХУABCEHKMOPTXy0123456789'
        # Распознайте номер с помощью Tesseract OCR
        number = pytesseract.image_to_string(car_img, config=custom_config)

        number = number.replace(" ", "")
        if len(number) >= 6 and re.match(r"[A-ZА-Яа-я]\d{3}[A-ZА-Яа-я]{2}.+", number):
            count_num += 1
            # Сохраните изображение с номером в папку img
            cv2.imwrite(f'img/{number}_{count_num}.jpg', car_img)

        # Отрисуйте прямоугольник вокруг номера на кадре
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Отобразите кадр с прямоугольниками вокруг номеров
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1000, 800)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освободите ресурсы
video.release()
cv2.destroyAllWindows()
