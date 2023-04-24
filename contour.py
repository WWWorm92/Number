import numpy as np
import cv2 as cv

hsv_min = np.array((98, 0, 142), np.uint8)
hsv_max = np.array((234, 155, 255), np.uint8)

if __name__ == '__main__':
    fn = 'test/test_i.jpg'  # имя файла, который будем анализировать
    img = cv.imread(fn)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # меняем цветовую модель с BGR на HSV
    thresh = cv.inRange(hsv, hsv_min, hsv_max)  # применяем цветовой фильтр
    contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.imshow('filter', thresh)
    # перебираем все найденные контуры в цикле
    for cnt in contours0:
        rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        cv.drawContours(img, [box], -1, (255, 0, 0), 0)  # рисуем прямоугольник

    cv.imshow('contours', img)  # вывод обработанного кадра в окно

    cv.waitKey()
    cv.destroyAllWindows()

