import cv2
import numpy as np
import os

SCALE = 3
THICK = 5
WHITE = (255, 255, 255)
DIRECTORY = os.path.abspath(".")


def generate_digits() -> list[np.ndarray]:
    digits = []
    for digit in map(str, range(10)):
        (width, height), bline = cv2.getTextSize(digit, cv2.FONT_HERSHEY_SIMPLEX,
                                                 SCALE, THICK)
        digits.append(np.zeros((height + bline, width), np.uint8))
        cv2.putText(digits[-1], digit, (0, height), cv2.FONT_HERSHEY_SIMPLEX,
                    SCALE, WHITE, THICK)
        x0, y0, w, h = cv2.boundingRect(digits[-1])
        digits[-1] = digits[-1][y0:y0 + h, x0:x0 + w]

    return digits


def rgb_to_gray(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    return color_image, gray_image


def to_thresh(gray_image: np.ndarray):
    _, thresh = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY_INV)
    return thresh


def ml_detect(img, digits: list[np.ndarray]) -> int:
    percent_white_pix = 0
    digit = -1
    for i, d in enumerate(digits):
        scaled_img = cv2.resize(img, d.shape[:2][::-1])

        bitwise = cv2.bitwise_and(d, cv2.bitwise_xor(scaled_img, d))

        before = np.sum(d == 255)
        matching = 100 - (np.sum(bitwise == 255) / before * 100)


        if percent_white_pix < matching:
            percent_white_pix = matching
            digit = i

    return digit


def load_dl_requirement() -> tuple:
    import tensorflow as tf

    model_path = os.path.join(DIRECTORY, "model/tarzan_model.h5")
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        raise Exception(f"this {model_path=} does't exists")

    classes_path = os.path.join(DIRECTORY, "model/classes")
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as classes_file:
            classes = [classe.strip("\n")
                            for classe in classes_file.readlines()]
    else:
        raise Exception(f"This {classes_path=} does't exists\n directory not found")

    return model, classes


def dl_detect(img: np.ndarray) -> str:
    model, classes = load_dl_requirement()
    prediction = model.predict(img)
    classe_of_prediction = np.argmax(prediction, axis=1)[0]
    return classes[classe_of_prediction]


def main(image_path: str, option: str) -> None:
    digits = generate_digits()
    color_image, gray_image = rgb_to_gray(image_path)
    thresh = to_thresh(gray_image)

    countours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    index = 0
    for cnt in countours:
        if cv2.contourArea(cnt) > 30:
            brect = cv2.boundingRect(cnt)

            x, y, w, h = brect
            roi = thresh[y: y+h, x:x+w]
            if option == "ml":
                digit = ml_detect(roi, digits)
            elif option == "dl":
                roi = color_image[y: y+h, x:x+w]
                roi_resize = cv2.resize(roi, (32, 32))
                roi_reshape = roi_resize.reshape((-1, 32, 32, 3))
                digit = dl_detect(roi_reshape)
            else:
                raise Exception("your option should 'dl' or 'ml'")
            index += 1
            cv2.rectangle(color_image, brect, (0,255,0), 2)
            cv2.putText(color_image, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (190, 123, 68), 2)

    cv2.imshow("result", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, nargs='?')
    parser.add_argument('--option', type=str, nargs='?')
    args = parser.parse_args()
    image_path = args.image
    option = args.option
    if image_path is not None and option is not None:
        main(image_path, option)
