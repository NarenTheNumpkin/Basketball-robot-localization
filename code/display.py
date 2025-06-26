import cv2 as cv
import os

def display(delay = 100):
    image_folder = "/Users/naren/Desktop/Robocon_2025/dataset/images"
    image_files = sorted([f for f in os.listdir(image_folder)])

    while True:
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            img = cv.imread(img_path)
            if img is None:
                continue
            cv.imshow('Flipbook', img[200:])
            key = cv.waitKey(delay)
            if key == 27:  
                cv.destroyAllWindows()
                return

def test():
    image = cv.imread("/Users/naren/Desktop/Robocon_2025/dataset/images/img_20250622_024617_073870.jpg")
    print(image[200:].shape)
    cv.waitKey(0)

def main():
    test()

if __name__ == "__main__":
    main()