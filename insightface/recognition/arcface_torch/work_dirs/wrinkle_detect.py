from PIL import Image
import cv2
import time
import numpy as np
from skimage.filters import frangi, gabor
from skimage import measure, morphology
import dlib

def master_control(image):

    b, g, r = cv2.split(image)  # image

    sk_frangi_img = frangi(g, sigmas=np.linspace(0, 1, num=100), beta=1.5, gamma=0.01)
    sk_frangi_img = morphology.closing(sk_frangi_img, morphology.disk(1))
    sk_gabor_img_1, sk_gabor_1 = gabor(g, frequency=0.35, theta=0)
    sk_gabor_img_2, sk_gabor_2 = gabor(g, frequency=0.35, theta=45)
    sk_gabor_img_3, sk_gabor_3 = gabor(g, frequency=0.35, theta=90)
    sk_gabor_img_4, sk_gabor_4 = gabor(g, frequency=0.35, theta=360)
    sk_gabor_img_1 = morphology.opening(sk_gabor_img_1, morphology.disk(2))
    sk_gabor_img_2 = morphology.opening(sk_gabor_img_2, morphology.disk(1))
    sk_gabor_img_3 = morphology.opening(sk_gabor_img_3, morphology.disk(2))
    sk_gabor_img_4 = morphology.opening(sk_gabor_img_4, morphology.disk(2))
    all_img = cv2.add(0.1 * sk_gabor_img_2, 0.9 * sk_frangi_img)
    all_img = morphology.closing(all_img, morphology.disk(1))
    _, all_img = cv2.threshold(all_img, 0.3, 1, 0)
    img1 = all_img

    bool_img = all_img.astype(bool)
    label_image = measure.label(bool_img)
    count = 0

    for region in measure.regionprops(label_image):
        if region.area < 10: #   or region.area > 700
            x = region.coords
            for i in range(len(x)):
                all_img[x[i][0]][x[i][1]] = 0
            continue
        if region.eccentricity > 0.98:
            count += 1
        else:
            x = region.coords
            for i in range(len(x)):
                all_img[x[i][0]][x[i][1]] = 0

    skel, distance = morphology.medial_axis(all_img.astype(int), return_distance=True)
    skels = morphology.closing(skel, morphology.disk(1))
    trans1 = skels
    return skels, count  # np.uint16(skels.astype(int))


def face_wrinkle(path, backage):
    result = cv2.imread(path)
    predictor = dlib.shape_predictor(backage)
    rect = dlib.rectangle(0, 0, result.shape[1], result.shape[0])
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, rect)
    landmarks = np.array([[pt.x, pt.y] for pt in shape.parts()])

    hull = cv2.convexHull(landmarks)
    hull_points = hull.reshape(-1, 2)

    min_y = np.min(hull_points[:, 1])
    offset = int(0.1 * result.shape[0])

    adjusted_hull = np.array(hull_points, dtype=np.int32)
    adjusted_hull = cv2.convexHull(adjusted_hull)

    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, adjusted_hull, 255)

    face_extracted = cv2.bitwise_and(result, result, mask=mask)
    img, count = master_control(result)
    print(img.astype(float))
    result[img > 0.1] = 255
    cv2.imshow("result", img.astype(float))
    cv2.waitKey(0)


if __name__ == '__main__':
    path = r"Asian_celebrity_align/KimSung-kyum/3_KimSung-kyum_71_m.jpg"
    backage = r'shape_predictor_81_face_landmarks.dat'

    face_wrinkle(path, backage)
