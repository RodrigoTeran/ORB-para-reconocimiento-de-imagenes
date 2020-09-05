import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("foto.jpeg", cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1, (500, 500), interpolation=cv2.INTER_CUBIC)

img2 = cv2.imread("foto2.jpeg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img2, (500, 500), interpolation=cv2.INTER_CUBIC)

# sift = cv2.xfeatures2d.SIFT_create()
# surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1000)

# keyPointsImg1, descriptorsImg1 = sift.detectAndCompute(img1, None)
# keyPointsImg2, descriptorsImg2 = sift.detectAndCompute(img2, None)
# keyPointsImg1, descriptorsImg1 = surf.detectAndCompute(img1, None)
# keyPointsImg2, descriptorsImg2 = surf.detectAndCompute(img2, None)
keyPointsImg1, descriptorsImg1 = orb.detectAndCompute(img1, None)
keyPointsImg2, descriptorsImg2 = orb.detectAndCompute(img2, None)

# Brute Force Matching
# cv2.NORM_HAMMING = usually with ORB detector
bruteForce = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bruteForce.match(descriptorsImg1, descriptorsImg2)

# Ordenarlo para tener los mejores primero
matches = sorted(matches, key=lambda x: x.distance)
matching_result = cv2.drawMatches(img1, keyPointsImg1, img2, keyPointsImg2, matches[:20], None, flags=2)
porcentaje = 100 - matches[20].distance
print("Porcentaje: ", porcentaje, "%")

# Draw result with matplotlib
plt.imshow(matching_result)
plt.show()

# img1 = cv2.drawKeypoints(img1, keyPoints, None)
# cv2.imshow("Image 1", img1)
# cv2.imshow("Image 2", img2)
# cv2.imshow("Matching Result", matching_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
