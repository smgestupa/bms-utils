# %%
# import modules
import cv2
import easyocr
import re
import sys
from pyzbar.pyzbar import decode

# %%
def decode_image(img):
    return decode(img)[0].data.encode("utf-8")

# %%
def extract_from_image(img):
    # upscale image
    img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    # apply adaptive thresholding
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 8)

    # apply morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # downscale image
    img = cv2.resize(img, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_CUBIC)

    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(img, detail=1, paragraph=True)

    pattern = "((?:[\dX]{13})|(?:[\d\-\_X]{17})|(?:[\dX]{10})|(?:[\d\-\_X]{13}))"
    code = "-1"
    for _, text in results:
        if re.search(pattern, text):
            code = re.search(pattern, text).group(1)
            code = re.sub("([-_])", "", code)
            break
    
    return code

# %%
# save given image and convert to grayscale
img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

# %%
extracted = "-1"
try: 
    extracted = decode_image(img)
except:
    extracted = extract_from_image(img)

print(extracted)


