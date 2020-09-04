import os
import PIL
from PIL import Image
import cv2

folder = "./corgi_images2/not_corgi/"
for file in os.listdir(folder):
  img = cv2.imread(os.path.join(folder, file))
  im = Image.open(folder + file)
  print(im)
  im.convert('RGB').save(folder + file, "JPEG")

