# Copyright 2017 Christoph Baur <c.baur@tum.de>
# ==============================================================================

def crop_center(img, cropx, cropy):
  y = img.shape[0]
  x = img.shape[1]
  startx = x//2-(cropx//2)
  starty = y//2-(cropy//2)
  if len(img.shape) > 2:
    return img[starty:starty+cropy, startx:startx+cropx,:]
  else:
    return img[starty:starty+cropy, startx:startx+cropx]

def crop(img, y, x, height, width):
  return img[y:y+height,x:x+width]