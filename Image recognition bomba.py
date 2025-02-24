import os


from PIL import Image

img = Image.open('image.jpg').convert('L')
img.save('greyscale.png')
image_folder1 = "dropbox"
image_folder2 = "Dropbox"
image_folder3 = "Camera Uploads"
image = "image.jpg"
image_path = os.path.join(os.getcwd(), image_folder1, image_folder2, image_folder3, image)
print(image_path)
test = Image.open(image_path)
test.show()