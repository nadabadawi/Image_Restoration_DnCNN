import glob
from PIL import Image


#****************************RGB*****************************
def resize_img(img):
    image = Image.open(img)
    height = image.size[0]
    width = image.size[1]
    # print("Height: ", height, " W: ", width)
    #**************** Change to 2040 when grayscale *********************
    right  = (1020 - width)/2
    # print("Right: ", right)
    left = (1020 - width)/2
    # print("Right: ", right, " L: ", left)
    top = (1020 - height)/2
    bottom = (1020 - height)/2

    #width, height = image.size
    new_width = width + right + left
    new_height = height + top + bottom
    print("Mode: ", image.mode)
    result = Image.new(image.mode, (int(new_width), int(new_height)), (255, 255, 255))
    result.paste(image, (int(left), int(top)))
    return result

def datagenerator(data_dir='data/Train400',verbose=False):
    
    file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files

    for i in range(len(file_list)):
        result = resize_img(file_list[i])
        result.save('data/trained_data/train400_Coloured_'+ str(i) + '.png')
        print("image printed in data/trained_data/train400_OUT", i, ".png")

datagenerator()
