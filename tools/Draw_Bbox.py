#批量生成bbox
import cv2
imput_path = 'submission/'
image_path = 'images/'
new_image_path = './outputimage/'
color = [(0, 0, 255),(0, 255, 0),(255, 0, 0)]

for i in range(7500):
    num_mark = 42500+i
    f = open(imput_path + str(num_mark) + ".txt")
    lines = f.readlines()
    img = cv2.imread(image_path + str(num_mark) + ".jpg")
    print(img.shape)

    if (len(lines) == 0):
        cv2.imwrite(new_image_path+str(num_mark)+'.jpg', img)
        continue
    p=0
    for line in lines:
        line = line.split()
        x_min = int(line[2])
        y_min = int(line[3])
        x_max = int(line[4])
        y_max = int(line[5])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color[p%3], 2)
        text = line[1]
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        cv2.putText(img, str(text), (x_min, y_min), font, 1, color[p%3], 2)
        p=p+1
    f.close()
    cv2.imwrite(new_image_path+str(num_mark)+'.jpg', img)
