import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy import interpolate
imput_path = '30/submission/'
gt_path = 'groundtruth/'


def compute_ap(rec,prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    #这里面的前提是把R和P按从小到大顺序拍好
    #横坐标为mrec

    # compute the precision envelope
    #插值
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_iou(rec1, rec2, S_rec1, S_rec2):
    """
    computing IoU
    (x0,y0,x1,y1)
    """
    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0,0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0, intersect

def main_compute(total_num,IOU_thread):
    rec_final = []
    prec_final = []
    sum_coordinate = []
    for i in range(20):

        cnfd_thd = i / 20.0
        iou_thd = IOU_thread
        avg_rec = 0
        avg_prec = 0

        for ii in range(total_num):

            f1 = open(imput_path + str(ii) + ".txt")
            f2 = open(gt_path + str(ii) + ".txt")
            lines = f1.readlines()
            comp_lines = f2.readlines()
            if len(comp_lines) == 0:
                continue
            if len(lines) == 0:
                continue
            # 打开两个文件，如果groundtruth没有就跳过
            pred_boxes = []
            comp_boxes = []
            for line in lines:
                line = line.split()
                pred_boxes.append(list(map(float, line[1:6])))
            pred_boxes = sorted(pred_boxes, key=lambda x: x[0], reverse=True)
            # 得到排好序(从大到小)的矩形框数据[confidence，x，y，x，y]

            for line in comp_lines:
                line = line.split()
                comp_boxes.append(list(map(float, line[1:5])))

            sum_rec1 = 0.0
            sum_rec2 = 0.0
            sum_intersection = 0.0

            intersection = 0.0
            flag = 0

            # 一张图就在同一类别怎么做
            for data in pred_boxes:
                if data[0] < cnfd_thd:
                    continue
                data = data[1:5]
                S_rec1 = (data[2] - data[0]) * (data[3] - data[1])
                intersectionfinal = 0.0
                iou_final = 0.0
                for data2 in comp_boxes:
                    S_rec2 = (data2[2] - data2[0]) * (data2[3] - data2[1])
                    iou, intersection = compute_iou(data, data2, S_rec1, S_rec2)

                    if iou >= iou_thd:
                        if iou_final < iou:
                            intersectionfinal = intersection
                            iou_final = iou

                    if flag == 0:
                        sum_rec2 = sum_rec2 + S_rec2

                flag = 1

                sum_intersection = sum_intersection + intersectionfinal
                sum_rec1 = sum_rec1 + S_rec1

            if sum_rec1 == 0:
                avg_prec = avg_prec + 1
                continue

            rec_1 = sum_intersection / sum_rec2 * 1.0
            prec_1 = sum_intersection / sum_rec1 * 1.0
            avg_rec = avg_rec + rec_1
            avg_prec = avg_prec + prec_1

        avg_rec = avg_rec / total_num
        avg_prec = avg_prec / total_num
        # rec_final.append(avg_rec)
        # prec_final.append(avg_prec)
        sum_coordinate.append([avg_rec, avg_prec])
        # 下一步算所有大于cnfd的图像面积和

    sum_coordinate_final = sorted(sum_coordinate)
    for line in sum_coordinate_final:
        rec_final.append(line[0])
    for line in sum_coordinate_final:
        prec_final.append(line[1])

    return  rec_final, prec_final

plt.figure(figsize=(6,4))
for i in range(4):
    i=i/10
    rec_final, prec_final = main_compute(42499, 0.5+i)
    print(rec_final)
    print(prec_final)
    final_result = compute_ap(rec_final,prec_final)
    print(final_result)
    #rec_final = rec_final[0:8]
    #prec_final = prec_final[0:8]
    #x_smooth=[]
    #for i in range(7):
    #    line_smooth = np.linspace(rec_final[i], rec_final[i+1], 10)
    #    for i in range(9):
    #        x_smooth.append(line_smooth[i])
    #x_smooth.append(rec_final[7])
    #func = interpolate.interp1d(rec_final, prec_final, kind='quadratic')
    #cubic,quadratic, slinear
    #y_smooth = make_interp_spline(rec_final, prec_final)(x_smooth)
    #y_smooth = func(x_smooth)
    #plt.plot(rec_final,prec_final,label="$AP@"+str(0.5+i),linewidth=2)
    plt.plot(rec_final,prec_final,label="PR@"+str(0.5+i),linewidth=2)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()


