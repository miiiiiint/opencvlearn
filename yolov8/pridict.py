import cv2  
import numpy as np
import pyautogui
from ultralytics import YOLO
  


def screenshot():
    # 设置视频捕获窗口的大小，可以根据需要进行调整
    capture_width, capture_height = 1280, 720 
 
    # 获取屏幕截图
    screenshot = pyautogui.screenshot()
 
    # 将截图转换为OpenCV图像
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
 
    # 调整捕获窗口的大小
    frame = cv2.resize(frame, (capture_width, capture_height))

    return frame


model = YOLO('yolov8n.pt')

while True:
    image = screenshot()
    results = model(image)
    for r in results:
        #用古法列表处理
        xyxy = r.boxes.xyxy.tolist()  
        conf = r.boxes.conf.tolist()  
        sort = r.boxes.cls.tolist() 
        names = r.names #这里是字典
        
        for i in range(len(xyxy)):
            xmin, ymin, xmax, ymax = xyxy[i]
            confidence = conf[i]
            category = sort[i]
            if confidence > 0.4:
            
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                #做个自适应文字大小
                box_width = xmax - xmin
                box_height = ymax - ymin
                canzhao = min(box_width, box_height) / 100
                font_scale = 0.3 * canzhao
                chuxi = 2
                high = int(6.8*canzhao)
                if font_scale < 0.5:
                    font_scale =0.5
                    chuxi = 1
                print(canzhao)
                # 添加类别和置信度标签
                label = f"{names[category]}: {confidence:.2f}"
                print(label)
                cv2.putText(image, label, (int(xmin), (int(ymin) + high)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), chuxi)
        cv2.imshow("YOLOv8 Inference", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
out.release()  
