import cv2
import numpy as np


#resmi çağırma imread->resmi oku
img = cv2.imread("pizza.jpg")

#print(img)

# resmin eni ve boyu 
img_width = img.shape[1]
img_height= img.shape[0]

# print(img_height)
# print(img_width)

#resmi işlem yapabilmek ölçeklendirir,boyut ayarlaması, renk düzenlemesi yapar (BGR-RGB)
img_blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB= True)

#tespit edilebilecek nesne sınıfları
labels =["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
          "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
          "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
          "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
          "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
          "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]

# tespitleri renkli dörtgenlerle gösterir
colors=["255,255,0", "0,255,0","255,0,255","0,255,255","0,0,255"]
#dizideki elemanları sırayla int e cevirir
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors =np.array(colors)
#colors elemanlarını cogaltır
colors = np.tile(colors, (20,1))


#yolo dahil etme cfg-weights
model = cv2.dnn.readNetFromDarknet("yolo_model/yolov3.cfg","yolo_model/yolov3.weights")

#katmanları listeler 
layers=model.getLayerNames()
#çıkış katmanları
output_layer = [layers[i - 1] for i in model.getUnconnectedOutLayers().flatten()]

#modelin giriş verisi olarak ayarlar
model.setInput(img_blob)

#model çıkısından gelen  tahmin sonuclarını ıcerir
detection_layers=model.forward(output_layer)

id_list=[]
boxes_list=[]
confidence_list=[]


for detection_layer in detection_layers:
    for object_detection in detection_layer:
        #tahminlerden ilk 5ini alır ilk 5 indis nesne koordinatlarını içerir
        scores = object_detection [5:]
        #en en büyük skora sahip index
        predicted_id=np.argmax(scores)
        #en büyük skoru güven skoru olarak alır
        confidence = scores[predicted_id]

        #güven aralıgı
        if confidence > 0.80:
            #en yüksek skora sahip sınıf etiketi
            label = labels [predicted_id]
            #görüntü boyutu 
            bounding_box = object_detection[0:4] *np.array([img_width,img_height,img_width,img_height])
            #köşe koordinatları
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")


            #kutunun sağ alt koşe koordinatları
            start_x =int(box_center_x - (box_width/2))
            start_y = int(box_center_y - (box_height/2))


            id_list.append(predicted_id)
            confidence_list.append(float(confidence))
            boxes_list.append([start_x,start_y,int(box_width), int(box_height)])

# aynı nesneleri 1 kez cerceveler
max_ids = cv2.dnn.NMSBoxes(boxes_list,confidence_list, 0.5, 0.4)
#geçerli ve en az 1 oge içreen dizi oldugunu kontrol eder
if isinstance(max_ids, np.ndarray) and len(max_ids) > 0:
    #dizi döngusu
    for max_id in max_ids.flatten():
        box = boxes_list[max_id]

        start_x = box[0]
        start_y = box[1]

        box_width = box[2]
        box_height = box[3]

        predicted_id = id_list[max_id]
        label = labels[predicted_id]
        confidence = confidence_list[max_id]

        end_x = start_x + box_width
        end_y = start_y + box_height

        #tahmin sınıfına bağlı renk
        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]

        #kutu çizme
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), box_color, 2)
        #etiket yazma
        cv2.putText(img, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

cv2.imshow("Tespit Ekrani", img)

cv2.waitKey(0)  # Bir tuşa basana kadar bekler
cv2.destroyAllWindows()