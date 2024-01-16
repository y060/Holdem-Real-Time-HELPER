import os
import cv2
import json
import numpy as np

import tkinter as tk
from PIL import Image, ImageTk

from ultralytics import YOLO
import supervision as sv
import torch


# Load a model
model = YOLO("./runs/detect/train_e150/weights/best.pt")  # load a custom model
saved_agent = torch.load("./rlcard/1A5R100W_64.pth", map_location=torch.device('cpu'))
model_forgame = saved_agent.q_estimator.qnet

APP = tk.Tk()
APP.title("Webcam")
APP.bind('<Escape>', lambda e: APP.quit()) 

cam = tk.Frame(APP, padx=10, pady=10)
show = tk.Frame(APP, padx=10, pady=10)
advise = tk.Frame(APP, padx=10, pady=10, bg='#d9d9d9')

PUBLICS = []
HANDS = []
ADVICE = []
ADVICE_TEXT = ["FOLD 棄牌", "CHECK_CALL 跟注", "RAISE_HALF_POT 加注1/2底池", "RAISE_POT 加注1底池", "ALL_IN 全下"]

write_publics3, write_publics1, write_hands = False, False, False
ORI_MEMORY = {"count": 0, "10C": [], "10D": [], "10H": [], "10S": [], "2C": [], "2D": [], "2H": [], "2S": [], "3C": [], "3D": [], "3H": [], "3S": [], "4C": [], "4D": [], "4H": [], "4S": [], "5C": [], "5D": [], "5H": [], "5S": [], "6C": [], "6D": [], "6H": [], "6S": [], "7C": [], "7D": [], "7H": [], "7S": [], "8C": [], "8D": [], "8H": [], "8S": [], "9C": [], "9D": [], "9H": [], "9S": [], "AC": [], "AD": [], "AH": [], "AS": [], "JC": [], "JD": [], "JH": [], "JS": [], "KC": [], "KD": [], "KH": [], "KS": [], "QC": [], "QD": [], "QH": [], "QS": []}
memory = ORI_MEMORY

# print(tk.font.families())


# --------------------------------------------------------------------------------
# 輸入 & 輸出


def writeCard(option):
    global write_publics3, write_publics1, write_hands
    
    if option == "p3":
        if not write_publics1 and not write_hands:
            write_publics3 = True
            
    elif option == "p1":
        if not write_publics3 and not write_hands and len(PUBLICS)<5:
            write_publics1 = True
            
    elif option == "h":
        if not write_publics3 and not write_publics1:
            write_hands = True

    
cw, ch = 100, 145

pubCanvas = tk.Canvas(show, width = cw*6, height = ch+50, highlightthickness=0)
pubCanvas.grid(column=0, row=0, sticky="ne", columnspan=2)
pubBtn1 = tk.Button(show, text = "Flop", command = lambda: writeCard("p3"))
pubBtn1.grid(column=0, row=1)
pubBtn3 = tk.Button(show, text = "翻開公共牌", command = lambda: writeCard("p1"))
pubBtn3.grid(column=1, row=1)

handCanvas = tk.Canvas(show, width = cw*6, height = ch+70, highlightthickness=0)
handCanvas.grid(column=0, row=2, sticky="ne", columnspan=2)
handBtn1 = tk.Button(show, text = "輸入手牌", command = lambda: writeCard("h"))
handBtn1.grid(column=0, row=3, columnspan=2)


card_images = {}
for filename in os.listdir("./poker-cards"):
    if filename[-5] != "_":
        img_path = os.path.join("./poker-cards", filename)
        img = Image.open(img_path)
        img = img.resize( (cw, ch) )
        image = ImageTk.PhotoImage(img)
        
        card_images[filename[:-4]] = image

public_text = "公共牌組"
pubCanvas.create_text(10, 10, text=public_text, anchor='nw', fill='#e67870', font=('思源宋體 TW', 12, 'bold'))

hand_text = "我的手牌"
handCanvas.create_text(10, 30, text=hand_text, anchor='nw', fill='#e67870', font=('思源宋體 TW', 12, 'bold'))


def showHands():
    handCanvas.create_image(15, 30+30, anchor="nw", image=card_images[HANDS[0]])
    handCanvas.create_image(20+cw, 30+30, anchor="nw", image=card_images[HANDS[1]])

def showPublics():
    y = 10+24+5
    for i in range(len(PUBLICS)):
        public = PUBLICS[i]
        pubCanvas.create_image(10 + (cw+5)*i, y, anchor="nw", image=card_images[public])


# --------------------------------------------------------------------------------
# 相機視窗


# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

camLabel = tk.Label(cam) 
camLabel.grid(column=0, row=0)

count_text = tk.IntVar()
count_text.set(0)
countText = tk.Label(cam, textvariable=count_text)
countText.grid(column=0, row=0, sticky="se")


def calcCard(memory, n):
    # 將平均值儲存起來
    average_memory = {}
    for key in memory:
        num = memory[key]
        if key != "count" and len(num) > 0 and key not in PUBLICS and key not in HANDS:
            average = sum(num) / len(num)
            average_memory[key] = average
    
    top_keys = sorted(average_memory, key=average_memory.get, reverse=True)[:n]
    
    return top_keys


def camera_stream():
    
    global PUBLICS, HANDS, memory, ORI_MEMORY
    global write_publics3, write_publics1, write_hands
    global count_text
    
    ret, original_frame = cap.read()
        
    results = model(original_frame, stream=False, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])
    
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id in zip(detections.confidence, detections.class_id)
    ]
    
    if write_publics3 or write_publics1 or write_hands:  
        memory["count"] += 1
        count_text.set(memory["count"])
        
        d = [[confidence, class_id]for confidence, class_id in zip(detections.confidence, detections.class_id)]
        for detection in d:
            memory[model.model.names[detection[1]]].append(detection[0])
        
        if memory["count"] >= 64:
            if write_publics3:
                PUBLICS = calcCard(memory, 3)
                showPublics()
                write_publics3 = False
                
            elif write_publics1:
                PUBLICS += calcCard(memory, 1)
                showPublics()
                write_publics1 = False
                
            elif write_hands:
                HANDS = calcCard(memory, 2)
                if len(HANDS) == 2:
                    showHands()
                write_hands = False
            
            memory = {"count": 0, "10C": [], "10D": [], "10H": [], "10S": [], "2C": [], "2D": [], "2H": [], "2S": [], "3C": [], "3D": [], "3H": [], "3S": [], "4C": [], "4D": [], "4H": [], "4S": [], "5C": [], "5D": [], "5H": [], "5S": [], "6C": [], "6D": [], "6H": [], "6S": [], "7C": [], "7D": [], "7H": [], "7S": [], "8C": [], "8D": [], "8H": [], "8S": [], "9C": [], "9D": [], "9H": [], "9S": [], "AC": [], "AD": [], "AH": [], "AS": [], "JC": [], "JD": [], "JH": [], "JS": [], "KC": [], "KD": [], "KH": [], "KS": [], "QC": [], "QD": [], "QH": [], "QS": []}
            count_text.set(memory["count"])
        
    frame = box_annotator.annotate(scene=original_frame, detections=detections, labels=labels)
    
    frame_cv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    frame_PIL = Image.fromarray(frame_cv2)
    frame_tk = ImageTk.PhotoImage(image = frame_PIL)
    
    camLabel.imgtk = frame_tk
    camLabel.config(image = frame_tk)
    cam.after(1, camera_stream)

camera_stream()


# --------------------------------------------------------------------------------
# 遊戲建議


def makeAdvice():
    global HANDS, PUBLICS, ADVICE
    global card_to_index
    # global myChip_val, maxChip_val
    global myChipText, maxChipText
    global a0, a1, a2, a3, a4, final_advice
    
    if len(HANDS) != 2:
        error = "手牌無輸入"
    else:
        state_vector = np.zeros(54)
        for card in HANDS:
            index = card_to_index[card]
            state_vector[index] = 1
        
        if len(PUBLICS) > 0:
            for card in PUBLICS:
                index = card_to_index[card]
                state_vector[index] = 1

        # state_vector[52] = myChip_val
        # state_vector[53] = maxChip_val 
        state_vector[52] = myChipText.get()
        state_vector[53] = maxChipText.get()
    
        state_tensor = torch.from_numpy(state_vector).float()
        state_tensor = state_tensor.unsqueeze(0)
        output = model_forgame(state_tensor)
        ADVICE = output.detach().numpy()[0]
        
        a0.set(f"FOLD 棄牌：{ADVICE[0]}")
        a1.set(f"CHECK_CALL 跟注：{ADVICE[1]}")
        a2.set(f"RAISE_HALF_POT 加注半池：{ADVICE[2]}")
        a3.set(f"RAISE_POT 加注一池：{ADVICE[3]}")
        a4.set(f"ALL_IN 全下：{ADVICE[4]}")
        
        final_advice_index = np.argmax(ADVICE)
        final_advice.set(ADVICE_TEXT[final_advice_index])
        

with open('./rlcard/card2index_tran.json', 'r') as file:
    card_to_index = json.load(file)

adviseBtn = tk.Button(advise, text = "給予建議", width=16, command = lambda: makeAdvice())
adviseBtn.grid(column=0, row=0, columnspan=3)

# 籌碼
chip = tk.LabelFrame(advise, text="籌碼設定", width=500, padx=20, pady=20, bg="#d9d9d9", font=('思源宋體 TW', 12, 'bold'))

myChip = tk.Label(chip, text="玩家已下注", bg="#d9d9d9", font=('思源宋體 TW SemiBold', 12)).pack()
# myChip_val = 1
# myChip_text = tk.IntVar()
# myChip_text.set(myChip_val)
# myChipText = tk.Label(chip, textvariable=myChip_text, bg="#d9d9d9", font=('思源宋體 TW SemiBold', 12)).pack()
myChipText = tk.Scale(chip, from_=1, to=100, length=450, orient='horizontal', bg="#d9d9d9", highlightthickness=0)
myChipText.pack()

maxChip = tk.Label(chip, text="全玩家最高下注", bg="#d9d9d9", font=('思源宋體 TW SemiBold', 12)).pack()
# maxChip_val = 2
# maxChip_text = tk.IntVar()
# maxChip_text.set(maxChip_val)
# maxChipText = tk.Label(chip, textvariable=maxChip_text, bg="#d9d9d9", font=('思源宋體 TW SemiBold', 12)).pack()
maxChipText = tk.Scale(chip, from_=1, to=100, length=450, orient='horizontal', bg="#d9d9d9", highlightthickness=0)
maxChipText.pack()

chip.grid(column=0, row=1, sticky="nsew")

# 建議
adviceLabel = tk.LabelFrame(advise, text="建議", width=500, padx=20, pady=20, bg="#d9d9d9", font=('思源宋體 TW', 12, 'bold'))
a0 = tk.StringVar()
a0.set(f"FOLD 棄牌：")
advice0 = tk.Label(adviceLabel, textvariable=a0, bg="#d9d9d9", font=('思源宋體 TW SemiBold', 10)).pack()
a1 = tk.StringVar()
a1.set(f"CHECK_CALL 跟注：")
advice1 = tk.Label(adviceLabel, textvariable=a1, bg="#d9d9d9", font=('思源宋體 TW SemiBold', 10)).pack()
a2 = tk.StringVar()
a2.set(f"RAISE_HALF_POT 加注1/2底池：")
advice2 = tk.Label(adviceLabel, textvariable=a2, bg="#d9d9d9", font=('思源宋體 TW SemiBold', 10)).pack()
a3 = tk.StringVar()
a3.set(f"RAISE_POT 加注1底池：")
advice3 = tk.Label(adviceLabel, textvariable=a3, bg="#d9d9d9", font=('思源宋體 TW SemiBold', 10)).pack()
a4 = tk.StringVar()
a4.set(f"ALL_IN 全下：")
advice4 = tk.Label(adviceLabel, textvariable=a4, bg="#d9d9d9", font=('思源宋體 TW SemiBold', 10)).pack()

adviceLabel.grid(column=1, row=1, sticky="nsew")

# 最終建議
finalLabel = tk.LabelFrame(advise, text="最終建議", width=500, padx=20, pady=20, bg="#d9d9d9", font=('思源宋體 TW', 12, 'bold'))
final_advice = tk.StringVar()
final_advice.set("")
finalAdvice = tk.Label(finalLabel, textvariable=final_advice, bg="#d9d9d9", font=('思源宋體 TW SemiBold', 12)).pack()
finalLabel.grid(column=2, row=1, sticky="nsew")


# --------------------------------------------------------------------------------

cam.grid(column=0, row=0)
show.grid(column=1, row=0, sticky="ne")
advise.grid(column=0, row=1, columnspan=2, sticky="nsew")

APP.mainloop()

cap.release()
cv2.destroyAllWindows()