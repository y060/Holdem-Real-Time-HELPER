# Holdem-Real-Time-HELPER
本專題利用人工智慧影像辨識，在即時識別撲克牌花色數字後，將辨識出之撲克牌於GUI介面上顯示出來。
希望我們的程式能夠給予新手輔助建議，讓剛上桌的新手彷彿有個老師在旁邊進行指導，提升德州撲克上手的容易程度，使得更多人會有意願開始遊戲。

## DataSet 資料集
來源：https://universe.roboflow.com/augmented-startups/playing-cards-ow27d

## 即時辨識撲克牌數字與花色 ── YOLOv8
https://github.com/ultralytics/ultralytics

## 德州撲克建議系統 ── RLCard & DQNAgent
本專題使用RLCard函式庫作為主要架構，這是一個專門訓練多種卡牌遊戲所做的訓練框架，而其中也包含了德州撲克遊戲。我們利用其中包含的德州撲克遊戲環境，並以DQNAgent模型訓練遊戲代理（Agent）。

官網：https://rlcard.org/index.html
GitHub：https://github.com/datamllab/rlcard