# train1000


[train with 1000](http://www.ok.sc.e.titech.ac.jp/~mtanaka/proj/train1000/)のcifar10をやってみた

## 成績

#### 初回コミット

| val_acc | val_loss |
|:-----------|------------:|
|0.4814|1.646|

#### BatchNormalization追加後

| val_acc | val_loss |
|:-----------|------------:|
|0.5095|1.5598|

#### 目的変数をval_lossからval_accに変更

| val_acc | val_loss |
|:-----------|------------:|
|0.5220|2.5123|

-----

## 参考・目標
### [ベースライン](https://github.com/mastnk/train1000)

| val_acc | val_loss |
|:-----------|------------:|
|0.5170|1.6784|

### @imenurok氏(78.4%)

https://qiita.com/imenurok/items/31490be74f3437dc8fed
