# TensorFlow tf.layers.denseを用いた3次元スパイラルデータの分類

## 目的
* matplotlibを用いて、3Dグラフ表現を探る
  * 3次元の非線形の(渦巻いている)データを乱数で作り、2層のNNで正しく分類できるかを試す  
* 「全結合レイヤー機能(tf.layers.dense)の仕様」を学ぶ
  * 裏側で、tensorflowが自動で作る計算グラフを調べる(便利機能を使う前に中身を把握する)
* 「計算グラフの管理」と「セッションの管理」を分割する設計を考える

## 3Dグラフ表現

### 正解データ
* 赤、黄緑、緑、青、紫で塗った5つのカテゴリに属するデータがあるとする。
* これを2層のニューラルネットワークで分類する。

![img](https://raw.githubusercontent.com/shikitari/tensorflow_dense_example/master/results/correct.png)

![img](https://raw.githubusercontent.com/shikitari/tensorflow_dense_example/master/results/vision.gif)

### 推測データ

* 推測が正しい場合、そのままの色。確率が低い場合、色が黒に近づく。

![img](https://raw.githubusercontent.com/shikitari/tensorflow_dense_example/master/results/predict.png)

![img](https://raw.githubusercontent.com/shikitari/tensorflow_dense_example/master/results/animation.gif)

### 正解と推測のアニメーション


### 3D視点切り替えアニメーション

## 全結合レイヤー機能(tf.layers.dense)の仕様
* 入力数、出力数、バイアスの有無、活性化関数などを設定できる
* 初期値はinitializerで設定できる。重みのデフォルトはは乱数、バイアスのデフォルトはゼロ
  * 乱数のシードは、graph.seedで設定できる。これを指定すれば固定できる。
* 重みとバイアスのVariableの計算グラフが自動で作られる。
  * graph.get_operations()を用いて、計算グラフの一覧を取得して、名前を手掛かりにオペレーションを取得することができる
  * おそらく、重みやバイアスを取得するメソッドは用意されていない(1.10.0の時点)。

## 謝辞
* https://github.com/oreilly-japan/deep-learning-from-scratch-2/ を参考にさせていただきました。
