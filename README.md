# TensorFlow tf.layers.denseを用いて3次元スパイラルデータの分類サンプル

## 目的
* 全結合レイヤー機能(tf.layers.dense)の使い方を学ぶ
  * tensorflowが自動で作る計算グラフを調べる(便利機能を使う前に中身を把握する)
* 「計算グラフの管理」と「セッションの管理」を分割する設計を考える
* matplotlibを用いて、分類の3Dグラフでの視覚表現を探る
  * 3次元の非線形の(渦巻いている)データを乱数で作り、2層のNNで正しく分類できるかを試す  

### 正解データ
* 緑がわかりづらいが、5つのカテゴリのデータがある。

![GitHub Logo](https://raw.githubusercontent.com/shikitari/tensorflow_dense_example/master/corect.png)

### 推測データ

* 正解率が低い場合、色が黒に近づく。
* この程度のばらつきであれば、3次元データでも分類できる。

![GitHub Logo](https://raw.githubusercontent.com/shikitari/tensorflow_dense_example/master/predict.png)

## tf.layers.denseの仕様調査まとめメモ
* バイアスの有無、活性化関数を設定できる
* 初期値はinitializerで設定できる。重みのデフォルトはは乱数、バイアスのデフォルトははゼロのようだ
  * 乱数のシードは、graph.seedで設定できる。これを指定すれば固定できる。
* TFで自動で、重みとバイアスのVariableの計算グラフが作られる。
  * graph.get_operations()を用いて、計算グラフの一覧を確認、取得してアクセスできる
  * おそらく、取得するメソッドは用意されていない(1.10.0の時点)ので、必要ならば自前でアクセスするAPIを作る必要がある

## 謝辞
* https://github.com/oreilly-japan/deep-learning-from-scratch-2/ を参考にさせていただきました。
