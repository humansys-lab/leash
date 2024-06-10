# Leash Bio Competitions Repo

Leash Bio - Predict New Medicines with BELKAの研究室リポです．
https://www.kaggle.com/competitions/leash-BELKA

コードは```scripts```にある．
環境はkaggle notebookと同じ環境．
dockerコンテナを使用する．

## kaggleコンテナ入り方
1. H100(PC3)に接続
2. kaggleのコンテナにアタッチ
```bash
docker attach kaggle
```

## 実行手順
1. ```scripts/dataset.ipynb```を実行し，チャンクデータセットを取得．もともとのデータが大きくメモリ効率が悪いため，チャンクに分けてモデル構築するのが目的．

2. ```scripts/baseline_{MODEL NAME}.py```を実行．
実験をしたい場合は```scripts/baseline.ipynb```を推奨．
これまでちゃんと計算を回したのはCNNとLSTM．
Transformerは計算中．
Flash Attention Transformerも試す．
baselineは以下のノートブックを参考にした．もともとkerasで実装されていたものをpytorchで再実装した．
https://www.kaggle.com/code/ahmedelfazouan/belka-1dcnn-starter-with-all-data



