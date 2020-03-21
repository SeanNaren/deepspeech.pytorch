# deepspeech.pytorch 日本語訳

PyTorchによるDeepSpeech2の実装です。[DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) アーキテクチャに基づいたモデルを作成し、CTCアクティベーションを使ったトレーニングを行う事ができます。

## インストール

### Docker

公式なDockerhub imageはありませんが、各環境毎のDockerfileを作成するに当たって参考にできるDockerfileが用意されています。

```bash
sudo nvidia-docker build -t  deepspeech2.docker .
sudo nvidia-docker run -ti -v `pwd`/data:/workspace/data -p 8888:8888 --net=host --ipc=host deepspeech2.docker # Opens a Jupyter notebook, mounting the /data drive in the container
```

entrypointを変更することで、コマンドラインでの実行も可能です。

```bash
sudo nvidia-docker run -ti -v `pwd`/data:/workspace/data --entrypoint=/bin/bash --net=host --ipc=host deepspeech2.docker

```

### ソースコードから使うには

トレーニングプロセスを実行するためにはいくつかのライブラリが必要です。ここでは、全てのライブラリとPyTorch 1.0がUbuntuにインストールされているものと仮定しまます。

PyTorchをインストール:
インストール手順は[こちら](https://github.com/pytorch/pytorch#installation)を参照してください。

Warp-CTCをインストール:
インストールは下記forkから行ってください。

```
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc; mkdir build; cd build; cmake ..; make
export CUDA_HOME="/usr/local/cuda"
cd ../pytorch_binding && python setup.py install
```

NVIDIA apexのインストール:

```
git clone --recursive https://github.com/NVIDIA/apex.git
cd apex && pip install .
```

必須ではありませんが、言語モデルとビームサーチを使ったデコーディングを有効にしたい場合は、ctcdecodeをインストールしてください。

```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
```

最後にこのレポジトリをクローンして、必要なライブラリをインストールしてください。

```
pip install -r requirements.txt
```

## トレーニング

### データセット

現在サポートされているデータセットは、AN4、 TEDLIUM、 Voxforge、 Common Voice 、そして LibriSpeechです。

データをロードするのに使用するマニフェストファイルを作成するスクリプトが用意されています。スクリプトは ``data/`` ディレクトリに存在します。ほとんどのスクリプトは、選択したデータセットを個別にダウンロードすることができるようになっています。

#### カスタムデータセット

カスタムデータセットを使う場合は、個々の学習データの場所が記載されたCSVファイルを準備する必要があります。

フォーマットは以下の形式です。

```
/path/to/audio.wav,/path/to/text.txt
/path/to/audio2.wav,/path/to/text2.txt
...
```
最初のパスは音声ファイルのパスです。二つ目のパスは対応するテキストを記載したファイルへのパスです。
このCSVファイルのことをマニフェストファイルと呼びます。

#### 複数のマニフェストファイルを連結する

複数のマニフェストファイルを一つに纏めることで複数のデータセットでのトレーニングが可能になります。この際に短過ぎるクリップや長過ぎるクリップを削除することも出来ます。

```
cd data/
python merge_manifests.py --output-path merged_manifest.csv --merge-dir all-manifests/ --min-duration 1 --max-duration 15 # durations in seconds
```

### モデルのトレーニング

```
python train.py --train-manifest data/train_manifest.csv --val-manifest data/val_manifest.csv
```

指定可能なパラメータやオプションの詳細は`python train.py --help`にて確認できます。

[Visdom](https://github.com/facebookresearch/visdom)をサポートしているので、トレーイングを可視化することができます。試したい場合は、Visdomのサーバーを起動させた後に以下のコマンドを実行してください。

```
python train.py --visdom
```

 [Tensorboard](https://github.com/lanpa/tensorboard-pytorch) もサポートされています。使用する場合は、以下のオプションを指定してください。
 
```
python train.py --tensorboard --logdir log_dir/ # Make sure the Tensorboard instance is made pointing to this log directory
```
どちらのツールも `--id` パラメータを変更することで、可視化ツールで使用する名前を変更することができます。

### 複数のGPUでのトレーニング

distributed parallel wrapperを使ったマルチGPUでのトレーニングがサポートされています。詳細は [こちら](https://github.com/NVIDIA/sentiment-discovery/blob/master/analysis/scale.md) と[こちら](https://github.com/SeanNaren/deepspeech.pytorch/issues/211)を参照してください。

マルチGPUトレーイングを有効にしたい場合は以下のようにオプションを指定してください。

```
python -m multiproc train.py --visdom --cuda # Add your parameters as normal, multiproc will scale to all GPUs automatically
```

multiprocオプションを指定するとメインのプロセスだけでなく、全てのプロセスでログ取得が有効になります。

下記オプションを指定することで、利用可能な全てのGPUではなく、特定のGPUのみを使用するよう指定する事も可能です。

```
python -m multiproc train.py --visdom --cuda --device-ids 0,1,2,3 # Add your parameters as normal, will only run on 4 GPUs
```

インフィニバンドにアクセス出来ない場合は、NCCLバックエンドの使用を推奨します。


### Mixed-Precision

NVIDIAのvoltaか、それより上位のグラフィックカードを使用しているのであれば、メモリ使用の効率化とスピードアップの観点からmixed precisionを有効にすることを推奨します。詳細は[こちら](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)を参照してください。

Optimization levelsも指定可能です。Nvidia Apex APIでできることの詳細は [こちら](https://nvidia.github.io/apex/amp.html#opt-levels)を参照してください。

```
python train.py --train-manifest data/train_manifest.csv --val-manifest data/val_manifest.csv --opt-level O1 --loss-scale 1.0
```

mixed-precisionを有効にしてトレーニングを行うという事は、内部的に32ビットの少数か、その半分の精度の少数を使うという事です。デフォルトはFloat型を使用します。半分の精度で推論を行いたい場合は、テストや文字起こしの際に`--half`フラグを有効にしてください。スピードアップとメモリ使用量の効率化が期待できます。

### Augmentation

3種類のData Augmentationの手法がサポートされています。SpecAugment、ノイズインジェクション、そしてランダムなテンポと音量の微調整です。

#### SpecAugment

Spectral AugmentationのテクニックをMel spectogramに適用することで、インプットデータにバラエティを持たせることができ、より一般化されたモデルをトレーニングすることが可能になります。SpecAugmentを有効にするには`--spec-augment`フラグを使用してください。

SpecAugmentの実装は [このプロジェクト](https://github.com/DemisEom/SpecAugment)を参考にしています。

#### ノイズインジェクション

学習用音声データにノイズを動的に追加することで、より一般化されたモデルをトレーニングすることが可能になります。ノイズインジェクションを有効にするには、まず、ノイズ音声を配置するディレクトリを作成し、使用したい全てのノイズ音声ファイルをそこに配置してください。ノイズインジェクションが有効になると、データローダーはランダムにノイズ音声を選択し、動的にノイズを学習データに追加します。

ノイズインジェクションを有効にするには、`--noise-dir /path/to/noise/dir/`を設定してください。いくつか指定可能なノイズ関連のパラメータがあります。
`--noise_prob` はノイズを追加する確率をしていできますし、 `--noise-min`と`--noise-max`はノイズが追加される際のノイズのボリュームの最大値と最小値をそれぞれ指定できます。

ノイズインジェクションの結果をチェックできるスクリプトが用意されているので、ノイズを追加した際に、実際にどのような音声になるのかをチェックすることが可能です。

```
python noise_inject.py --input-path /path/to/input.wav --noise-path /path/to/noise.wav --output-path /path/to/input_injected.wav --noise-level 0.5 # higher levels means more noise
```

#### テンポと音量の微調整

音声データをロードする際にテンポと音量を微妙に変更することで、より一般化されたモデルをトレーニングする事が可能になります。
有効にするには`--speed-volume-perturb`フラグを有効にしてください。

### チェックポイント

トレーニングスクリプトはチェックポイントをサポートしています。保存されたチェックポイントからトレーニングを再開する事が可能です。
エポック事にチェックポイントを保存したい場合は、以下のようにオプションを指定してください。

```
python train.py --checkpoint
```

Nバッチ事に1回、チェックポイントを保存したい場合は以下のようにオプションを指定してください。

```
python train.py --checkpoint --checkpoint-per-batch N # N is the number of batches to wait till saving a checkpoint at this batch.
```

注記: バッチ毎のチェックポイントを有効にすると、以降バッチサイズを変更できなくなります。チェックポイント作成時に使用されていたバッチサイズと再トレーイング時に使用するバッチサイズが異なると、チェックポイントのウェイトをロードできません。

チェックポイントで保存されたウェイトからモデルのトレーニングを再開したい場合は、以下のオプションを指定してください。

```
python train.py --continue-from models/deepspeech_checkpoint_epoch_N_iter_N.pth
```

このオプションでは、前回トレーニング時のステータスやvisdomのグラフが引き継がれます。

もし、前回トレーニング時のステータスを引き継がずに、前回保存したウェイトだけを利用したい場合は、`--continue-from` とともに`--finetune` フラグを有効にしてください。

### 　バッチサイズの指定

以下のスクリプトは、指定したバッチサイズでのトレーニングがハードウェア的に可能かどうかを評価してくれます。

```
python benchmark.py --batch-size 32
```

`--help`オプションをつけてコマンドを実行することでも他のオプションの詳細を確認できます。


### トレーニング時のメタデータを確認する

保存されたモデルはトレーニングプロセスで使用されたメタデータの情報も保持しています。メタデータを確認したい場合は以下のオプションをつけてスクリプトを実行してください。

```
python model.py --model-path models/deepspeech.pth
```

注記: モデルの最後のレイヤーにsoftmaxはついていません。これは、warp-ctcデコーダーが内部でsoftmaxの処理を行っているからですが、この事はより複雑なデコーダーを自作する際に留意しておく必要があります。

## テスト/推論

トレーニング済みのモデルをテスト用データセットで評価したい場合は以下のスクリプトを実行してください。もちろん、フォーマットはトレーニング用データセットと同様である必要があります。

```
python test.py --model-path models/deepspeech.pth --test-manifest /path/to/test_manifest.csv --cuda
```

一つの音声の文字起こしをテストできるスクリプトも用意されています。

```
python transcribe.py --model-path models/deepspeech.pth --audio-path /path/to/audio.wav
```

もし、トレーニング時にmixed-precisionを有効にしていた場合は、`--half`フラグを有効にすることでスピードアップとメモリ使用の効率化を図る事ができます。

## 推論サーバー

音声ファイルを受け取って、文字起こしを行う、基本的なサーバー機能のスクリプトも用意されています。

```
python server.py --host 0.0.0.0 --port 8000 # Run on one window

curl -X POST http://0.0.0.0:8000/transcribe -H "Content-type: multipart/form-data" -F "file=@/path/to/input.wav"
```

## ARPA LM の使用

KenLMベースの言語モデルがサポートされています。以下のセクションではLibriSpeechを元とした言語モデルをダウンロードして、より良いデコードを実現するためのチューニング方法を紹介します。

LibriSpeechの言語モデルは[こちら](http://www.openslr.org/11/) でダウンロードできます。


### LibriSpeechの言語モデルをチューニング

まず、librispeechデータセットを``data/``ディレクトリにダウンロードします。次いリリースページからlibrispeechでPre-Trained（学習済み）のモデルをダウンロードします。そして、チューニングしたいARPAモデル（言語モデル）を[こちら](http://www.openslr.org/11/)からダウンロードします。ここでは3-gram ARPAモデル（3e-7 prune）を使用します。

まず、評価、分析するための音声モデルの結果を生成します。

```
python test.py --test-manifest data/librispeech_val_manifest.csv --model-path librispeech_pretrained_v2.pth --cuda --half --save-output librispeech_val_output.npy
```

ここではビームサーチのビーム幅は128とします。グリッドサーチを実行するためにCPUパワーの十分なマシンで実行することを推奨します。

```
python search_lm_params.py --num-workers 16 --saved-output librispeech_val_output.npy --output-path libri_tune_output.json --lm-alpha-from 0 --lm-alpha-to 5 --lm-beta-from 0 --lm-beta-to 3 --lm-path 3-gram.pruned.3e-7.arpa  --model-path librispeech_pretrained_v2.pth --beam-width 128 --lm-workers 16
```

このスクリプトはビーム幅128で、様々なalpha/betaのパラメータを試します。次のスクリプトはベストなalpha/betaの値を見つけます。

```
python select_lm_params.py --input-path libri_tune_output.json
```

見つかった alpha/betaの値は、ビームデコーダーの中で指定して使う事ができます。


### カスタム言語モデルの作成

カスタム言語モデルの作成には、[KenLM](https://github.com/kpu/kenlm)が必要です。ドキュメントに従い、言語モデルを学習させてみてください。出来上がったら、上記ステップで適切なalpha/beta値を見つける事ができます。


### その他のデコーダー

`test.py` と`transcribe.py`はデフォルトで`GreedyDecoder`を使用します。`GreedyDecoder`は各タイムステップ毎に一番高いスコアのラベルを選択し、その後、繰り返しとブランクのラベルの調整を行います。

オプションで、ビームデコーダーを使用することができますが、`ctcdecode`ライブラリをインストールする必要があります。ビームデコーダーを使用したい場合は、`test` 、及び`transcribe` スクリプトのオプションに`--decoder beam`を指定してください。ビームデコーダーでは以下のパラメータを指定できます。

parameters:
- **beam_width** 各タイムステップで保持するビーム幅
- **lm_path** KenLMベースの言語モデルのパス
- **alpha** 言語モデルへのウェイト
- **beta** 単語へのボーナスウェイト


### タイムオフセット

`transcribe.py`スクリプトを実行する際に `--offsets`を有効にすることで、各文字の位置情報を取得することができます。位置情報のデータはアウトプットテンソルの大きさを基準にしています。
取得された位置をテンソルのサイズで割って、音声の長さを掛ける事で各文字が音声ファイルの何秒目に位置しているのかを計算することができます。


## Pre-trained（学習済み）モデル

Pre-trainedは[こちら](https://github.com/SeanNaren/deepspeech.pytorch/releases)からダウンロード可能です。

## 謝辞

[Egor](https://github.com/EgorLakomkin)と[Ryan](https://github.com/ryanleary)、そして多くのコントリビュータに感謝します。

