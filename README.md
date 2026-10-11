[[Japanese](README.md)/[English](README_EN.md)]

# Strands-Decider-2B-ONNX-CPU

[`StrandsAgents/strands-decider-2B-hobson-v21`](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21)（Strands Decider。構造化判定モデル。テキストと画像）を、onnxruntime と numpy だけで推論する最小依存の推論実装です。 ※GPU推論も可（ただし、検証した GPU では速くなりませんでした。後述）<br>
公式の推論コード（`strands-decider`）が要求する torch・transformers は使いません。

# Features
以下の特徴があります。
- 最小依存：実行時に必要なパッケージは onnxruntime と numpy の2つだけ（画像を使う場合は pillow を足した3つ）
- 動かすのに必要なスクリプトは `download_model.py`（モデル取得）と `demo_inference_text.py`（推論）の2つ。画像は `demo_inference_image.py`、カメラは `demo_inference_webcam.py`
- 判定は参照実装（Strands 公式エンジン、PyTorch fp32）と一致（テキスト 16/16、画像 8/8。詳細は「Verification」）
- 1問が1回の順伝播で決まる（生成は走らない）。質問は3種類：`noul`（はい・いいえ）、`choice`（2〜255個から1つ）、`score`（2〜10段階の順序尺度）。どの答えにも確信度が付く
- 画像対応（1回に最大4枚。画像は長辺 448 px まで縮小してから `<state>` に埋め込む）
- 前処理（Qwen byte-level BPE、プロンプトの組み立て、3次元 M-RoPE、画像のパッチ化と視覚エンコーダの入力）を純Pythonで移植し、参照実装との突き合わせ検証を同梱
- CPU のほか、GPU（CUDA または DirectML）でも動かせる（`--gpu`）

# Purpose of This Repository
以下の検証を目的としています。
- Strands Decider 2B（Qwen3.5-2B-Base ＋ LoRA ＋ ポインタ型の読み出しヘッド）を、実行時依存2〜3つだけでどこまで動かせるか
- 参照実装の前処理を純Pythonに移植して、同じ入力が作れるか
- 画像つきの判定が、ONNX 化した視覚エンコーダで元の実装と同じ答えになるか

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later
pillow       10.0 or later   （画像を使う場合のみ）
```
実行に必要なのはこれだけです。モデルの取得も標準ライブラリのみで動きます。<br>
**メモリは、約 8 GB 以上の空きが必要です**（重みは fp32 で 7.5 GB。系列長 L=384 のときのピークは約 7.4 GB、画像を使うと視覚エンコーダの 1.3 GB が加わります）。<br>
環境構築は[uv](https://docs.astral.sh/uv/)を使います。<br>
オプションのグループがあります。
- `verify`：トークナイザの突き合わせ検証（`check_tokenizer_fuzz`、`check_combining`）用の `tokenizers`
- `test`：`tests/` 用の pytest（モデルは要りません）
- `webcam`：`demo_inference_webcam.py` 用の OpenCV
- `export`：ONNX を自分で書き出すときの torch・transformers など（実行時には使いません）

# Installation

```bash
git clone https://github.com/Kazuhito00/Strands-Decider-2B-ONNX-CPU
cd Strands-Decider-2B-ONNX-CPU

# 実行時の依存（onnxruntime・numpy・pillow）のみインストール
uv sync
```
`uv sync` は、このリポジトリの `strands_decide` パッケージも環境に入れます。`uv pip install git+https://github.com/Kazuhito00/Strands-Decider-2B-ONNX-CPU` でも `import strands_decide` できます（モデルのファイルは別に取得します）。

# Download Model
Strands Decider には公開されている ONNX がありません。書き出した ONNX を[Releases](https://github.com/Kazuhito00/Strands-Decider-2B-ONNX-CPU/releases)に置く前提です（リポジトリには含めていません。Release の作り方は `release/make_release.py` と下の「Release の中身」）。<br>
以下のスクリプトで `models/onnx/` に取得します。標準ライブラリのHTTPSのみで動き、gitもgit-lfsも不要です。
```bash
uv run download_model.py
```
- **引数なしでは、Release の全ファイル（L=384・512・1280 の全グラフ、視覚エンコーダ、ヘッド、トークナイザ、ライセンス文。18 個のアセット、約 9.4 GB）を取得します**
- 取得元は、既定で `Kazuhito00/Strands-Decider-2B-ONNX-CPU` の最新の Release です（`--repo OWNER/NAME` か、環境変数 `STRANDS_ONNX_REPO` で別のリポジトリにできます。優先順位は 引数 > 環境変数 > 既定）
- 取得したファイルは、Release の `manifest.json` に書かれた SHA-256 と照合します（不一致ならファイルを削除して停止）
- GitHub の制限（1ファイル 2 GiB 未満）で分割されたファイル（共有の重み `torso_L512.onnx.data`、7.5 GB を 4 分割）は、取得後に結合して照合します（結合の間は、分割ファイルと結合後のファイルで、一時的に約 7.5 GB 余分にディスクを使います）
- 取得済みなら何もせず、中断した場合は同じコマンドで続きから再開します

```bash
uv run download_model.py --list                                          # ファイル一覧
uv run download_model.py --exclude "vision_encoder*"                     # テキスト専用（視覚エンコーダを取らない。約 1.3 GB 少ない）
uv run download_model.py --seq-len 384                                   # 既定の L=384 のグラフだけ（共有の重み・視覚エンコーダは付いてきます）
uv run download_model.py --tag v0.0.0                                    # タグを指定（既定は latest）
```

Release には L=384（既定）・512・1280 の3つのグラフが入っています。**重み（`torso_L512.onnx.data`、7.5 GB）は3つで共有しているので、グラフを1つ増やしても約 0.17 GB しか増えません。** そのため `--seq-len 384` でも重みは 7.5 GB 取得します（節約できるのはグラフ 2 つ分の約 0.35 GB だけです）。`manifest.json` には、`.onnx` ごとに「必要なデータファイル」が書かれているので、`--seq-len` や `--exclude` で、グラフだけが残って重みが無い、ということは起きません（`--exclude "torso_L512*"` と書いても、共有の重みは他のグラフが必要とする限り残ります）。<br>
L=1024 のグラフは Release に入れていません（自前の重みのコピーを持つので、7.5 GB 増えます）。要るときは自分で書き出します（次節）。

自分で書き出す場合（元の checkpoint を取得して ONNX にします。torch が要り、書き出しはメモリ・時間とも重い作業です）。
```bash
uv sync --group export
uv run export/prepare_model.py                       # Qwen3.5-2B-Base と Strands の LoRA・ヘッドを取得（約 4.6 GB。リビジョン固定、SHA-256 照合）
uv run export/export_onnx.py --seq-len 512           # 最初に書き出すバケット（重みの持ち主になる）
uv run export/export_onnx.py --seq-len 384           # 2つ目以降のバケット（このままだと 7.5 GB の重みがもう1つ増えます）
uv run export/share_external_data.py models/onnx/torso_L384.onnx   # 重みを torso_L512.onnx.data に寄せる（バイト列が同一のテンソルだけ）
uv run export/export_vision.py                       # 視覚エンコーダ
```
`export_onnx.py` は `models/onnx/head.npz` と `graph_meta_L<N>.json` も書きます。`tokenizer.json` と `preprocessor_config.json` は、`models/release/tokenizer.json`（Strands）と `models/base/preprocessor_config.json`（Qwen3.5）を `models/onnx/` にコピーしてください。

### Release の中身
`uv run release/make_release.py --dry-run` で一覧を出せます（ハッシュは計算せず、何も書きません）。`release/make_release.py` が `models/` から `dist/release/` に作ります（アップロードはしません）。

| アセット名 | サイズ |
|---|---:|
| `onnx__torso_L384.onnx` / `onnx__torso_L512.onnx` / `onnx__torso_L1280.onnx` | 167.2 MB / 168.6 MB / 177.1 MB |
| `onnx__torso_L512.onnx.data.part000` 〜 `part002`（共有の重み 7,527,333,888 バイトを分割） | 各 2,040,109,465 バイト（1.9 GiB） |
| `onnx__torso_L512.onnx.data.part003` | 1,407,005,493 バイト |
| `onnx__vision_encoder.onnx` / `onnx__vision_encoder.onnx.data` | 5.6 MB / 1,325.7 MB |
| `onnx__head.npz` / `onnx__tokenizer.json` / `onnx__preprocessor_config.json` | 4.2 MB / 20.0 MB / 390 バイト |
| `onnx__graph_meta_L384.json` / `_L512.json` / `_L1280.json` / `onnx__provenance.json` | 各 1 KB 未満 |
| `onnx__LICENSE-STRANDS-DECIDER.md` / `onnx__LICENSE-QWEN` | 13 KB / 11 KB |
| `manifest.json` / `SHA256SUMS` | 小 |

合計は約 9.4 GB（18 アセット + `manifest.json` + `SHA256SUMS`）です。

# Usage

### 実行例(Python)
```bash
uv run demo_inference_text.py
uv run demo_inference_image.py
```
テキストのデモは、組み込みの例（カスタマーサポートのチケットに「怒っているか」「どのチームか」「どれくらい緊急か」を聞く）に答えます。自分の入力は次のように渡します。
```bash
uv run demo_inference_text.py --state "My card was charged twice." \
    --choice "team=billing,tech" --noul "angry=The customer is angry" --score "urgency=low,medium,high"
```
画像のデモは、既定で `artifacts/images/sample_chart.png`（棒グラフ。「どの棒が最も高いか」「棒グラフか」を聞きます）を使います。このサンプルは、Wikipedia 画像ベンチの合成データ（スクリプトで描いた画像。`syn-0001`）の1枚で、写真ではなく、Wikipedia の素材も含みません。自分の画像を使うときは、画像のパスと質問を渡します（最大4枚）。
```bash
uv run demo_inference_image.py shot.png --choice "status=paid,pending,failed" --noul "paid=Is it paid?"
```

```python
from strands_decide import Decider, choice, noul, score

decider = Decider("models/onnx")          # 既定: L=384（seq_len=512 などで変えられます）

decider.decide("I was charged twice for my Pro subscription this month.", {
    "team": choice("Which team?", {"billing": "charges, refunds", "tech": "bugs"}),
    "angry": noul("The customer is angry"),
    "urgency": score("How urgent?", ["low", "medium", "high", "critical"]),
})
```

**質問ごとに1回の順伝播です。** ポインタ型のヘッドが、`<answer>` の位置の隠れ状態（クエリ）と、各選択肢の行の最後のトークンの隠れ状態（キー）を読んで確率を出します。生成は一切走りません。

画像は同じリクエストに混ぜます（最大4枚）。
```python
decider.decide("Look at the screenshot.", {...}, images=["shot.png"])
```
`images` には、ファイルのパス、`data:image/...;base64,` の URL、バイト列、Pillow の画像を渡せます。長辺 448 px を超える画像は 448 px に縮小されます（正方形で 196 トークン）。

### カメラで「人が映っているか」を判定する
OpenCV でカメラの映像を取得し、最新のフレームについて「人が映っているか」を判定して、結果を映像に重ねて表示します。
```bash
uv run --group webcam demo_inference_webcam.py                        # カメラ 0
uv run --group webcam demo_inference_webcam.py --camera 1 --mirror
uv run --group webcam demo_inference_webcam.py --question "Is the person wearing glasses?"
uv run --group webcam demo_inference_webcam.py --camera artifacts/images/sample_chart.png   # カメラの代わりに画像・動画ファイル（人は映っていないので NO になります）
```
- 判定は別スレッドで動かし、映像の表示は止めません。判定が出るたびに、上部の帯（緑＝はい／赤＝いいえ）と「はい」の確率が更新されます
- 質問は `--question`（はい・いいえの質問）で変えられます。モデルに渡す画像は長辺 224 px に縮小します（`--size`）
- `q` または `Esc` で終了します。OpenCV は、このデモだけの依存です

### 質問の種類
| 種類 | 作り方 | 答え |
|---|---|---|
| `choice` | `choice(指示, {名前: 説明})`（2〜255個） | 選択肢から1つ（`choice`、`confidence`、`probabilities`） |
| `score` | `score(指示, [最低, ..., 最高])`（2〜10段階） | 順序尺度の期待値（`score`、`legend`、`confidence`）。先頭が最低 |
| `noul` | `noul("文")` | 文が成り立つか（`noul` = はいの確率）。選択肢は `false` / `true` 固定 |

`state` には、文字列か、JSON にできるもの（辞書など）を渡せます。

### GPUで実行する（任意）
既定は CPU です。GPU（CUDA）で動かすときは、`onnxruntime` の代わりに `onnxruntime-gpu` を入れます。**この2つは同じ `onnxruntime/` フォルダに書き込むため、同居できません。** extra `gpu` を使うときは、入れたあとに入れ替えます。
```bash
uv sync --extra gpu                                              # onnxruntime-gpu も入る（CPU 版と同居した状態）
uv pip install --force-reinstall --no-deps onnxruntime-gpu       # 同じフォルダへ GPU 版を上書きして揃える
uv run --no-sync demo_inference_text.py --device cuda            # 以後は --no-sync で動かす（素の uv run は CPU 版に戻す）
```
- **要件**：CUDA 13 と cuDNN 9 の `bin` フォルダを PATH に入れます（`onnxruntime-gpu` は CUDA の DLL を同梱せず PATH から探します。入れたあとは、ターミナルを開き直します）。Windows の例です（バージョンは環境に合わせてください）。
  ```text
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64
  C:\Program Files\NVIDIA\CUDNN\v9.20\bin\13.2\x64
  ```
  DirectML（Windows。CUDA は不要）は、`onnxruntime-directml` を入れて `--device auto` で使います（このリポジトリでは動作を確認していません）
- **選び方**：`--device cpu`（既定）／`--device cuda`（CUDA、だめなら CPU）／`--device auto`（CUDA か DirectML があればそれ、なければ CPU）。`--gpu` は `--device auto` と同じです。`demo_inference_text.py`、`demo_inference_image.py`、`demo_inference_webcam.py`、`verify/bench.py` で使えます
- **静かな CPU へのフォールバックを防ぎます**：実際に使われたプロバイダを、デモの最終行（`providers: CUDAExecutionProvider, CPUExecutionProvider` のように、セッションが実際に得た順）と `verify/bench.py` の出力に表示します。GPU を頼んだのにロードできなかった（DLL が見つからない、VRAM 不足など）ときは、理由つきの警告を出して CPU で動きます。**`--require-gpu` を付けると、警告ではなくエラーで止まります**（CUDA を使えるビルドでなければ、モデルを読む前に止まります）
```python
from strands_decide import Decider, select_providers

decider = Decider("models/onnx", providers=select_providers("cuda"), require_gpu=True)
print(decider.providers)   # 実際に使われたプロバイダ
```
- **VRAM の目安**：torso の fp32 の重みだけで約 7.5 GB、視覚エンコーダが +1.3 GB です。VRAM 4 GB の GPU には載りません（上の `--require-gpu` で確かめてください）。十分な VRAM がある GPU のための任意機能です
- GPU での確率は、CPU と完全には一致しません（演算の順序の違い）。参照実装との一致検証は CPU のみです

**GPU で速くなる、とは言えません。** 手元の測定では、次のとおりでした。
- 作業時の測定（RTX 3050 Ti Laptop、VRAM 4 GB、CUDA 13.4、cuDNN 9.20、onnxruntime-gpu 1.31）：CUDA セッションは作れるものの（VRAM は約 3.7 GB で頭打ち）、1 問あたりの時間は CPU（8 スレッド）と同等で、GPU の利点はありませんでした
- Wikipedia 画像ベンチの比較表でも、Strands の GPU 列は載せていません（較正のとれた測定がありません）
- このリポジトリで `--device cuda --require-gpu` を上の環境で実行したところ、CUDA セッションが作られ（`providers: CUDAExecutionProvider, CPUExecutionProvider`）、組み込みの例の出力は CPU と同じでした。最初の呼び出しは 165 秒かかりましたが、重みの読み込みを含む値で、速度の測定ではありません。**VRAM 不足のエラー（OOM）は、この環境では起きませんでした。** OOM のときの挙動（警告を出して CPU で動く／`--require-gpu` ならエラー）は、擬似的な失敗を使った単体テスト（`tests/test_providers.py`）でだけ確認しています。実機での OOM は未検証です

# 系列長は固定です（重要）
モデルの linear attention が 64 トークンのチャンクを Python のループで回すため、**エクスポート時に展開されて固定回数になります**。動的な系列長のグラフは作れません（ある長さで書き出したグラフは、それより長い入力の後半を黙って無視します）。

そのため固定長 L（64 の倍数）で書き出し、推論時に L まで**右パディング**します。読み出し位置はすべてパディングより前で、モデルは causal なので結果は変わりません（`verify/check_padding.py`：最大差 0）。**コストは実トークン数ではなく L に比例します。**

L=384、512、1280 の3つを Release に入れてあり、**既定は L=384** です（`Decider(..., seq_len=384)`、デモと `verify/bench.py` は `--seq-len`）。

- `seq_len` 以上で最小のグラフを最初に読みます
- リクエストが収まらないときは、**収まる最小のバケット**に自動で切り替えます（古いグラフを解放してから読むので、大きなグラフがメモリに2つ載ることはありません）。切り替えたあとは、そのバケットのままです
- 収まるバケットが1つも無いときだけ、必要な長さを示す `ValueError` になります
- 3つのグラフは**重みのファイルを共有**します（`torso_L384.onnx` と `torso_L1280.onnx` が `torso_L512.onnx.data` を指します。バイト列が同一のテンソルを同じ位置から読むだけなので、数値は変わりません。`cmp` で一致を確認済み）。**`torso_L512.onnx.data` を消さないでください**
- 画像を使うと、プロンプトが長くなります（448 px の画像1枚で約 140〜200 トークン。4枚では約 1024 トークンが要るため、Release のグラフでは L=1280 を読みます）

メモリのピーク（fp32、8 スレッド、テキスト 2 問、作業時の記録。ほかのジョブが動いていたため上限の目安です）：

| L | ピークRAM |
|---|---:|
| 384（既定） | 7.4 GB |
| 512 | 8.1 GB |
| 1024（Release には無し） | 12.6 GB |
| 1280 | 15.8 GB |

**速度（秒）は載せていません。** 較正のとれた静かなマシンでの測定が、まだ無いためです（ほかの処理が動いている状態の値しかありません）。ディスクから重みを読み込む最初の1回は、特に遅くなります。測るときは、ほかの重い処理を止めて `uv run --no-sync verify/bench.py --seq-len N --questions 2 --runs 5` を使ってください。

# Verification
```bash
uv run verify/run_all.py                                 # 全ゲート（モデルを読むものを含む。RAM が約 8 GB 以上要ります）
uv run verify/run_all.py --skip-model                    # モデルを読むゲートを飛ばす（速い）
uv run --group verify verify/run_all.py --full-fuzz      # tokenizers があればファズも本気で回す
uv run --group test pytest                               # 単体テスト（モデル不要）
```

| ゲート | 内容 | 結果 | 必要なもの |
|---|---|---|---|
| check_prompt | プロンプト文字列・`input_ids`・読み出し位置が、参照実装と完全一致 | 16/16 完全一致 | `models/onnx/tokenizer.json` |
| verify | 判定と確率が参照実装（fp32）と一致。`--seq-len N` で各バケットを検証 | 16/16 で判定一致、確率差 最大 8.6e-4 | モデル |
| check_padding | 右パディングの中身が読み出し位置の隠れ状態を動かさないこと | 最大差 0 | モデル |
| verify_image | 視覚エンコーダの出力・画像つきプロンプトの ids・読み出し位置・確率が、参照の視覚エンジンと一致 | 8/8 問、ids 完全一致、確率差 最大 7.3e-4、視覚エンコーダ出力差 最大 4.4e-4 | モデル + pillow |
| check_vision_inputs | 画像のピクセル・bilinear の位置・位置 id が transformers と一致 | 18/18（ピクセルは 1 ulp 以内） | + torch・transformers・`models/base` |
| check_tokenizer_fuzz | Rust `tokenizers` との突合（id とオフセット） | id の不一致 0 | + tokenizers |
| check_combining | 結合文字で HF とオフセットが 1 ずれても、選択肢の読み出し位置が動かないこと | 動かない | + tokenizers |

結果は、作業時（この実装を作ったとき）の記録で、`check_prompt`・`check_padding`・`verify`・`verify_image`・`check_tokenizer_fuzz`・`check_combining` は、このリポジトリで再実行して同じ結果を確かめています（`check_vision_inputs` は、4.5 GB のベースモデルと torch が要るため再実行していません）。

`artifacts/golden.json`・`golden_hidden.npz`・`golden_image.json`・`golden_image.npz` は、参照実装（Strands 公式エンジン、PyTorch fp32、CPU。画像は公式リポジトリの main の視覚エンジン。コミットは `golden_image.json` に記録）を実際に動かして作った正解データです。同じ重みを両者が走らせるので、差が出たら移植のバグと断定できます。1 ファイル 5 MB 以下に抑えるため、`golden_image.npz` に保存した視覚エンコーダの出力は、画像 7 枚のうち合成画像 3 枚（shapes・screen・chart）の分だけです（質問側の照合は、すべての画像で行います）。`artifacts/images/` の写真は、Wikimedia Commons の CC0／パブリックドメインの画像です（`LICENSES.json`）。

既知の差：結合文字（例：`a` + U+0301）を含むテキストでは、文字オフセットが HF と 1 ずれます（HF は NFC 前の範囲を返します）。トークン id は同一で、選択肢の読み出し位置は動かないことを `check_combining` で確認しています。

### 精度の目安
このリポジトリで新しく測った精度はありません。次の2つは、他の測定の引用です。
- 元のモデルカード（[`StrandsAgents/strands-decider-2B-hobson-v21`](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21)）の報告：JevBench public（4096 トークン窓）で正答率 0.7619（176/231）。ほかの評価の値もモデルカードにあります（PyTorch 実装での値です）
- Wikipedia 画像ベンチ（`wikipedia-vision-decision-bench`。8 スレッド・224px）での、この ONNX 実装の正答率（%）：

| Commons 英 | Commons 日 | synthetic 英 | synthetic 日 | reasoning 英 | reasoning 日 |
|---|---|---|---|---|---|
| 98.1 | 94.2 | 92.7 | 92.4 | 50.2 | 48.0 |

画像つき 1875 問（Commons 1268・synthetic 382・reasoning 225）の値で、ベンチ側の「移行前の測定」です。reasoning は選択肢が少なく当てずっぽうでも当たりやすい形式を含むため、偶然の正答率を意識して読んでください。

# Limitations
- 系列長は固定です。既定は L=384 で、収まらなければ大きいバケットに切り替えます。最大の L（Release では 1280）を超えると例外になります（必要な `--seq-len` を提示します）
- fp32 のみです（fp16 は未対応）。動画は未対応です。画像は1回に最大4枚です
- 画像について、元の実装の開発元は、アダプタ（LoRA）はテキストだけで学習し、画像の答えは土台のモデル（Qwen3.5）の視覚エンコーダによるもの、としています。画像が無いときと有るときで、確信度は区別できません
- `tokenizer.py` / `protocol.py` はこのチェックポイント専用で、想定外の入力は例外で弾きます
- メモリを多く使います（最小でも約 8 GB の空きが要ります）
- GPU では確率が CPU と完全には一致しません。VRAM が小さいと載らず CPU に切り替わります

# Project Structure

```text
README.md                # README（日本語）
README_EN.md             # README（英語）
LICENSE                  # Apache-2.0
pyproject.toml           # 依存定義（実行時は onnxruntime・numpy・pillow。グループ: verify / test / webcam / export）
uv.lock                  # 依存の固定
download_model.py        # モデル取得（GitHub Releases、SHA-256照合、分割ファイルの結合）
demo_inference_text.py   # テキストの推論デモ
demo_inference_image.py  # 画像の推論デモ
demo_inference_webcam.py # カメラで「人が映っているか」を判定（OpenCV）
strands_decide/          # 推論実装（軽い依存のみ。export/ を import しない）
  runtime.py             #   ORT セッション + 右パディング + 読み出し + バケット切り替え + GPU プロバイダ
  tokenizer.py           #   Qwen byte-level BPE の純Python実装
  protocol.py            #   プロンプトの組み立て・ポインタヘッド・温度・確信度の計算
  image.py               #   Pillow デコード + smart_resize + パッチ化 + 視覚エンコーダの入力 + 3次元 M-RoPE
  _regex_translate.py    #   Rust 方言の正規表現を Python の re に直す
  _uniprops.py           #   生成物: Unicode範囲表
export/                  # 重い依存（torch / transformers / onnx）
  prepare_model.py       #   checkpoint 取得（リビジョン固定・SHA-256 照合）
  merge_lora.py          #   LoRA を畳み込む・ヘッドを読む
  export_onnx.py         #   torso（LoRA 畳み込み済み）+ 読み出しの gather を ONNX に
  export_vision.py       #   視覚エンコーダを ONNX に（vision_export_patch.py で単一画像版に差し替え）
  share_external_data.py #   バケット間で重みのファイルを共有（同一バイト列のテンソルだけ）
  make_golden.py         #   artifacts/golden.json を作る（cases.py の質問を参照実装で走らせる）
  make_golden_image.py   #   artifacts/golden_image.* を作る（cases_image.py）
  gen_uniprops.py        #   _uniprops.py を作る
  download.py            #   Hugging Face から標準ライブラリだけで取得
release/
  make_release.py        # models/ から Release 用のファイルを作る（分割・SHA-256・manifest.json）。--dry-run で一覧だけ
  notes.md               # Release の本文
tests/                   # 単体テスト（モデル不要）
verify/                  # 検証ゲート（run_all.pyで一括）、bench.py（速度・メモリの測定）
artifacts/               # golden*（検証の正解データ）、images/（デモとテスト用の画像）
models/                  # ONNX・トークナイザ・設定（gitignore）
```

# Reference
- [StrandsAgents/strands-decider-2B-hobson-v21](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21)（元のモデル。LoRA アダプタとポインタヘッド。Apache-2.0）
- [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider)（元の推論・学習コード。Apache-2.0）
- [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)（元のモデルのベース。Alibaba Cloud。Apache-2.0）

# License
Strands-Decider-2B-ONNX-CPU is under [Apache-2.0 license](LICENSE).<br>
モデルの重みは、元のモデル（[StrandsAgents/strands-decider-2B-hobson-v21](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21)、Apache-2.0。ベースの Qwen3.5-2B-Base も Apache-2.0）のライセンスに従います。Release には、両方のライセンス文（`LICENSE-STRANDS-DECIDER.md`、`LICENSE-QWEN`）を元のまま同梱しています。どちらのリポジトリにも NOTICE ファイルは無いことを確認しています（2026-10-11 時点）。<br>
`strands_decide/` のコードは、Strands Decider（Apache-2.0）の推論コード、Hugging Face Transformers（Apache-2.0）の Qwen2-VL の画像処理や Qwen3.5 の視覚位置の計算を、numpy で書き直した移植です。

## 変換物であることについて
Release に含まれる ONNX ファイルは、公式には ONNX が公開されていない [Strands Decider 2B](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21)（Apache-2.0。LoRA アダプタとポインタヘッドだけのリポジトリ）を、このリポジトリの `export/` で ONNX 形式に変換したものです。具体的には、次の改変をしています。
- ベースモデル [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)（Alibaba Cloud。Apache-2.0）の重みに、LoRA アダプタを畳み込み（`W + (alpha / r) * B @ A`）、テキストのデコーダ（torso）を fp32 の ONNX グラフとして書き出しています。元のリポジトリには、ベースの重みは含まれていません
- 系列長ごとの固定長グラフに分け、グラフ間で重みのファイルを共有しています
- ベースモデルの視覚エンコーダ（Qwen3.5 のもの。重みは無調整）を、単一画像用の ONNX として書き出しています
- ポインタヘッド（LayerNorm と 2 つの射影）は、`head.npz`（fp32）に変換し、numpy で計算します

重みは fp32 のまま書き出しており、元の実装（PyTorch fp32）との判定の一致を確認しています（テキスト 16/16、画像 8/8。確率の差は最大 8.6e-4）。元のモデルの学習データには、Apache-2.0・CC-BY・CC-BY-SA・CC0・MIT のほか、ライセンス表記が `other`・`unknown`・無しのものも含まれます（元のモデルカードの `LICENSE.md` に一覧があります）。再配布する前に、元のモデルカードと各データのライセンスを確認してください。元のモデルの提供元およびこの変換物の動作を、提供元は保証していません。
