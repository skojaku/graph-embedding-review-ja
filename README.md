# グラフ埋め込み法が拓くネットワーク科学の最前線

本レポジトリは以下に掲載されたデータと分析に用いたプログラムを保管するものです。

```
幸若 完壮. 2021. “埋め込み法が拓くネットワーク科学の新展開 (「複雑ネットワーク研究の最前線」特集号).” 
システム・制御・情報 = Systems, Control and Information : システム制御情報学会誌 65 (5): 182–87.
```

# Data

論文で用いたデータは以下から入手できます。

https://drive.google.com/drive/folders/1pVEftngukiZM3i7NIMBfrNy9O35pZILC?usp=sharing

# グラフ埋め込み法のコード

```python
import sys

sys.path.insert(0, "libs/network_embedding/")
import utils
from network_embedding import embedding, projection

vec = embedding.embed_network(A, window_length=10, dim=128)
```

- `A`: Adjacency matrix (numpy.array)
- `window_size`: Window size (int)
- `dim`: Dimension of embedding space (int)
- `vec`: embedding vectors. vec[i, :] represents the embedding vector for the ith node in the network. 


# 図を生成したノートブック

(ダウンロードしたデータをdataフォルダーに入れ、ノートブック内のパスを適宜再設定してください)

- [球団と都道府県の埋め込み](notebook/ja-embedding.ipynb)
- [空港網の埋め込み](notebook/plot-airport.ipynb)
- [Health Science雑誌ネットワークの埋め込み](notebook/plot-journals.ipynb)


