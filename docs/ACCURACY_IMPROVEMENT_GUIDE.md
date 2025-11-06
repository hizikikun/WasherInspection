# 精度向上のための改善ガイド

## 📊 現在の状況分析

### 問題点
1. **EfficientNetB0, B1の精度が非常に低い**（5%, 8%）
2. **EfficientNetB2だけが高い精度**（80.54%）
3. **データ拡張が強すぎる**（訓練精度が低い）
4. **クラス不均衡**（good: 1144枚 vs 欠陥: 88-117枚）

## 🎯 精度向上のための施策

### 優先度1: データ拡張の調整（即効性あり）

#### 現在の問題
- データ拡張が強すぎて訓練精度が低い
- モデルが本質的な特徴を学習できない

#### 改善策
```python
# 中程度の拡張に変更
sparse_augmentation_params = {
    'rotation_range': 15,           # 30→15（弱める）
    'width_shift_range': 0.1,        # 0.2→0.1（弱める）
    'height_shift_range': 0.1,
    'shear_range': 0.1,              # 0.2→0.1（弱める）
    'zoom_range': 0.1,               # 0.2→0.1（弱める）
    'horizontal_flip': True,
    'vertical_flip': False,          # 垂直反転は無効化
    'brightness_range': [0.9, 1.1],  # [0.8, 1.2]→[0.9, 1.1]（弱める）
    'channel_shift_range': 0.05,     # 0.1→0.05（弱める）
}
```

**期待効果**: 訓練精度が上がり、モデルが適切に学習できる

---

### 優先度2: 正則化パラメータの調整

#### 現在の問題
- `dropout_rate: 0.5`が高すぎる（50%のニューロンを無効化）
- 学習が妨げられている

#### 改善策
```python
sparse_regularization = {
    'l1_lambda': 0.0005,      # 0.001→0.0005（緩める）
    'l2_lambda': 0.00005,     # 0.0001→0.00005（緩める）
    'dropout_rate': 0.3,      # 0.5→0.3（緩める）
    'sparse_threshold': 0.1,
}
```

**期待効果**: 学習能力が向上し、精度が上がる

---

### 優先度3: クラス不均衡対策の強化

#### 現在の状況
```
good: 1144枚（78%）
black_spot: 88枚（6%）
chipping: 117枚（8%）
scratch: 112枚（8%）
```

#### 改善策

**方法1: クラス重みの調整**
```python
# より強いクラス重みを適用
class_weights = compute_class_weight(
    'balanced',  # または 'balanced_subsample'
    classes=np.unique(y_train),
    y=y_train
)
# 必要に応じて手動で調整
class_weight_dict = {
    0: 0.5,  # good（多数派）の重みを下げる
    1: 2.0,  # black_spot（少数派）の重みを上げる
    2: 1.5,  # chippingの重みを上げる
    3: 1.5,  # scratchの重みを上げる
}
```

**方法2: データオーバーサンプリング**
```python
from imblearn.over_sampling import SMOTE
# 画像データの場合は適応的サンプリング
# または、少数派クラスの画像を回転・反転で増やす
```

**期待効果**: 少数派クラス（欠陥）の検出精度が向上

---

### 優先度4: 学習率とエポック数の最適化

#### 改善策
```python
# 学習率スケジューラの調整
initial_learning_rate = 0.001
first_decay_steps = 1000
t_mul = 2.0
m_mul = 1.0
alpha = 0.0001

lr_schedule = optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate,
    first_decay_steps,
    t_mul=t_mul,
    m_mul=m_mul,
    alpha=alpha
)

# エポック数を増やす
max_epochs = 200-300  # 現在より多く
patience = 50  # Early Stoppingの忍耐度
```

**期待効果**: より長く学習することで、精度が向上

---

### 優先度5: モデルアーキテクチャの改善

#### 現在の問題
- EfficientNetB0, B1がうまく学習できていない
- アンサンブルの効果が十分に発揮されていない

#### 改善策

**方法1: 事前学習済み重みの使用**
```python
# 現在: weights=None（ランダム初期化）
# 改善: ImageNetの事前学習済み重みを使用
base_model = EfficientNetB0(
    weights='imagenet',  # ← これを追加
    include_top=False,
    input_shape=(224, 224, 3)
)
```

**方法2: 転移学習の最適化**
```python
# より多くの層を学習可能にする
for layer in base_model.layers[:-50]:  # より多くの層を凍結
    layer.trainable = False

# または、段階的な解凍
# 1. 最初の50エポック: 最後の層だけ学習
# 2. 次の50エポック: 最後の100層を学習
# 3. 最後: 全層を学習
```

**期待効果**: B0, B1の精度が大幅に向上し、アンサンブルが機能する

---

### 優先度6: データ品質の向上

#### 改善策
1. **データの追加**
   - 特に少数派クラス（black_spot, chipping, scratch）のデータを増やす
   - 目標: 各クラス200-300枚以上

2. **データのクリーニング**
   - ノイズの多い画像を削除
   - ラベル付けの誤りを修正

3. **データの多様性**
   - 異なる照明条件
   - 異なる角度
   - 異なるサイズの欠陥

---

### 優先度7: 学習手法の改善

#### 改善策

**方法1: Focal Lossの使用**
```python
# クラス不均衡に強い損失関数
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        # 実装
        ...
    return focal_loss_fixed

model.compile(
    optimizer=optimizer,
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy']
)
```

**方法2: クロスバリデーション**
```python
# より厳密な評価と安定した学習
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    # 各フォールドで学習
    ...
```

**方法3: テスト時拡張（TTA: Test Time Augmentation）**
```python
# 評価時に複数の拡張画像で予測し、平均を取る
def predict_with_tta(model, X_test, n_augmentations=5):
    predictions = []
    for _ in range(n_augmentations):
        # 軽い拡張を適用
        X_aug = augment_images(X_test)
        pred = model.predict(X_aug)
        predictions.append(pred)
    return np.mean(predictions, axis=0)
```

---

## 📈 期待される効果

### 短期的な改善（即座に実装可能）
1. データ拡張の調整 → **+5-10%**
2. 正則化の緩和 → **+3-5%**
3. 事前学習済み重みの使用 → **+10-15%**

### 中期的な改善（データ追加が必要）
4. クラス不均衡対策 → **+5-10%**
5. データ追加 → **+5-10%**

### 長期的な改善（アーキテクチャ変更）
6. Focal Loss → **+2-5%**
7. TTA → **+1-3%**

**合計**: 現在80% → **90-95%以上**を目標

---

## ✅ 推奨実行順序

1. **即座に実行**（10分）
   - データ拡張を弱める
   - Dropoutを下げる
   - 事前学習済み重みを使用

2. **すぐに実行**（1時間）
   - クラス重みを調整
   - 学習率スケジューラを最適化
   - エポック数を増やす

3. **中期的に実行**（数日）
   - 少数派クラスのデータを追加
   - データのクリーニング

4. **長期的に実行**（数週間）
   - Focal Lossの導入
   - TTAの実装
   - アーキテクチャの改善

---

## 🔧 実装の準備

これらの改善を自動的に適用するスクリプトを作成できます。

**どの改善から始めますか？**
1. データ拡張と正則化の調整（即効性あり）
2. 事前学習済み重みの使用（大幅な精度向上）
3. すべての改善を一度に適用




















