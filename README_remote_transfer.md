# リモート転送機能

学習完了後に、別のPCに自動的に学習結果を転送する機能です。

## セットアップ

### 1. 設定ファイルの作成

```bash
python scripts/configure_remote_transfer.py
```

### 2. ネットワーク共有の準備

**送信側PC（学習を実行するPC）:**
- ネットワーク共有にアクセスできることを確認

**受信側PC（転送先のPC）:**
- `WasherInspection` フォルダを共有する
- 共有名: `WasherInspection`
- アクセス権限: 読み書き可能

### 3. 設定例

**Windowsネットワーク共有の場合:**
```
共有パス: \\192.168.1.100\WasherInspection
転送先フォルダ: remote_models
```

## 転送されるファイル

- 学習済みモデルファイル (`.h5`)
- チェックポイントファイル
- ログファイル (`.json`, `.csv`)
- 学習履歴ファイル
- 設定ファイル

## 自動転送の動作

学習が正常に完了（終了コード0）した場合、自動的に転送が実行されます。

## 手動転送

設定後に手動で転送をテストする場合:

```bash
python scripts/remote_transfer.py --test
```

実際の転送を実行する場合:

```bash
python scripts/remote_transfer.py
```

## トラブルシューティング

### ネットワーク共有にアクセスできない

1. ネットワーク接続を確認
2. 共有フォルダのアクセス権限を確認
3. ファイアウォール設定を確認
4. Windowsの「ネットワーク探索」が有効になっているか確認

### 転送が失敗する

1. 設定ファイルを確認: `config/remote_transfer_config.json`
2. ログを確認: `logs/remote_transfer.log`
3. 手動でネットワーク共有にアクセスできるか確認




