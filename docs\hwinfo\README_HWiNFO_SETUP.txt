HWiNFO自動再起動システム セットアップ手順
==========================================

【重要】タスクスケジューラへの登録には管理者権限が必要です。

■ ステップ1: タスクスケジューラへの登録

方法A: バッチファイルで実行（推奨）
  1. scripts/setup_hwinfo_scheduler_admin.bat を右クリック
  2. 「管理者として実行」を選択
  3. 実行完了後、ステップ2へ

方法B: PowerShellで実行
  1. PowerShellを管理者として起動
  2. 以下を実行:
     cd C:\Users\tomoh\WasherInspection\scripts
     .\setup_hwinfo_scheduler_admin.bat

■ ステップ2: 繰り返し間隔の設定（必須）

1. タスクスケジューラを開く（スタートメニュー > 「タスク」と検索）
2. 「タスクスケジューラライブラリ」を展開
3. 「HWiNFO_AutoRestart」タスクを選択
4. 右側の「トリガー」タブをクリック
5. 既存のトリガーを選択して「編集」をクリック
6. 以下の設定を変更:
   □ 「タスクを繰り返す」にチェックを入れる
   - 繰り返し間隔: 12時間（または11時間）
   - 継続時間: 無期限
   - 「有効」にチェック
7. 「OK」をクリック

■ 動作確認

タスクを手動で実行してテスト:
  タスクスケジューラで「HWiNFO_AutoRestart」を右クリック > 「実行」

または、コマンドラインから:
  cd C:\Users\tomoh\WasherInspection\scripts
  python hwinfo_auto_restart.py restart

■ トラブルシューティング

- HWiNFO64が見つからない
  → HWiNFO64のインストールパスを確認してください
  → scripts/hwinfo_auto_restart.py の find_hwinfo() 関数でパスを追加できまム

- 管理者権限エラー
  → タスクスケジューラで「最上位の特権で実行する」が有効になっているか確認
  → タスクを右クリック > 「プロパティ」> 「全般」タブ

- タスクが実行されない
  → タスクスケジューラの「履歴」タブでエラーを確認
  → Pythonパスが正しいか確認

