#!/bin/bash
# Windows用のGitコミットメッセージ修正スクリプト
# Git Bashで実行してください

# 修正マッピングを適用
git filter-branch -f --msg-filter '
python -c "
import sys
msg = sys.stdin.read()
fixes = {
    \"菫ｮ豁｣: PROJECT_STRUCTURE.mdの荳ｻ隕√ヵファイル隱ｬ譏弱そ繧ｯ繧ｷ繝ｧ繝ｳ繧呈峩譁ｰ\": \"修正: PROJECT_STRUCTURE.mdの主要ファイル説明セクションを更新\",
    \"譖ｴ譁ｰ: docs/PROJECT_STRUCTURE.mdのGitHub繝・・繝ｫ荳隕ｧ繧ゆｿｮ豁｣\": \"更新: docs/PROJECT_STRUCTURE.mdのGitHubドキュメントを更新\",
}
if msg.strip() in fixes:
    print(fixes[msg.strip()])
else:
    print(msg)
"
' -- --all


