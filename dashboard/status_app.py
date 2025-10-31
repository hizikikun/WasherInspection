#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import time
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse


APP = FastAPI(title="WasherInspection Training Status Dashboard")
STATUS_FILE = Path(__file__).resolve().parents[1] / 'logs' / 'training_status.json'


def human(s: Any) -> str:
    if s is None:
        return "-"
    try:
        v = float(s)
    except Exception:
        return str(s)
    if v >= 2 * 3600:
        return f"約{int(round(v/3600.0))}時間"
    if v >= 10 * 60:
        return f"約{int(round(v/60.0))}分"
    if v >= 60:
        mins = int(v // 60)
        secs = int(v % 60)
        return f"{mins}分{secs}秒"
    return f"{int(v)}秒"


def load_status() -> Dict[str, Any]:
    if not STATUS_FILE.exists():
        return {
            "stage": "待機中",
            "message": "statusファイル未生成",
            "timestamp": time.time(),
        }
    try:
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return {
            "stage": "読込エラー",
            "message": "JSONの読み込みに失敗",
            "timestamp": time.time(),
        }
    # 既存フォーマットに人間可読を補完（後方互換）
    if 'overall_elapsed_human' not in data and 'overall_elapsed_seconds' in data:
        data['overall_elapsed_human'] = human(data.get('overall_elapsed_seconds'))
    if 'overall_remaining_human' not in data and 'overall_remaining_est_seconds' in data:
        data['overall_remaining_human'] = human(data.get('overall_remaining_est_seconds'))
    if 'overall_eta_human' not in data and 'overall_estimated_total_seconds' in data:
        data['overall_eta_human'] = human(data.get('overall_estimated_total_seconds'))
    if 'item_elapsed_human' not in data and 'item_elapsed_seconds' in data:
        data['item_elapsed_human'] = human(data.get('item_elapsed_seconds'))
    if 'item_remaining_human' not in data and 'item_remaining_est_seconds' in data:
        data['item_remaining_human'] = human(data.get('item_remaining_est_seconds'))
    return data


@APP.get('/status')
def get_status() -> JSONResponse:
    return JSONResponse(load_status())


@APP.get('/')
def index() -> HTMLResponse:
    html = """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Training Status</title>
  <style>
    :root { --fg:#111; --muted:#666; --ok:#0a7; --bg:#fafafa; }
    body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; background: var(--bg); color: var(--fg); }
    .card { max-width: 720px; background:#fff; border:1px solid #eee; border-radius:12px; padding:20px; box-shadow:0 2px 8px rgba(0,0,0,.04); }
    .title { font-size: 18px; margin-bottom: 10px; }
    .row { display:flex; gap:12px; flex-wrap: wrap; }
    .pill { background:#f3f5f7; border-radius:999px; padding:8px 12px; font-variant-numeric: tabular-nums; }
    .mono { font-variant-numeric: tabular-nums; }
    .muted { color: var(--muted); }
    .bar { height:8px; background:#eef2f4; border-radius:999px; overflow:hidden; margin:12px 0 6px; }
    .bar > div { height:100%; background:linear-gradient(90deg,#2dd4bf,#22c55e); width:0%; transition: width .6s ease; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 8px 16px; }
  </style>
  <script>
    async function fetchStatus(){
      try{
        const res = await fetch('/status', {cache:'no-store'});
        const s = await res.json();
        // 上段（全体）
        document.getElementById('overall-line').textContent =
          `【全体】${(s.overall_progress_percent??0).toFixed(0)}%｜経過 ${s.overall_elapsed_human||'-'}｜残り ${s.overall_remaining_human||'-'}｜完了予測 ${s.overall_eta_human||'-'}`;
        const w = Math.max(0, Math.min(100, Number(s.overall_progress_percent||0)));
        document.getElementById('overall-bar').style.width = w + '%';
        // 下段（モデル）
        const model = s.model_name || '-';
        const idx = s.model_index || 0;
        const tot = s.models_total || 0;
        const ep = s.epoch || 0;
        const epTot = s.epochs_total || 0;
        const acc = s.accuracy_percent!=null ? s.accuracy_percent.toFixed(2) : '-';
        const vacc = s.val_accuracy_percent!=null ? s.val_accuracy_percent.toFixed(2) : '-';
        const lr = s.learning_rate || '-';
        document.getElementById('model-line').textContent =
          `【モデル】${model} (${idx}/${tot})｜エポック ${ep}/${epTot}（${(s.epoch_progress_percent||0).toFixed(1)}%）｜精度 ${acc}%｜検証 ${vacc}%｜LR ${lr}`;
        const mw = Math.max(0, Math.min(100, Number(s.epoch_progress_percent||0)));
        document.getElementById('model-bar').style.width = mw + '%';
        // 更新時刻
        const ts = s.timestamp? new Date(s.timestamp*1000) : new Date();
        document.getElementById('updated').textContent = ts.toLocaleTimeString();
      }catch(e){
        document.getElementById('overall-line').textContent = '読み込みエラー';
      }
    }
    setInterval(fetchStatus, 2000);
    window.addEventListener('load', fetchStatus);
  </script>
  
</head>
<body>
  <div class="card">
    <div class="title">学習ステータス</div>
    <div class="mono" id="overall-line">読み込み中...</div>
    <div class="bar"><div id="overall-bar"></div></div>
    <div class="mono" id="model-line" style="margin-top:8px;">-</div>
    <div class="bar"><div id="model-bar"></div></div>
    <div class="muted" style="margin-top:8px;">更新: <span id="updated">-</span></div>
  </div>
</body>
</html>
    """
    return HTMLResponse(html)


app = APP  # for uvicorn discovery





