#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk

STATUS_FILE = Path(__file__).resolve().parents[1] / 'logs' / 'training_status.json'


def human(seconds):
    if seconds is None:
        return "-"
    try:
        v = float(seconds)
    except Exception:
        return "-"
    if v >= 2 * 3600:
        return f"約{int(round(v/3600.0))}時間"
    if v >= 10 * 60:
        return f"約{int(round(v/60.0))}分"
    if v >= 60:
        mins = int(v // 60)
        secs = int(v % 60)
        return f"{mins}分{secs}秒"
    return f"{int(v)}秒"


def load_status():
    if not STATUS_FILE.exists():
        return None
    try:
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


class StatusViewer(tk.Tk):
    def __init__(self, refresh_ms=2000, always_on_top=False):
        super().__init__()
        self.title('学習ステータス（スパースモデリング）')
        self.geometry('540x360')
        if always_on_top:
            self.attributes('-topmost', True)

        # base styles
        default_font = ('Yu Gothic UI', 10)
        mono_font = ('Cascadia Mono', 10)

        container = ttk.Frame(self, padding=12)
        container.pack(fill='both', expand=True)

        self.lines = []
        bullets = [
            '最新進捗（安定表示／アプリ）',
            'テモデル: ',
            'テエポック: ',
            'テ訓練全体: ',
            'テ全体進捗: ',
            'テ全体経過: ',
            'テ全体残り: ',
            'テ完了予測: ',
            'テ精度: ',
            'テLR: ',
        ]

        for i, text in enumerate(bullets):
            lbl = ttk.Label(container, text=text, font=mono_font if i>0 else default_font, anchor='w')
            lbl.pack(fill='x', pady=(0 if i else 2, 2))
            self.lines.append(lbl)

        self.status_bar = ttk.Label(container, text='更新: -', font=default_font, anchor='e')
        self.status_bar.pack(fill='x', pady=(6, 0))

        self.refresh_ms = refresh_ms
        self.after(100, self.update_status)

    def update_status(self):
        s = load_status()
        if s is None:
            self.lines[0]['text'] = '最新進捗（安定表示／アプリ）'
            for i in range(1, len(self.lines)):
                self.lines[i]['text'] = ['テモデル: -','テエポック: -','テ訓練全体: -','テ全体進捗: -','テ全体経過: -','テ全体残り: -','テ完了予測: -','テ精度: -','テLR: -'][i-1]
            self.status_bar['text'] = '更新: 進捗ファイル未検出'
            self.after(self.refresh_ms, self.update_status)
            return

        # values with safe defaults
        model = s.get('model_name', '-')
        model_idx = s.get('model_index', 0)
        models_total = s.get('models_total', 0)
        epoch = s.get('epoch', 0)
        epochs_total = s.get('epochs_total', 0)
        ep_pct = s.get('epoch_progress_percent', 0)
        train_epoch_pct = s.get('training_epoch_percent', 0)
        overall_pct = s.get('overall_progress_percent', 0)
        overall_elapsed = s.get('overall_elapsed_human') or human(s.get('overall_elapsed_seconds'))
        overall_remaining = s.get('overall_remaining_human') or human(s.get('overall_remaining_est_seconds'))
        eta = s.get('overall_eta_human') or human(s.get('overall_estimated_total_seconds'))
        acc = s.get('accuracy_percent')
        vacc = s.get('val_accuracy_percent')
        lr = s.get('learning_rate', '-')

        # update UI (fixed layout, minimal flicker)
        self.lines[0]['text'] = '最新進捗（安定表示／アプリ）'
        self.lines[1]['text'] = f'テモデル: {model} ({model_idx}/{models_total})'
        self.lines[2]['text'] = f'テエポック: {epoch}/{epochs_total}（{ep_pct:.1f}%）'
        self.lines[3]['text'] = f'テ訓練全体（エポック換算）: {train_epoch_pct:.1f}%'
        self.lines[4]['text'] = f'テ全体進捗: {int(round(overall_pct))}%'
        self.lines[5]['text'] = f'テ全体経過: {overall_elapsed}'
        self.lines[6]['text'] = f'テ全体残り: {overall_remaining}'
        self.lines[7]['text'] = f'テ完了予測: {eta}'
        if acc is not None and vacc is not None:
            self.lines[8]['text'] = f'テ精度: {float(acc):.2f}% | 検証: {float(vacc):.2f}%'
        else:
            self.lines[8]['text'] = 'テ精度: -'
        self.lines[9]['text'] = f'テLR: {lr}'

        ts = s.get('timestamp')
        if ts:
            tstr = time.strftime('%H:%M:%S', time.localtime(ts))
        else:
            tstr = time.strftime('%H:%M:%S')
        self.status_bar['text'] = f'更新: {tstr}'

        self.after(self.refresh_ms, self.update_status)


if __name__ == '__main__':
    app = StatusViewer(refresh_ms=2000, always_on_top=False)
    app.mainloop()



