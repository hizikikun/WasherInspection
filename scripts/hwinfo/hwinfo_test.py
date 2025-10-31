#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HWiNFO連携テストスクリプト"""

import sys

# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    from hwinfo_reader import read_hwinfo_shared_memory
    HAS_HWINFO = True
except Exception:
    HAS_HWINFO = False
    print("[ERROR] hwinfo_reader module not available")

if HAS_HWINFO:
    print("HWiNFO連携テスト")
    print("=" * 50)
    data = read_hwinfo_shared_memory()
    if data:
        print("[OK] HWiNFO Shared Memoryからデータを取得しました")
        print()
        print("GPU情報:")
        print(f"  使用率: {data.get('gpu_util_percent', 'N/A')}%")
        print(f"  温度: {data.get('gpu_temp_c', 'N/A')}°C")
        print(f"  電力: {data.get('gpu_power_w', 'N/A')}W")
        print(f"  メモリ使用: {data.get('gpu_mem_used_mb', 'N/A')} MB / {data.get('gpu_mem_total_mb', 'N/A')} MB")
        print()
        print("CPU情報:")
        print(f"  使用率: {data.get('cpu_percent', 'N/A')}%")
        print(f"  温度: {data.get('cpu_temp_c', 'N/A')}°C")
        print()
        print("メモリ情報:")
        print(f"  使用率: {data.get('mem_percent', 'N/A')}%")
        print(f"  使用量: {data.get('mem_used_mb', 'N/A')} MB / {data.get('mem_total_mb', 'N/A')} MB")
    else:
        print("[WARNING] HWiNFO Shared Memoryが利用できません")
        print("  - HWiNFOが起動しているか確認してください")
        print("  - HWiNFOの設定で「Shared Memory Support」が有効になっているか確認してください")

