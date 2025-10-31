#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HWiNFO連携状況確認スクリプト"""

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
    from hwinfo_reader import read_hwinfo_shared_memory, HWiNFO_SHMEM_NAMES
    import ctypes
    
    if sys.platform == 'win32':
        kernel32 = ctypes.windll.kernel32
        FILE_MAP_READ = 0x0004
        
        # Shared Memory接続確認
        connected = False
        for name in HWiNFO_SHMEM_NAMES:
            try:
                handle = kernel32.OpenFileMappingW(FILE_MAP_READ, False, name)
                if handle and handle != 0:
                    connected = True
                    kernel32.CloseHandle(handle)
                    break
            except:
                pass
        
        print("=" * 60)
        print("HWiNFO連携状況確認")
        print("=" * 60)
        print()
        print(f"Shared Memory接続: {'OK' if connected else 'NG'}")
        
        # データ取得確認
        data = read_hwinfo_shared_memory()
        if data:
            gpu_data_available = any([
                data.get('gpu_util_percent') is not None,
                data.get('gpu_temp_c') is not None,
                data.get('gpu_power_w') is not None,
                data.get('gpu_mem_used_mb') is not None
            ])
            
            print(f"データ取得: {'OK' if gpu_data_available else 'NG (接続は成功だがデータが取得できていません)'}")
            print()
            print("取得データ:")
            print(f"  GPU使用率: {data.get('gpu_util_percent', 'None')}")
            print(f"  GPU温度: {data.get('gpu_temp_c', 'None')}")
            print(f"  GPU電力: {data.get('gpu_power_w', 'None')}")
            print(f"  GPUメモリ: {data.get('gpu_mem_used_mb', 'None')} MB / {data.get('gpu_mem_total_mb', 'None')} MB")
            print()
            
            if gpu_data_available:
                print("連携ステータス: [完了] 正常に動作していまム")
            else:
                print("連携ステータス: [部分的完了]")
                print("  - 接続: OK")
                print("  - データ取得: NG (構造体の解析に問題があります)")
        else:
            print("データ取得: NG (HWiNFO Shared Memoryが利用できません)")
            print()
            print("連携ステータス: [未完了]")
        
        print("=" * 60)
        
except Exception as e:
    print(f"エラー: {e}")
    import traceback
    traceback.print_exc()

