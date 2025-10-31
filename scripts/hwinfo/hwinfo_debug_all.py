#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HWiNFOすべての読み取り値を表示（デバッグ用）"""

import sys
# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
import ctypes
from hwinfo_reader import (
    HWiNFO_SHARED_MEM_HEADER, HWiNFO_SENSOR_ELEMENT, HWiNFO_READING_ELEMENT,
    HWiNFO_SHMEM_NAMES, HWiNFO_SIGNATURES, kernel32, FILE_MAP_READ
)

if sys.platform != 'win32':
    print("Windows only")
    sys.exit(1)

shmem_handle = None
shmem_ptr = None
shmem_size = 4 * 1024 * 1024

# Shared Memoryを開く
for shmem_name in HWiNFO_SHMEM_NAMES:
    try:
        handle = kernel32.OpenFileMappingW(FILE_MAP_READ, False, shmem_name)
        if handle and handle != 0:
            shmem_ptr = kernel32.MapViewOfFile(handle, FILE_MAP_READ, 0, 0, shmem_size)
            if shmem_ptr and shmem_ptr != 0:
                shmem_handle = handle
                print(f"[OK] Opened: {shmem_name}")
                break
    except:
        continue

if not shmem_handle:
    print("[ERROR] Cannot open HWiNFO Shared Memory")
    sys.exit(1)

try:
    # ヘッダーを読み取る
    header = HWiNFO_SHARED_MEM_HEADER()
    header_ptr = ctypes.cast(shmem_ptr, ctypes.POINTER(HWiNFO_SHARED_MEM_HEADER))
    ctypes.memmove(ctypes.addressof(header), header_ptr, ctypes.sizeof(HWiNFO_SHARED_MEM_HEADER))
    
    if header.dwSignature not in HWiNFO_SIGNATURES:
        print(f"[ERROR] Invalid signature: {hex(header.dwSignature)}")
        sys.exit(1)
    
    print(f"Version: {header.dwVersion}, Revision: {header.dwRevision}")
    print(f"Sensors: {header.dwNumOfSensorElements}, Readings: {header.dwNumOfReadingElements}")
    print()
    
    sensor_offset = header.dwOffsetOfSensorSection
    reading_offset = header.dwOffsetOfReadingSection
    sensor_size = ctypes.sizeof(HWiNFO_SENSOR_ELEMENT)
    reading_size = ctypes.sizeof(HWiNFO_READING_ELEMENT)
    
    # GPUセンサーIDを特定（すべてのセンサーを確認）
    print("=" * 80)
    print("GPU関連と思われるセンサー:")
    print("=" * 80)
    gpu_sensor_ids = []
    
    for i in range(min(header.dwNumOfSensorElements, 100)):
        offset = sensor_offset + i * sensor_size
        sensor_ptr = ctypes.cast(shmem_ptr + offset, ctypes.POINTER(HWiNFO_SENSOR_ELEMENT))
        sensor = sensor_ptr.contents
        
        # 複数のエンコーディングで試み
        sensor_name = ""
        sensor_inst = ""
        for encoding in ['utf-8', 'shift-jis', 'windows-1252', 'latin-1', 'cp1252']:
            try:
                sensor_name = sensor.szSensorName.decode(encoding, errors='ignore').strip()
                sensor_inst = sensor.szSensorNameInst.decode(encoding, errors='ignore').strip()
                if sensor_name or sensor_inst:
                    break
            except:
                continue
        
        # GPU関連と思われるセンサーを表示
        if (sensor.dwSensorID > 1000000 or 'GPU' in sensor_name.upper() or 
            'RTX' in sensor_inst.upper() or 'GEFORCE' in sensor_inst.upper() or
            'NVIDIA' in sensor_inst.upper()):
            print(f"  [{i}] ID:{sensor.dwSensorID:10d} Inst:{sensor.dwSensorInstance}")
            try:
                print(f"      Name: '{sensor_name}' | Instance: '{sensor_inst}'")
            except:
                print(f"      Name: [binary] | Instance: [binary]")
            if sensor.dwSensorID > 1000000:
                gpu_sensor_ids.append((sensor.dwSensorID, sensor.dwSensorInstance))
    
    print()
    print(f"GPU Sensor IDs found: {len(gpu_sensor_ids)}")
    print()
    
    # すべての読み取り値を表示（GPUセンサーに関連するもの）
    print("=" * 80)
    print("GPUセンサーに関連する読み取り値:")
    print("=" * 80)
    
    for i in range(min(header.dwNumOfReadingElements, 500)):
        offset = reading_offset + i * reading_size
        reading_ptr = ctypes.cast(shmem_ptr + offset, ctypes.POINTER(HWiNFO_READING_ELEMENT))
        reading = reading_ptr.contents
        
        # GPUセンサーに関連するかチェック
        is_gpu_related = any(reading.dwSensorID == sid and reading.dwSensorInstance == si 
                            for sid, si in gpu_sensor_ids)
        if not is_gpu_related and reading.dwSensorID > 1000000:
            is_gpu_related = True
        
        if is_gpu_related:
            reading_name = ""
            reading_label = ""
            for encoding in ['utf-8', 'shift-jis', 'windows-1252', 'latin-1', 'cp1252']:
                try:
                    reading_name = reading.szReadingName.decode(encoding, errors='ignore').strip()
                    reading_label = reading.szReadingLabel.decode(encoding, errors='ignore').strip()
                    if reading_name or reading_label:
                        break
                except:
                    continue
            
            value = reading.tReading
            unit = reading.dwUnit
            
            print(f"  [{i}] ID:{reading.dwSensorID:10d} Inst:{reading.dwSensorInstance}")
            try:
                print(f"      Name: '{reading_name}' | Label: '{reading_label}'")
            except:
                print(f"      Name: [binary] | Label: [binary]")
            print(f"      Value: {value:.2f} | Unit: {hex(unit)}")
            
            # 値の範囲から推測
            if 0 <= value <= 100:
                print(f"      -> 可能性: GPU使用率 ({value}%)")
            elif 20 <= value <= 150:
                print(f"      -> 可能性: GPU温度 ({value}°C)")
            elif 5 <= value <= 500:
                print(f"      -> 可能性: GPU電力 ({value}W)")
            elif value >= 100:
                print(f"      -> 可能性: GPUメモリ ({value} MB)")
            print()

finally:
    if shmem_handle:
        try:
            if shmem_ptr:
                kernel32.UnmapViewOfFile(shmem_ptr)
            kernel32.CloseHandle(shmem_handle)
        except:
            pass

