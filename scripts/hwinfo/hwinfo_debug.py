#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HWiNFO詳細デバッグスクリプト - すべてのセンサー情報を表示"""

try:
    from hwinfo_reader import read_hwinfo_shared_memory
    import ctypes
    import sys
# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
    
    if sys.platform != 'win32':
        print("Windows only")
        sys.exit(1)
    
    # HWiNFO Shared Memoryの構造体をインポート
    from hwinfo_reader import (
        HWiNFO_SHARED_MEM_HEADER, HWiNFO_SENSOR_ELEMENT, HWiNFO_READING_ELEMENT,
        HWiNFO_SHMEM_NAMES, HWiNFO_SIGNATURES, kernel32, FILE_MAP_READ
    )
    
    print("=" * 70)
    print("HWiNFO詳細デバッグ - すべてのセンサーと読み取り値を表示")
    print("=" * 70)
    print()
    
    # Shared Memoryを開く
    shmem_handle = None
    shmem_ptr = None
    shmem_size = 1024 * 1024
    
    for shmem_name in HWiNFO_SHMEM_NAMES:
        try:
            handle = kernel32.OpenFileMappingW(FILE_MAP_READ, False, shmem_name)
            if handle and handle != 0:
                shmem_ptr = kernel32.MapViewOfFile(handle, FILE_MAP_READ, 0, 0, shmem_size)
                if shmem_ptr and shmem_ptr != 0:
                    shmem_handle = handle
                    print(f"[OK] HWiNFO Shared Memory opened: {shmem_name}")
                    break
        except Exception as e:
            continue
    
    if not shmem_handle:
        print("[ERROR] HWiNFO Shared Memory not found")
        print("  Available names tried:", HWiNFO_SHMEM_NAMES)
        sys.exit(1)
    
    try:
        # ヘッダーを読み取る
        header = HWiNFO_SHARED_MEM_HEADER()
        header_bytes = ctypes.create_string_buffer(ctypes.sizeof(HWiNFO_SHARED_MEM_HEADER))
        ctypes.memmove(header_bytes, ctypes.cast(shmem_ptr, ctypes.POINTER(ctypes.c_char)), 
                      ctypes.sizeof(HWiNFO_SHARED_MEM_HEADER))
        ctypes.memmove(ctypes.addressof(header), header_bytes, 
                      ctypes.sizeof(HWiNFO_SHARED_MEM_HEADER))
        
        if header.dwSignature not in HWiNFO_SIGNATURES:
            print(f"[ERROR] Invalid signature: {hex(header.dwSignature)}")
            print(f"  Expected one of: {[hex(s) for s in HWiNFO_SIGNATURES]}")
            sys.exit(1)
        
        print(f"Version: {header.dwVersion}, Revision: {header.dwRevision}")
        print(f"Sensors: {header.dwNumOfSensorElements}, Readings: {header.dwNumOfReadingElements}")
        print()
        
        # センサー情報を表示
        sensor_offset = header.dwOffsetOfSensorSection
        reading_offset = header.dwOffsetOfReadingSection
        sensor_size = ctypes.sizeof(HWiNFO_SENSOR_ELEMENT)
        
        print("=" * 70)
        print("センサー一覧:")
        print("=" * 70)
        
        for i in range(min(header.dwNumOfSensorElements, 50)):  # 最大50個まで
            sensor = HWiNFO_SENSOR_ELEMENT()
            offset = sensor_offset + i * sensor_size
            ctypes.memmove(ctypes.addressof(sensor), 
                          ctypes.cast(shmem_ptr + offset, ctypes.POINTER(ctypes.c_char * sensor_size)).contents, 
                          sensor_size)
            sensor_name = sensor.szSensorName.decode('utf-8', errors='ignore')
            sensor_inst = sensor.szSensorNameInst.decode('utf-8', errors='ignore')
            print(f"  [{i}] ID:{sensor.dwSensorID} Inst:{sensor.dwSensorInstance}")
            print(f"      Name: {sensor_name}")
            print(f"      Instance: {sensor_inst}")
        
        print()
        print("=" * 70)
        print("読み取り値一覧 (GPU関連):")
        print("=" * 70)
        
        reading_size = ctypes.sizeof(HWiNFO_READING_ELEMENT)
        
        # GPUセンサーIDを特定
        gpu_sensor_ids = []
        for i in range(min(header.dwNumOfSensorElements, 50)):
            sensor = HWiNFO_SENSOR_ELEMENT()
            offset = sensor_offset + i * sensor_size
            ctypes.memmove(ctypes.addressof(sensor), 
                          ctypes.cast(shmem_ptr + offset, ctypes.POINTER(ctypes.c_char * sensor_size)).contents, 
                          sensor_size)
            sensor_name = sensor.szSensorName.decode('utf-8', errors='ignore').upper()
            sensor_inst = sensor.szSensorNameInst.decode('utf-8', errors='ignore').upper()
            if ('GPU' in sensor_name or 'VIDEO' in sensor_name or 'GRAPHICS' in sensor_name or
                'RTX' in sensor_inst or 'GEFORCE' in sensor_inst):
                gpu_sensor_ids.append((sensor.dwSensorID, sensor.dwSensorInstance))
        
        # GPU関連の読み取り値を表示
        gpu_count = 0
        for i in range(min(header.dwNumOfReadingElements, 200)):  # 最大200個まで
            reading = HWiNFO_READING_ELEMENT()
            offset = reading_offset + i * reading_size
            ctypes.memmove(ctypes.addressof(reading), 
                          ctypes.cast(shmem_ptr + offset, ctypes.POINTER(ctypes.c_char * reading_size)).contents, 
                          reading_size)
            
            is_gpu = any(reading.dwSensorID == sid and reading.dwSensorInstance == si 
                        for sid, si in gpu_sensor_ids)
            
            if is_gpu:
                reading_name = reading.szReadingName.decode('utf-8', errors='ignore')
                reading_label = reading.szReadingLabel.decode('utf-8', errors='ignore')
                value = reading.tReading
                unit = reading.dwUnit
                print(f"  [{gpu_count}] {reading_name} = {value} {reading_label} (Unit: {hex(unit)})")
                gpu_count += 1
                if gpu_count >= 30:  # 最大30個まで
                    break
        
        print()
        print("=" * 70)
        print("統合された情報:")
        print("=" * 70)
        result = read_hwinfo_shared_memory()
        if result:
            for key, val in result.items():
                print(f"  {key}: {val}")
        else:
            print("  [ERROR] 情報の取得に失敗しました")
    
    finally:
        if shmem_handle:
            try:
                if shmem_ptr:
                    kernel32.UnmapViewOfFile(shmem_ptr)
                kernel32.CloseHandle(shmem_handle)
            except:
                pass

except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("hwinfo_reader module is required")

