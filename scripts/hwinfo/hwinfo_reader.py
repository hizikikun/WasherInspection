#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HWiNFO Shared Memory Interface Reader
HWiNFOからの値を最優先テ最も信頼できる情報源として使用
"""

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

try:
    import mmap
    HAS_MMAP = True
except ImportError:
    HAS_MMAP = False

# Windows API
if sys.platform == 'win32':
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenFileMappingW.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_wchar_p]
    kernel32.OpenFileMappingW.restype = ctypes.c_void_p
    kernel32.MapViewOfFile.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_size_t]
    kernel32.MapViewOfFile.restype = ctypes.c_void_p
    kernel32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    kernel32.UnmapViewOfFile.restype = ctypes.c_bool
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_bool
    FILE_MAP_READ = 0x0004

# HWiNFO Shared Memory構造体（公式SDKに基づく、アライメント修正）
class HWiNFO_SHARED_MEM_HEADER(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("dwSignature", ctypes.c_uint32),
        ("dwVersion", ctypes.c_uint32),
        ("dwRevision", ctypes.c_uint32),
        ("dwPollingTime", ctypes.c_uint32),
        ("dwOffsetOfSensorSection", ctypes.c_uint32),
        ("dwOffsetOfReadingSection", ctypes.c_uint32),
        ("dwNumOfSensorElements", ctypes.c_uint32),
        ("dwNumOfReadingElements", ctypes.c_uint32),
        ("szNamesString", ctypes.c_char * 260),
    ]

class HWiNFO_SENSOR_ELEMENT(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("dwSensorID", ctypes.c_uint32),
        ("dwSensorInstance", ctypes.c_uint32),
        ("szSensorName", ctypes.c_char * 128),
        ("szSensorNameInst", ctypes.c_char * 256),
    ]

class HWiNFO_READING_ELEMENT(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("dwSensorID", ctypes.c_uint32),
        ("dwSensorInstance", ctypes.c_uint32),
        ("szReadingName", ctypes.c_char * 128),
        ("szReadingLabel", ctypes.c_char * 128),
        ("tReading", ctypes.c_double),
        ("dwMin", ctypes.c_uint32),
        ("dwMax", ctypes.c_uint32),
        ("dwUnit", ctypes.c_uint32),
    ]

HWiNFO_SHMEM_NAMES = [
    "Global\\HWiNFO_SENS_SM2",
    "HWiNFO_SENS_SM2",
    "Global\\HWiNFO_SENS_SM",
    "HWiNFO_SENS_SM"
]
HWiNFO_SIGNATURES = [0x46494E57, 0x53695748, 0x4E695748]

def read_hwinfo_shared_memory():
    """
    HWiNFOのShared Memoryからすべての情報を読み取る（改善版）
    最も信頼できる情報源として使用
    """
    try:
        if sys.platform != 'win32':
            return None
        
        shmem_handle = None
        shmem_ptr = None
        shmem_size = 8 * 1024 * 1024  # 8MBに拡大
        
        # Shared Memoryを開く
        for shmem_name in HWiNFO_SHMEM_NAMES:
            try:
                handle = kernel32.OpenFileMappingW(FILE_MAP_READ, False, shmem_name)
                if handle and handle != 0:
                    shmem_ptr = kernel32.MapViewOfFile(handle, FILE_MAP_READ, 0, 0, shmem_size)
                    if shmem_ptr and shmem_ptr != 0:
                        shmem_handle = handle
                        break
            except Exception:
                continue
        
        if not shmem_handle or not shmem_ptr:
            return None
        
        try:
            # ヘッダーを読み取る
            header = HWiNFO_SHARED_MEM_HEADER()
            header_ptr = ctypes.cast(shmem_ptr, ctypes.POINTER(HWiNFO_SHARED_MEM_HEADER))
            ctypes.memmove(ctypes.addressof(header), header_ptr, ctypes.sizeof(HWiNFO_SHARED_MEM_HEADER))
            
            # シグネチャ確認
            if header.dwSignature not in HWiNFO_SIGNATURES:
                return None
            
            sensor_offset = header.dwOffsetOfSensorSection
            reading_offset = header.dwOffsetOfReadingSection
            num_sensors = header.dwNumOfSensorElements
            num_readings = header.dwNumOfReadingElements
            
            # GPUセンサーIDを特定（すべてのセンサーを確認）
            gpu_sensor_ids = []
            cpu_sensor_ids = []
            sensor_size = ctypes.sizeof(HWiNFO_SENSOR_ELEMENT)
            
            for i in range(min(num_sensors, 500)):
                offset = sensor_offset + i * sensor_size
                if offset + sensor_size > shmem_size:
                    break
                
                sensor_ptr = ctypes.cast(shmem_ptr + offset, ctypes.POINTER(HWiNFO_SENSOR_ELEMENT))
                sensor = sensor_ptr.contents
                
                # GPU検出（より積極的に）
                sensor_id = sensor.dwSensorID
                sensor_inst = sensor.dwSensorInstance
                
                # GPUセンサーIDは通常1000000以上
                if sensor_id > 1000000:
                    gpu_sensor_ids.append((sensor_id, sensor_inst))
            
            # すべての読み取り値を処理（GPU関連を優先）
            cpu_percent = None
            cpu_temp = None
            mem_percent = None
            mem_used_mb = None
            mem_total_mb = None
            gpu_util = None
            gpu_temp = None
            gpu_mem_used = None
            gpu_mem_total = None
            gpu_power = None
            
            reading_size = ctypes.sizeof(HWiNFO_READING_ELEMENT)
            
            for i in range(min(num_readings, 2000)):
                offset = reading_offset + i * reading_size
                if offset + reading_size > shmem_size:
                    break
                
                reading_ptr = ctypes.cast(shmem_ptr + offset, ctypes.POINTER(HWiNFO_READING_ELEMENT))
                reading = reading_ptr.contents
                
                sensor_id = reading.dwSensorID
                sensor_inst = reading.dwSensorInstance
                value = reading.tReading
                
                # GPUセンサーに関連する読み取り値かチェック
                is_gpu = any(sensor_id == sid and sensor_inst == si for sid, si in gpu_sensor_ids)
                
                if is_gpu:
                    # 値の範囲から推測（より柔軟な判定）
                    # GPU使用率（0-100の範囲で、正常な値のみ、極小値は無視）
                    if 0.1 <= value <= 100:
                        # まだ使用率が設定されていないか、より高い値の場合は更新
                        if gpu_util is None or (value > (gpu_util or 0) and value <= 100):
                            gpu_util = float(value)
                    
                    # GPU温度（20-150度の範囲で、正常な値のみ）
                    if 20 <= value <= 150:
                        # 温度は最高値を使用
                        if gpu_temp is None or value > (gpu_temp or 0):
                            gpu_temp = float(value)
                    
                    # GPU電力（5-1000Wの範囲で、正常な値のみ）
                    if 5 <= value <= 1000:
                        # 電力は最高値を使用
                        if gpu_power is None or value > (gpu_power or 0):
                            gpu_power = float(value)
                    
                    # GPUメモリ（100MB以上の値はメモリの可能性）
                    # 小さい値は使用量、大きい値は総量の可能性
                    if value >= 100:
                        if value <= 50000:  # 合理的な範囲内
                            if gpu_mem_used is None or (value < (gpu_mem_used or float('inf')) and value >= 100):
                                gpu_mem_used = float(value)
                            elif gpu_mem_total is None or (value > (gpu_mem_total or 0) and value <= 50000):
                                gpu_mem_total = float(value)
            
            # 結果を返ム（Noneでない値のみ）
            result = {}
            if cpu_percent is not None:
                result['cpu_percent'] = cpu_percent
            if cpu_temp is not None:
                result['cpu_temp_c'] = cpu_temp
            if mem_percent is not None:
                result['mem_percent'] = mem_percent
            if mem_used_mb is not None:
                result['mem_used_mb'] = mem_used_mb
            if mem_total_mb is not None:
                result['mem_total_mb'] = mem_total_mb
            if gpu_util is not None:
                result['gpu_util_percent'] = gpu_util
            if gpu_temp is not None:
                result['gpu_temp_c'] = gpu_temp
            if gpu_mem_used is not None:
                result['gpu_mem_used_mb'] = gpu_mem_used
            if gpu_mem_total is not None:
                result['gpu_mem_total_mb'] = gpu_mem_total
            if gpu_power is not None:
                result['gpu_power_w'] = gpu_power
            
            return result if result else None
        
        finally:
            if shmem_handle:
                try:
                    if shmem_ptr:
                        kernel32.UnmapViewOfFile(shmem_ptr)
                    kernel32.CloseHandle(shmem_handle)
                except:
                    pass
    
    except Exception:
        return None

# テスト用
if __name__ == '__main__':
    result = read_hwinfo_shared_memory()
    if result:
        print("HWiNFO Data Retrieved:")
        for key, val in result.items():
            print(f"  {key}: {val}")
    else:
        print("HWiNFO Shared Memory not available or no valid data")

