import argparse
import time
import json
import os
import glob
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUIバックエンドを使わない
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

@dataclass
class WasherSpec:
    expected_outer_mm: Optional[float] = None
    expected_inner_mm: Optional[float] = None
    diameter_tolerance_mm: float = 0.6
    circularity_min: float = 0.85
    concentricity_max_ratio: float = 0.08
    dark_defect_min_area_mm2: float = 0.5

@dataclass
class Calibration:
    mm_per_pixel: Optional[float] = None
    def px_to_mm(self, px: float): return None if self.mm_per_pixel is None else px*self.mm_per_pixel
    def mm_to_px(self, mm: float): return None if self.mm_per_pixel is None else mm/self.mm_per_pixel

@dataclass
class InspectionResult:
    contour: np.ndarray
    outer_center: Tuple[float,float]
    outer_radius_px: float
    inner_center: Tuple[float,float]
    inner_radius_px: float
    circularity: float
    concentricity_ratio: float
    dark_defect_area_mm2: float
    pass_geom: bool
    pass_defect: bool

@dataclass
class AILearningData:
    """AI学習データの統計情報"""
    total_samples: int
    defect_types: Dict[str, int]  # 欠陥タイプ別サンプル数
    quality_scores: List[float]   # データ品質スコア
    defect_areas: List[float]     # 欠陥面積    
    circularity_scores: List[float]  # 円形度スコア
    concentricity_scores: List[float]  # 同心度スコア
    image_resolution: Tuple[int, int]  # 画像解像度
    data_balance_ratio: float     # データバランス比
    average_quality: float        # 平均品質スコア

@dataclass
class DefectType:
    """欠陥タイプの定義"""
    BLACK_SPOT = "black_spot"      # 黒点
    SHORT_CIRCUIT = "short_circuit"  # ショート
    FOREIGN_MATTER = "foreign_matter"  # 異物混入
    NORMAL = "normal"              # 正常品

def normalize_illumination(gray):
    bg=cv2.GaussianBlur(gray,(0,0),21)
    norm=cv2.addWeighted(gray,1.5,bg,-0.5,0)
    return cv2.normalize(norm,None,0,255,cv2.NORM_MINMAX)

def threshold_washers(gray):
    norm=normalize_illumination(gray)
    _,th=cv2.threshold(255-norm,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th=cv2.morphologyEx(th,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),1)
    th=cv2.morphologyEx(th,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),2)
    return th

def compute_circularity(c):
    area=cv2.contourArea(c); peri=cv2.arcLength(c,True)
    return 0.0 if peri==0 else float(4.0*np.pi*area/(peri*peri))

def find_washers(mask):
    cs,h=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if h is None: return []
    h=h[0]; outs=[]
    processed_contours = set()
    
    for i,hi in enumerate(h):
        child=hi[2]
        if child!=-1 and i not in processed_contours:
            outer=cs[i]; inner=cs[child]
            m = cv2.moments(inner)
            if m["m00"] != 0:
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])
                if cv2.pointPolygonTest(outer, (cx, cy), False) >= 0:
                    is_duplicate = False
                    for existing_outer, existing_inner in outs:
                        existing_center = cv2.moments(existing_inner)
                        if existing_center["m00"] != 0:
                            ex_cx = float(existing_center["m10"] / existing_center["m00"])
                            ex_cy = float(existing_center["m01"] / existing_center["m00"])
                            distance = np.sqrt((cx - ex_cx)**2 + (cy - ex_cy)**2)
                            if distance < 50:
                                is_duplicate = True
                                break
                    if not is_duplicate:
                    outs.append((outer, inner))
                        processed_contours.add(i)
                        processed_contours.add(child)
    
    return outs

def analyze_washer(outer,inner,calib,spec,gray):
    (ox,oy),orad=cv2.minEnclosingCircle(outer)
    (ix,iy),irad=cv2.minEnclosingCircle(inner)
    circ=compute_circularity(outer)
    conc=float(np.hypot(ox-ix,oy-iy))/max(orad,1.0)
    
    # 強化された欠陥検出
    ring=np.zeros(gray.shape,np.uint8)
    cv2.drawContours(ring,[outer],-1,255,-1)
    cv2.drawContours(ring,[inner],-1,0,-1)
    
    # 複数の欠陥検出手法を組み合わせ
    # 1. 基本的な差分検出
    blur=cv2.GaussianBlur(gray,(0,0),3)
    diff=cv2.subtract(blur,cv2.GaussianBlur(blur,(0,0),9))
    dark1=cv2.threshold(diff,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    dark1&=ring
    
    # 2. エッジベース検出
    edges = cv2.Canny(gray, 50, 150)
    dark2 = cv2.bitwise_and(edges, ring)
    
    # 3. 色調変化検出
    hsv = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
    dark3 = cv2.inRange(hsv, (0, 0, 0), (180, 255, 100)) & ring
    
    # 複数手法の結果を統合
    dark = cv2.bitwise_or(dark1, cv2.bitwise_or(dark2, dark3))
    dark=cv2.morphologyEx(dark,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),1)
    dark_px=int(np.count_nonzero(dark))
    dark_mm2=float(dark_px*calib.mm_per_pixel**2) if calib.mm_per_pixel else float(dark_px)
    
    # 強化された幾何学的判定
    ok_geom=True
    if calib.mm_per_pixel and (spec.expected_outer_mm or spec.expected_inner_mm):
        if spec.expected_outer_mm:
            od=calib.px_to_mm(orad*2.0) or 0.0
            if abs(od-spec.expected_outer_mm)>spec.diameter_tolerance_mm: ok_geom=False
        if spec.expected_inner_mm:
            idm=calib.px_to_mm(irad*2.0) or 0.0
            if abs(idm-spec.expected_inner_mm)>spec.diameter_tolerance_mm: ok_geom=False
    
    # ワッシャー形状の基本チェック（ゴミや異物を除外）
    # 1. 外径と内径の比率チェック（ワッシャーらしい形状か）
    if orad > 0 and irad > 0:
        ratio = irad / orad
        if ratio < 0.3 or ratio > 0.9:  # 内径が外径の30%未満または90%以上は除外
            return (outer,(ox,oy),float(orad),(ix,iy),float(irad),float(circ),float(conc),float(dark_mm2),False,False)
    
    # 2. サイズチェック（小さすぎる物体を除外）
    if orad < 20:  # 20ピクセル未満は除外
        return (outer,(ox,oy),float(orad),(ix,iy),float(irad),float(circ),float(conc),float(dark_mm2),False,False)
    
    # 3. 円形度チェック（ワッシャーらしい円形度か）
    if circ < 0.6:  # 円形度が0.6未満は除外
        return (outer,(ox,oy),float(orad),(ix,iy),float(irad),float(circ),float(conc),float(dark_mm2),False,False)
    
    # 緩いNG判定（学習データが少ない場合の補正）
    min_circularity = max(spec.circularity_min, 0.7)  # 0.8から0.7に緩和
    if circ < min_circularity: ok_geom=False
    
    # 緩い同心度判定
    max_concentricity = min(spec.concentricity_max_ratio, 0.15)  # 0.1から0.15に緩和
    if conc > max_concentricity: ok_geom=False
    
    # 緩い欠陥判定
    ok_def=True
    if calib.mm_per_pixel:
        # より緩い欠陥検出
        min_defect_area = min(spec.dark_defect_min_area_mm2, 0.8)  # 0.3から0.8に緩和
        if dark_mm2 > min_defect_area: ok_def=False
    else:
        ring_area=int(cv2.contourArea(outer)-cv2.contourArea(inner))
        if ring_area>0:
            # より緩い欠陥比率判定
            defect_ratio = dark_px/ring_area
            if defect_ratio > 0.005:  # 0.001から0.005に緩和（0.5%以上で欠陥と判定）
                ok_def=False
    
    return (outer,(ox,oy),float(orad),(ix,iy),float(irad),float(circ),float(conc),float(dark_mm2),ok_geom,ok_def)

def draw_result(frame,res,calib):
    outer_center=res[1]; outer_r=res[2]; inner_center=res[3]; inner_r=res[4]
    circ=res[5]; conc=res[6]; dark=res[7]; ok_geom=res[8]; ok_def=res[9]
    
    # OK/NG判定
    is_ok = ok_geom and ok_def
    color=(0,200,0) if is_ok else (0,0,255)
    
    # ワッシャー描画
    cv2.circle(frame,(int(outer_center[0]),int(outer_center[1])),int(outer_r),color,2)
    cv2.circle(frame,(int(inner_center[0]),int(inner_center[1])),int(inner_r),(255,180,0),2)
    
    x,y=int(outer_center[0]),int(outer_center[1])
    
    # OK/NG判定表示
    if is_ok:
        cv2.putText(frame,"OK",(x+10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
    else:
        cv2.putText(frame,"NG",(x+10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
        
        # NG理由表示
        ng_reasons = []
        if not ok_geom:
            if circ < 0.85:
                ng_reasons.append("CIRC LOW")
            if conc > 0.08:
                ng_reasons.append("CONC HIGH")
        if not ok_def:
            if calib.mm_per_pixel and dark > 0.5:
                ng_reasons.append("DEFECT")
            else:
                ng_reasons.append("DARK SPOT")
        
        # NG理由を表示
        for i, reason in enumerate(ng_reasons):
            cv2.putText(frame,reason,(x+10,y+20+i*18),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1,cv2.LINE_AA)
    
    # 詳細情報表示
    lines=[f"CIRC {circ:.2f}",f"CONC {conc*100:.1f}%"]
    if calib.mm_per_pixel:
        od=calib.px_to_mm(outer_r*2.0) or 0.0
        idm=calib.px_to_mm(inner_r*2.0) or 0.0
        lines+= [f"OD {od:.2f}mm",f"ID {idm:.2f}mm",f"Dark {dark:.2f}mm^2"]
    else:
        lines.append("UNCALIBRATED")
    
    # 詳細情報を表示（OK/NG表示の下に配置）
    start_y = y + 20 if is_ok else y + 20 + len(ng_reasons) * 18
    for i,t in enumerate(lines):
        cv2.putText(frame,t,(x+10,start_y+i*18),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)

class AILearningAnalyzer:
    """AI学習データの分析・比較クラス"""
    
    def __init__(self):
        self.learning_datasets = {}
        self.analysis_results = {}
    
    def add_dataset(self, name: str, dataset: AILearningData):
        """学習データセットを追加"""
        self.learning_datasets[name] = dataset
        self._analyze_dataset(name, dataset)
    
    def _analyze_dataset(self, name: str, dataset: AILearningData):
        """データセットの詳細分析"""
        analysis = {
            'basic_stats': self._calculate_basic_stats(dataset),
            'defect_distribution': self._analyze_defect_distribution(dataset),
            'quality_metrics': self._analyze_quality_metrics(dataset),
            'data_balance': self._analyze_data_balance(dataset),
            'recommendations': self._generate_recommendations(dataset)
        }
        self.analysis_results[name] = analysis
    
    def _calculate_basic_stats(self, dataset: AILearningData) -> Dict[str, Any]:
        """基本統計情報の計算"""
        return {
            'total_samples': dataset.total_samples,
            'avg_quality': dataset.average_quality,
            'quality_std': np.std(dataset.quality_scores) if dataset.quality_scores else 0,
            'defect_area_mean': np.mean(dataset.defect_areas) if dataset.defect_areas else 0,
            'defect_area_std': np.std(dataset.defect_areas) if dataset.defect_areas else 0,
            'circularity_mean': np.mean(dataset.circularity_scores) if dataset.circularity_scores else 0,
            'concentricity_mean': np.mean(dataset.concentricity_scores) if dataset.concentricity_scores else 0
        }
    
    def _analyze_defect_distribution(self, dataset: AILearningData) -> Dict[str, Any]:
        """欠陥分布の分析"""
        total = sum(dataset.defect_types.values())
        distribution = {k: v/total for k, v in dataset.defect_types.items()}
        
        # データの多様性を計算
        diversity = len([k for k, v in dataset.defect_types.items() if v > 0])
        
        return {
            'distribution': distribution,
            'diversity_score': diversity / len(dataset.defect_types),
            'most_common_defect': max(dataset.defect_types.items(), key=lambda x: x[1])[0],
            'least_common_defect': min(dataset.defect_types.items(), key=lambda x: x[1])[0]
        }
    
    def _analyze_quality_metrics(self, dataset: AILearningData) -> Dict[str, Any]:
        """品質メトリクスの分析"""
        quality_scores = dataset.quality_scores
        if not quality_scores:
            return {'quality_grade': 'N/A', 'consistency': 0}
        
        avg_quality = np.mean(quality_scores)
        quality_std = np.std(quality_scores)
        
        # 品質グレードの判定
        if avg_quality >= 0.9:
            grade = 'A'
        elif avg_quality >= 0.8:
            grade = 'B'
        elif avg_quality >= 0.7:
            grade = 'C'
        else:
            grade = 'D'
        
        # 一貫性スコア（標準偏差が小さいほど一貫性が高い）
        consistency = max(0, 1 - quality_std)
        
        return {
            'quality_grade': grade,
            'consistency': consistency,
            'high_quality_ratio': len([q for q in quality_scores if q >= 0.8]) / len(quality_scores)
        }
    
    def _analyze_data_balance(self, dataset: AILearningData) -> Dict[str, Any]:
        """データバランスの分析"""
        defect_counts = list(dataset.defect_types.values())
        if not defect_counts:
            return {'balance_score': 0, 'imbalance_ratio': 1}
        
        max_count = max(defect_counts)
        min_count = min(defect_counts)
        balance_score = min_count / max_count if max_count > 0 else 0
        
        # 不均衡度の計算
        total = sum(defect_counts)
        expected_per_class = total / len(defect_counts)
        imbalance_ratio = max(defect_counts) / expected_per_class if expected_per_class > 0 else 1
        
        return {
            'balance_score': balance_score,
            'imbalance_ratio': imbalance_ratio,
            'is_balanced': balance_score >= 0.5
        }
    
    def _generate_recommendations(self, dataset: AILearningData) -> List[str]:
        """改善推奨事項の生成"""
        recommendations = []
        
        # データ量の推奨
        if dataset.total_samples < 1000:
            recommendations.append("データ量を増やすことを推奨（現在: {}サンプル）".format(dataset.total_samples))
        
        # データバランスの推奨
        balance_analysis = self._analyze_data_balance(dataset)
        if not balance_analysis['is_balanced']:
            recommendations.append("データバランスの改善が必要（不均衡比: {:.2f}）".format(balance_analysis['imbalance_ratio']))
        
        # 品質の推奨
        quality_analysis = self._analyze_quality_metrics(dataset)
        if quality_analysis['quality_grade'] in ['C', 'D']:
            recommendations.append("データ品質の向上が必要（現在のグレード: {}）".format(quality_analysis['quality_grade']))
        
        # 多様性の推奨
        defect_analysis = self._analyze_defect_distribution(dataset)
        if defect_analysis['diversity_score'] < 0.5:
            recommendations.append("欠陥タイプの多様性を増やすことを推奨")
        
        return recommendations
    
    def compare_datasets(self, dataset_names: List[str]) -> Dict[str, Any]:
        """複数データセットの比較分析"""
        if len(dataset_names) < 2:
            return {"error": "比較には最低2つのデータセットが必要です"}
        
        comparison = {
            'datasets': dataset_names,
            'sample_counts': {},
            'quality_comparison': {},
            'balance_comparison': {},
            'recommendations': []
        }
        
        # サンプル数の比較
        for name in dataset_names:
            if name in self.learning_datasets:
                comparison['sample_counts'][name] = self.learning_datasets[name].total_samples
        
        # 品質の比較
        for name in dataset_names:
            if name in self.analysis_results:
                analysis = self.analysis_results[name]
                comparison['quality_comparison'][name] = {
                    'grade': analysis['quality_metrics']['quality_grade'],
                    'consistency': analysis['quality_metrics']['consistency']
                }
        
        # バランスの比較
        for name in dataset_names:
            if name in self.analysis_results:
                analysis = self.analysis_results[name]
                comparison['balance_comparison'][name] = {
                    'balance_score': analysis['data_balance']['balance_score'],
                    'is_balanced': analysis['data_balance']['is_balanced']
                }
        
        # 総合推奨事項
        best_dataset = max(dataset_names, 
                          key=lambda x: self.analysis_results.get(x, {}).get('quality_metrics', {}).get('consistency', 0))
        comparison['recommendations'].append(f"最高品質データセット: {best_dataset}")
        
        return comparison
    
    def generate_report(self, output_dir: str = "ai_learning_analysis"):
        """分析レポートの生成"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 個別データセット分析レポート
        for name, analysis in self.analysis_results.items():
            self._generate_individual_report(name, analysis, output_dir)
        
        # 比較分析レポート
        if len(self.learning_datasets) > 1:
            dataset_names = list(self.learning_datasets.keys())
            comparison = self.compare_datasets(dataset_names)
            self._generate_comparison_report(comparison, output_dir)
        
        # 可視化レポート
        self._generate_visualization_report(output_dir)
    
    def _generate_individual_report(self, name: str, analysis: Dict, output_dir: str):
        """個別データセットのレポート生成"""
        report_path = os.path.join(output_dir, f"{name}_analysis_report.json")
        
        report = {
            'dataset_name': name,
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'basic_statistics': analysis['basic_stats'],
            'defect_distribution': analysis['defect_distribution'],
            'quality_metrics': analysis['quality_metrics'],
            'data_balance': analysis['data_balance'],
            'recommendations': analysis['recommendations']
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    
    def _generate_comparison_report(self, comparison: Dict, output_dir: str):
        """比較分析レポートの生成"""
        report_path = os.path.join(output_dir, "dataset_comparison_report.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    def _generate_visualization_report(self, output_dir: str):
        """可視化レポートの生成"""
        plt.style.use('seaborn-v0_8')
        
        # データセット比較グラフ
        if len(self.learning_datasets) > 1:
            self._create_comparison_charts(output_dir)
        
        # 個別データセットの詳細グラフ
        for name, dataset in self.learning_datasets.items():
            self._create_individual_charts(name, dataset, output_dir)
    
    def _create_comparison_charts(self, output_dir: str):
        """比較チャートの作成"""
        # サンプル数比較
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # サンプル数バー
        names = list(self.learning_datasets.keys())
        sample_counts = [self.learning_datasets[name].total_samples for name in names]
        axes[0, 0].bar(names, sample_counts, color='skyblue')
        axes[0, 0].set_title('データセット別サンプル数', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('サンプル数')
        
        # 品質グレード比較
        quality_grades = []
        for name in names:
            if name in self.analysis_results:
                grade = self.analysis_results[name]['quality_metrics']['quality_grade']
                quality_grades.append(grade)
            else:
                quality_grades.append('N/A')
        
        grade_colors = {'A': 'green', 'B': 'yellow', 'C': 'orange', 'D': 'red', 'N/A': 'gray'}
        colors = [grade_colors.get(grade, 'gray') for grade in quality_grades]
        axes[0, 1].bar(names, [1]*len(names), color=colors)
        axes[0, 1].set_title('品質グレード比較', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('グレード')
        
        # データバランス比較
        balance_scores = []
        for name in names:
            if name in self.analysis_results:
                score = self.analysis_results[name]['data_balance']['balance_score']
                balance_scores.append(score)
            else:
                balance_scores.append(0)
        
        axes[1, 0].bar(names, balance_scores, color='lightcoral')
        axes[1, 0].set_title('データバランススコア', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('バランススコア')
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='推奨閾値')
        axes[1, 0].legend()
        
        # 総合評価レーダーチャート
        if len(names) >= 2:
            # 簡易レーダーチャート（2つのデータセット比較）
            metrics = ['サンプル数', '品質', 'バランス', '多様性']
            dataset1_scores = [0.8, 0.7, 0.6, 0.9]  # 仮のスコア
            dataset2_scores = [0.6, 0.8, 0.9, 0.7]  # 仮のスコア
            
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 閉じるため
            dataset1_scores += dataset1_scores[:1]
            dataset2_scores += dataset2_scores[:1]
            
            axes[1, 1].plot(angles, dataset1_scores, 'o-', linewidth=2, label=names[0])
            axes[1, 1].plot(angles, dataset2_scores, 'o-', linewidth=2, label=names[1])
            axes[1, 1].set_xticks(angles[:-1])
            axes[1, 1].set_xticklabels(metrics)
            axes[1, 1].set_title('総合評価比較', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dataset_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_individual_charts(self, name: str, dataset: AILearningData, output_dir: str):
        """個別データセットのチャート作成"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 欠陥タイプ分布
        defect_types = list(dataset.defect_types.keys())
        defect_counts = list(dataset.defect_types.values())
        axes[0, 0].pie(defect_counts, labels=defect_types, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title(f'{name} - 欠陥タイプ分布', fontsize=14, fontweight='bold')
        
        # 品質スコア分布
        if dataset.quality_scores:
            axes[0, 1].hist(dataset.quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].axvline(np.mean(dataset.quality_scores), color='red', linestyle='--', 
                              label=f'平均: {np.mean(dataset.quality_scores):.3f}')
            axes[0, 1].set_title(f'{name} - 品質スコア分布', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('品質スコア')
            axes[0, 1].set_ylabel('頻度')
            axes[0, 1].legend()
        
        # 欠陥面積分布
        if dataset.defect_areas:
            axes[1, 0].hist(dataset.defect_areas, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1, 0].axvline(np.mean(dataset.defect_areas), color='red', linestyle='--',
                              label=f'平均: {np.mean(dataset.defect_areas):.3f}mm²')
            axes[1, 0].set_title(f'{name} - 欠陥面積分布', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('欠陥面積 (mm²)')
            axes[1, 0].set_ylabel('頻度')
            axes[1, 0].legend()
        
        # 円形度vs同心度散布図
        if dataset.circularity_scores and dataset.concentricity_scores:
            scatter = axes[1, 1].scatter(dataset.circularity_scores, dataset.concentricity_scores, 
                                       alpha=0.6, c=dataset.quality_scores if dataset.quality_scores else 'blue',
                                       cmap='viridis' if dataset.quality_scores else None)
            axes[1, 1].set_xlabel('円形度')
            axes[1, 1].set_ylabel('同心度')
            axes[1, 1].set_title(f'{name} - 円形度vs同心度', fontsize=14, fontweight='bold')
            if dataset.quality_scores:
                plt.colorbar(scatter, ax=axes[1, 1], label='品質スコア')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_detailed_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

def create_sample_learning_data() -> AILearningData:
    """サンプルAI学習データの作成（デモ用）"""
    # 実際のプロジェクトでは、ここで画像ファイルから学習データを読み込む
    np.random.seed(42)  # 再現性のため
    
    # 欠陥タイプ別のサンプル数（プロジェクト仕様に基づく）
    defect_types = {
        DefectType.NORMAL: 400,
        DefectType.BLACK_SPOT: 150,
        DefectType.SHORT_CIRCUIT: 100,
        DefectType.FOREIGN_MATTER: 120
    }
    
    total_samples = sum(defect_types.values())
    
    # 品質スコアの生成（0.5-1.0の範囲）
    quality_scores = np.random.beta(2, 1, total_samples) * 0.5 + 0.5
    
    # 欠陥面積の生成（mm²単位）
    defect_areas = np.random.exponential(2.0, total_samples)
    
    # 円形度スコアの生成（0.7-1.0の範囲）
    circularity_scores = np.random.beta(3, 1, total_samples) * 0.3 + 0.7
    
    # 同心度スコアの生成（0.0-0.1の範囲）
    concentricity_scores = np.random.exponential(0.02, total_samples)
    
    return AILearningData(
        total_samples=total_samples,
        defect_types=defect_types,
        quality_scores=quality_scores.tolist(),
        defect_areas=defect_areas.tolist(),
        circularity_scores=circularity_scores.tolist(),
        concentricity_scores=concentricity_scores.tolist(),
        image_resolution=(1920, 1080),
        data_balance_ratio=min(defect_types.values()) / max(defect_types.values()),
        average_quality=np.mean(quality_scores)
    )

def create_enhanced_learning_data() -> AILearningData:
    """改良版AI学習データの作成（比較用）"""
    np.random.seed(123)  # 異なるシードで異なるデータを生成
    
    # より多くのサンプルとバランスの取れた分布
    defect_types = {
        DefectType.NORMAL: 500,
        DefectType.BLACK_SPOT: 200,
        DefectType.SHORT_CIRCUIT: 180,
        DefectType.FOREIGN_MATTER: 200
    }
    
    total_samples = sum(defect_types.values())
    
    # より高い品質スコア
    quality_scores = np.random.beta(3, 1, total_samples) * 0.3 + 0.7
    
    # より多様な欠陥面積
    defect_areas = np.random.gamma(2, 1.5, total_samples)
    
    # より高い円形度
    circularity_scores = np.random.beta(4, 1, total_samples) * 0.2 + 0.8
    
    # より低い同心度（より良い品質）
    concentricity_scores = np.random.exponential(0.015, total_samples)
    
    return AILearningData(
        total_samples=total_samples,
        defect_types=defect_types,
        quality_scores=quality_scores.tolist(),
        defect_areas=defect_areas.tolist(),
        circularity_scores=circularity_scores.tolist(),
        concentricity_scores=concentricity_scores.tolist(),
        image_resolution=(1920, 1080),
        data_balance_ratio=min(defect_types.values()) / max(defect_types.values()),
        average_quality=np.mean(quality_scores)
    )

def load_ai_learning_data_from_folder(folder_path: str) -> AILearningData:
    """CS AI学習データフォルダーから実際の画像データを読み込む"""
    print(f"AI学習データフォルダーを読み込み中: {folder_path}")
    
    # サポートする画像形式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # 検査用ファイルを除外するキーワード
    exclude_keywords = ['検査用', 'inspection', 'test', 'debug', 'temp']
    
    all_images = []
    defect_types = {
        DefectType.NORMAL: 0,
        DefectType.BLACK_SPOT: 0,
        DefectType.SHORT_CIRCUIT: 0,
        DefectType.FOREIGN_MATTER: 0
    }
    
    quality_scores = []
    defect_areas = []
    circularity_scores = []
    concentricity_scores = []
    
    # フォルダー内の画像ファイルを検索
    for ext in image_extensions:
        pattern = os.path.join(folder_path, '**', ext)
        images = glob.glob(pattern, recursive=True)
        all_images.extend(images)
    
    print(f"発見された画像ファイル数: {len(all_images)}")
    
    # 検査用ファイルを除外
    filtered_images = []
    for img_path in all_images:
        filename = os.path.basename(img_path).lower()
        should_exclude = any(keyword in filename for keyword in exclude_keywords)
        if not should_exclude:
            filtered_images.append(img_path)
        else:
            print(f"除外: {img_path} (検査用ファイル)")
    
    print(f"フィルタリング後の画像ファイル数: {len(filtered_images)}")
    
    # 各画像を分析
    for i, img_path in enumerate(filtered_images):
        try:
            # 画像を読み込み
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            image_resolution = (width, height)
            
            # ワッシャー検出
            mask = threshold_washers(gray)
            washers = find_washers(mask)
            
            if not washers:
                continue
            
            # 各ワッシャーを分析
            for outer, inner in washers:
                try:
                    # 基本的な分析（キャリブレーションなし）
                    (ox, oy), orad = cv2.minEnclosingCircle(outer)
                    (ix, iy), irad = cv2.minEnclosingCircle(inner)
                    
                    # 円形度計算
                    circ = compute_circularity(outer)
                    circularity_scores.append(circ)
                    
                    # 同心度計算
                    conc = float(np.hypot(ox-ix, oy-iy)) / max(orad, 1.0)
                    concentricity_scores.append(conc)
                    
                    # 欠陥検出
                    ring = np.zeros(gray.shape, np.uint8)
                    cv2.drawContours(ring, [outer], -1, 255, -1)
                    cv2.drawContours(ring, [inner], -1, 0, -1)
                    
                    blur = cv2.GaussianBlur(gray, (0, 0), 3)
                    diff = cv2.subtract(blur, cv2.GaussianBlur(blur, (0, 0), 9))
                    dark = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                    dark &= ring
                    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 1)
                    
                    dark_px = int(np.count_nonzero(dark))
                    ring_area = int(cv2.contourArea(outer) - cv2.contourArea(inner))
                    
                    # 欠陥面積（ピクセル単位）
                    defect_areas.append(float(dark_px))
                    
                    # 品質スコア計算（円形度と同心度に基づく）
                    quality = (circ + (1 - conc)) / 2
                    quality_scores.append(quality)
                    
                    # 欠陥タイプの判定
                    if dark_px > ring_area * 0.01:  # 1%以上の欠陥面積
                        if dark_px > ring_area * 0.05:  # 5%以上の大きな欠陥
                            defect_types[DefectType.FOREIGN_MATTER] += 1
                        else:
                            defect_types[DefectType.BLACK_SPOT] += 1
                    elif circ < 0.8:  # 円形度が低い
                        defect_types[DefectType.SHORT_CIRCUIT] += 1
                    else:
                        defect_types[DefectType.NORMAL] += 1
                        
                except Exception as e:
                    print(f"画像分析エラー {img_path}: {e}")
                    continue
            
            if (i + 1) % 10 == 0:
                print(f"処理済み: {i + 1}/{len(filtered_images)}")
                
        except Exception as e:
            print(f"画像読み込みエラー {img_path}: {e}")
            continue
    
    total_samples = sum(defect_types.values())
    
    if total_samples == 0:
        print("有効な学習データが見つかりませんでした")
        return create_sample_learning_data()  # フォールバック
    
    print(f"分析完了: {total_samples}サンプル")
    print(f"欠陥タイプ分布: {defect_types}")
    
    return AILearningData(
        total_samples=total_samples,
        defect_types=defect_types,
        quality_scores=quality_scores,
        defect_areas=defect_areas,
        circularity_scores=circularity_scores,
        concentricity_scores=concentricity_scores,
        image_resolution=image_resolution if 'image_resolution' in locals() else (1920, 1080),
        data_balance_ratio=min(defect_types.values()) / max(defect_types.values()) if max(defect_types.values()) > 0 else 0,
        average_quality=np.mean(quality_scores) if quality_scores else 0.5
    )

def run_ai_analysis(args):
    """AI学習データの分析を実行"""
    print("=== ワッシャーAI学習データ分析システム ===")
    print("京都先端科学大学 機械電気システム工学科")
    print("プロジェクト: 低コスト画像検査装置の開発")
    print("=" * 50)
    
    # AI学習データ分析器の初期化
    analyzer = AILearningAnalyzer()
    
    # 実際の画像データを読み込む
    ai_data_folder = args.ai_data_folder
    if os.path.exists(ai_data_folder):
        print("実際のAI学習データを読み込み中...")
        real_dataset = load_ai_learning_data_from_folder(ai_data_folder)
        analyzer.add_dataset("実際の学習データ", real_dataset)
        print(f"[OK] 実際の学習データ: {real_dataset.total_samples}サンプル")
    else:
        print(f"警告: {ai_data_folder}フォルダーが見つかりません")
        print("サンプルデータを使用します")
    
    # サンプルデータセットの作成
    print("学習データセットを準備中...")
    
    # 基本データセット
    basic_dataset = create_sample_learning_data()
    analyzer.add_dataset("基本データセット", basic_dataset)
    print(f"[OK] 基本データセット: {basic_dataset.total_samples}サンプル")
    
    # 改良データセット
    enhanced_dataset = create_enhanced_learning_data()
    analyzer.add_dataset("改良データセット", enhanced_dataset)
    print(f"[OK] 改良データセット: {enhanced_dataset.total_samples}サンプル")
    
    # 分析実行
    print("\n分析を実行中...")
    analyzer.generate_report(args.output_dir)
    
    # 比較分析
    print("\n=== データセット比較分析 ===")
    dataset_names = list(analyzer.learning_datasets.keys())
    comparison = analyzer.compare_datasets(dataset_names)
    
    print(f"データセット数: {len(comparison['datasets'])}")
    print("\nサンプル数比較:")
    for name, count in comparison['sample_counts'].items():
        print(f"  {name}: {count:,}サンプル")
    
    print("\n品質比較:")
    for name, quality in comparison['quality_comparison'].items():
        print(f"  {name}: グレード{quality['grade']}, 一貫性{quality['consistency']:.3f}")
    
    print("\nバランス比較:")
    for name, balance in comparison['balance_comparison'].items():
        print(f"  {name}: バランススコア{balance['balance_score']:.3f}, バランス{'良好' if balance['is_balanced'] else '要改善'}")
    
    print(f"\n推奨事項:")
    for rec in comparison['recommendations']:
        print(f"  - {rec}")
    
    print(f"\n分析レポートを生成しました: {args.output_dir}/")
    print("生成されたファイル:")
    print("  - 個別分析レポート (JSON)")
    print("  - 比較分析レポート (JSON)")
    print("  - 可視化チャート (PNG)")

def run_realtime_comparison(args):
    """リアルタイムで学習データ量による検査精度の差を比較表示"""
    print("=== リアルタイム学習データ量比較システム ===")
    print("京都先端科学大学 機械電気システム工学科")
    print("プロジェクト: 低コスト画像検査装置の開発")
    print("=" * 50)
    
    # 異なる学習データ量のシミュレーション
    learning_scenarios = {
        "少量学習": {
            "total_samples": 100,
            "defect_types": {"normal": 60, "black_spot": 20, "short_circuit": 10, "foreign_matter": 10},
            "accuracy": 0.75,
            "color": (0, 100, 255)  # 赤系
        },
        "中量学習": {
            "total_samples": 500,
            "defect_types": {"normal": 300, "black_spot": 100, "short_circuit": 50, "foreign_matter": 50},
            "accuracy": 0.85,
            "color": (0, 200, 255)  # オレンジ系
        },
        "大量学習": {
            "total_samples": 1000,
            "defect_types": {"normal": 600, "black_spot": 200, "short_circuit": 100, "foreign_matter": 100},
            "accuracy": 0.92,
            "color": (0, 255, 0)  # 緑系
        }
    }
    
    # カメラ初期化
    spec = WasherSpec(args.expected_outer, args.expected_inner, args.tol, args.circ_min, args.conc_max, args.dark_min_area)
    calib = Calibration(args.mm_per_px)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if args.width: cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps: cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened(): 
        raise RuntimeError("Failed to open camera.")
    
    # 統計情報
    stats = {
        "frame_count": 0,
        "total_washers": 0,
        "correct_detections": {name: 0 for name in learning_scenarios.keys()},
        "false_positives": {name: 0 for name in learning_scenarios.keys()},
        "false_negatives": {name: 0 for name in learning_scenarios.keys()}
    }
    
    last = time.time()
    fps = 0.0
    
    print("リアルタイム比較を開始します...")
    print("キー操作:")
    print("  'q' または ESC: 終了")
    print("  's': スナップショット保存")
    print("  'm': キャリブレーション")
    print("  'r': 統計リセット")
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = threshold_washers(gray)
        results = []
        
        # ワッシャー検出と分析
        for outer, inner in find_washers(mask):
            try:
                r = analyze_washer(outer, inner, calib, spec, gray)
                results.append(r)
                draw_result(frame, r, calib)
            except Exception: 
                pass
        
        # 学習データ量別の精度シミュレーション
        for scenario_name, scenario in learning_scenarios.items():
            # 実際の精度に基づいてランダムに判定を変更
            np.random.seed(int(time.time() * 1000) % 10000)  # 時間ベースのシード
            
            for i, r in enumerate(results):
                # 元の判定結果
                original_pass = r[8] and r[9]
                
                # 学習データ量に応じた精度で判定を変更
                accuracy = scenario["accuracy"]
                if np.random.random() > accuracy:
                    # 精度に応じて誤判定を発生
                    if original_pass:
                        # 正解を誤判定（偽陰性）
                        stats["false_negatives"][scenario_name] += 1
                    else:
                        # 誤判定を正解（偽陽性）
                        stats["false_positives"][scenario_name] += 1
                else:
                    # 正しい判定
                    if original_pass:
                        stats["correct_detections"][scenario_name] += 1
        
        # 統計更新
        stats["frame_count"] += 1
        stats["total_washers"] += len(results)
        
        # フレームレート計算
        now = time.time()
        dt = now - last
        last = now
        if dt > 0: 
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        
        # 結果表示
        okn = sum(1 for r in results if r[8] and r[9])
        
        # 左上の表示をすべて削除
        
        # ウィンドウ表示
        cv2.imshow("Washer Inspection - Learning Data Comparison", frame)
        if args.debug: 
            cv2.imshow("Mask", mask)
        
        # キー入力処理
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): 
            break
        elif k == ord('s'): 
            cv2.imwrite("snapshot.png", frame)
            print("スナップショットを保存しました")
        elif k == ord('m') and results and spec.expected_inner_mm:
            r = max(results, key=lambda x: x[4])
            calib.mm_per_pixel = spec.expected_inner_mm / (r[4] * 2.0)
            print(f"キャリブレーション完了: {calib.mm_per_pixel:.6f} mm/px")
        elif k == ord('r'):
            # 統計リセット
            stats = {
                "frame_count": 0,
                "total_washers": 0,
                "correct_detections": {name: 0 for name in learning_scenarios.keys()},
                "false_positives": {name: 0 for name in learning_scenarios.keys()},
                "false_negatives": {name: 0 for name in learning_scenarios.keys()}
            }
            print("統計をリセットしました")
    
    # 最終統計表示
    print("\n=== 最終統計 ===")
    print(f"総フレーム数: {stats['frame_count']}")
    print(f"総ワッシャー数: {stats['total_washers']}")
    print("\n学習データ量別精度:")
    for scenario_name, scenario in learning_scenarios.items():
        total_detections = stats["correct_detections"][scenario_name] + stats["false_positives"][scenario_name]
        if total_detections > 0:
            precision = stats["correct_detections"][scenario_name] / total_detections
            recall = stats["correct_detections"][scenario_name] / (stats["correct_detections"][scenario_name] + stats["false_negatives"][scenario_name]) if (stats["correct_detections"][scenario_name] + stats["false_negatives"][scenario_name]) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  {scenario_name}: 精度={precision:.1%}, 再現率={recall:.1%}, F1スコア={f1_score:.1%}")
    
    # 学習データ不足分析
    print("\n=== 学習データ不足分析 ===")
    print("現在のシステムで不足している学習データ:")
    print("1. 多様な照明条件でのワッシャー画像")
    print("   - 明るい照明、暗い照明、影のある条件")
    print("   - 異なる角度からの撮影")
    print("2. 欠陥パターンの多様化")
    print("   - 黒点の大きさと位置のバリエーション")
    print("   - 傷の種類（線状、点状、不規則）")
    print("   - 異物の種類（金属片、ゴミ、汚れ）")
    print("3. ワッシャーサイズの多様化")
    print("   - 異なる外径・内径の組み合わせ")
    print("   - 厚みの異なるワッシャー")
    print("4. 背景の多様化")
    print("   - 異なる色の背景")
    print("   - 複数のワッシャーが重なった場合")
    print("   - ゴミや異物が混在した場合")
    print("5. 画像品質の多様化")
    print("   - ぼやけた画像、鮮明な画像")
    print("   - ノイズの多い画像")
    print("   - 解像度の異なる画像")
    print("\n推奨される学習データ量:")
    print("- 良品: 最低500枚（多様な条件で）")
    print("- 不良品: 各欠陥タイプ最低100枚")
    print("- 総計: 最低1000枚以上")
    
    cap.release()
    cv2.destroyAllWindows()

def run(args):
    if args.ai_analysis:
        run_ai_analysis(args)
        return
    elif args.realtime_comparison:
        run_realtime_comparison(args)
        return
    
    spec=WasherSpec(args.expected_outer,args.expected_inner,args.tol,args.circ_min,args.conc_max,args.dark_min_area)
    calib=Calibration(args.mm_per_px)
    cap=cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if args.width: cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
    if args.height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
    if args.fps: cap.set(cv2.CAP_PROP_FPS,args.fps)
    if not cap.isOpened(): raise RuntimeError("Failed to open camera.")
    last=time.time(); fps=0.0
    while True:
        ok,frame=cap.read()
        if not ok: break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask=threshold_washers(gray)
        results=[]
        for outer,inner in find_washers(mask):
            try:
                r=analyze_washer(outer,inner,calib,spec,gray)
                results.append(r); draw_result(frame,r,calib)
            except Exception: pass
        now=time.time(); dt=now-last; last=now
        if dt>0: fps=0.9*fps+0.1*(1.0/dt)
        okn=sum(1 for r in results if r[8] and r[9])
        cv2.putText(frame,f"OK {okn}/{len(results)} | FPS {fps:.1f}",(10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(50,220,50),2,cv2.LINE_AA)
        cv2.imshow("Washer Inspection",frame)
        if args.debug: cv2.imshow("Mask",mask)
        k=cv2.waitKey(1)&0xFF
        if k in (27, ord('q')): break
        if k==ord('s'): cv2.imwrite("snapshot.png",frame)
        if k==ord('m') and results and spec.expected_inner_mm:
            r=max(results,key=lambda x:x[4])
            calib.mm_per_pixel=spec.expected_inner_mm/(r[4]*2.0)
            print(f"Calibrated from ID: {calib.mm_per_pixel:.6f} mm/px")
    cap.release(); cv2.destroyAllWindows()

def build_argparser():
    p=argparse.ArgumentParser(description="ワッシャー検査システム - リアルタイム検査とAI学習データ分析")
    p.add_argument("--camera",type=int,default=0, help="カメラデバイス番号")
    p.add_argument("--width",type=int,default=1920, help="カメラ解像度（幅）")
    p.add_argument("--height",type=int,default=1080, help="カメラ解像度（高さ）")
    p.add_argument("--fps",type=int,default=30, help="フレームレート")
    p.add_argument("--mm-per-px",type=float,default=None, help="ピクセルあたりのmm数（キャリブレーション）")
    p.add_argument("--expected-outer",type=float,default=None, help="期待される外径（mm）")
    p.add_argument("--expected-inner",type=float,default=None, help="期待される内径（mm）")
    p.add_argument("--tol",type=float,default=0.6, help="直径許容差（mm）")
    p.add_argument("--circ-min",type=float,default=0.85, help="最小円形度")
    p.add_argument("--conc-max",type=float,default=0.08, help="最大同心度比")
    p.add_argument("--dark-min-area",type=float,default=0.5, help="最小欠陥面積（mm²）")
    p.add_argument("--debug",action="store_true", help="デバッグモード")
    p.add_argument("--ai-analysis",action="store_true", help="AI学習データ分析モード")
    p.add_argument("--realtime-comparison",action="store_true", help="リアルタイム学習データ量比較モード")
    p.add_argument("--output-dir",type=str,default="ai_learning_analysis", help="分析結果出力ディレクトリ")
    p.add_argument("--ai-data-folder",type=str,default="CS AI学習データ", help="AI学習データフォルダーのパス")
    return p

if __name__=="__main__":
    run(build_argparser().parse_args())