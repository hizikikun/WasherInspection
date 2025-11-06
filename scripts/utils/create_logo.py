#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WasherInspectionアプリ用ロゴ生成スクリプト
"""

import sys
import os
from pathlib import Path

# UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("警告: PIL（Pillow）がインストールされていません。'pip install Pillow'を実行してください。")
    sys.exit(1)

def create_logo_icon(size=(256, 256), output_path=None):
    """アプリ用アイコンを作成（よりわかりやすいシンプルデザイン）"""
    import math
    
    # 背景色（黒と銀のグラデーション）
    # 上から下へ：黒 → ダークグレー → シルバーグレー
    img = Image.new('RGB', size, color=(30, 30, 35))
    draw = ImageDraw.Draw(img)
    
    # 背景グラデーション（黒から銀へ）
    for y in range(size[1]):
        ratio = y / size[1]
        # 黒(30,30,35) → ダークグレー(60,60,65) → シルバーグレー(180,180,185)
        r = int(30 + (180 - 30) * ratio * 0.8)
        g = int(30 + (180 - 30) * ratio * 0.8)
        b = int(35 + (185 - 35) * ratio * 0.8)
        draw.line([(0, y), (size[0], y)], fill=(r, g, b))
    
    bg_color = (60, 60, 65)  # 中間のグレー（ワッシャーの穴用）
    
    # 中央の座標
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # ワッシャーの半径（先に定義、検査ビームの計算に必要）
    washer_radius = int(size[0] * 0.42)  # より大きく
    
    # === 検査カメラアイコン（斜めから見た立体的なカメラ） ===
    if size[0] >= 64:
        import math
        
        # カメラの位置（上部中央、少し下に配置）
        camera_y = center_y - int(size[0] * 0.28)  # より下に移動
        camera_base_width = max(30, size[0] // 5.2)  # 本体を少し大きく
        camera_base_height = max(22, size[0] // 7.5)  # 本体の高さも調整
        
        # 斜めから見た角度のオフセット（右方向に傾き）
        perspective_offset = int(camera_base_width * 0.25)
        
        # カメラ本体の色
        camera_body_color = (50, 50, 55)
        camera_body_dark = (35, 35, 40)
        camera_outline_color = (25, 25, 30)
        
        # カメラの前面（手前側、右側）
        front_x = center_x + perspective_offset // 2
        front_width = int(camera_base_width * 0.9)
        front_left = front_x - front_width // 2
        front_right = front_x + front_width // 2
        front_top = camera_y - camera_base_height
        front_bottom = camera_y
        
        # カメラの側面（左側、奥側、より暗く）
        side_x = center_x - perspective_offset
        side_width = int(camera_base_width * 0.7)
        side_left = side_x - side_width // 2
        side_right = side_x + side_width // 2
        side_y_top = camera_y - camera_base_height
        side_y_bottom = camera_y - int(camera_base_height * 0.3)
        
        # 前面を描画（側面との接続部分を除く）
        draw.rectangle(
            [front_left, front_top, front_right, front_bottom],
            fill=camera_body_color, outline=None
        )
        # 前面の縁取り
        draw.line([(front_left, front_top), (front_right, front_top)], 
                 fill=camera_outline_color, width=2)
        draw.line([(front_right, front_top), (front_right, front_bottom)], 
                 fill=camera_outline_color, width=2)
        draw.line([(front_right, front_bottom), (front_left, front_bottom)], 
                 fill=camera_outline_color, width=2)
        
        # 側面の上部台形（上面との接続部分）
        side_top_points = [
            (side_left, side_y_top),                          # 左上
            (side_right, side_y_top),                         # 右上
            (front_left, side_y_bottom),                      # 右下（前面と接続）
            (side_left, side_y_bottom)                        # 左下
        ]
        draw.polygon(side_top_points, fill=camera_body_dark, outline=None)
        
        # 側面の下部（前面との接続部分）
        side_bottom_points = [
            (side_left, side_y_bottom),                       # 左上
            (front_left, side_y_bottom),                      # 右上
            (front_left, front_bottom),                       # 右下
            (side_left, front_bottom)                         # 左下
        ]
        draw.polygon(side_bottom_points, fill=camera_body_dark, outline=None)
        
        # 側面の縁取り
        draw.line([(side_left, side_y_top), (side_right, side_y_top)], 
                 fill=camera_outline_color, width=1)
        draw.line([(side_left, side_y_top), (side_left, front_bottom)], 
                 fill=camera_outline_color, width=1)
        draw.line([(side_left, front_bottom), (front_left, front_bottom)], 
                 fill=camera_outline_color, width=1)
        
        # カメラ上面（台形）
        top_points = [
            (side_left, side_y_top),                          # 左上
            (side_right, side_y_top),                         # 右上
            (front_right, front_top),                         # 右下
            (front_left, front_top)                           # 左下
        ]
        top_color = (60, 60, 65)  # 少し明るく（上面の光）
        draw.polygon(top_points, fill=top_color, outline=None)
        # 上面の縁取り
        draw.line([(side_left, side_y_top), (front_left, front_top)], 
                 fill=camera_outline_color, width=1)
        draw.line([(side_right, side_y_top), (front_right, front_top)], 
                 fill=camera_outline_color, width=1)
        
        # ビューファインダー（側面から見える部分）
        viewfinder_y = side_y_top + int(camera_base_height * 0.2)
        viewfinder_size = max(6, size[0] // 35)
        draw.ellipse(
            [side_x - viewfinder_size, viewfinder_y - viewfinder_size,
             side_x + viewfinder_size, viewfinder_y + viewfinder_size],
            fill=(20, 20, 25), outline=(10, 10, 15), width=1
        )
        
        # カメラレンズ（前面に配置、斜めから見た楕円、比率を調整）
        lens_center_x = front_x
        lens_center_y = camera_y - int(camera_base_height * 0.55)
        lens_width = max(10, size[0] // 15)  # レンズを少し小さく（本体との比率を改善）
        lens_height = int(lens_width * 0.85)  # 斜めから見たため少しつぶれた楕円
        
        # レンズの外側リング（金属質感）
        lens_ring_width = lens_width + 3
        lens_ring_height = lens_height + 2
        draw.ellipse(
            [lens_center_x - lens_ring_width, lens_center_y - lens_ring_height,
             lens_center_x + lens_ring_width, lens_center_y + lens_ring_height],
            fill=(110, 110, 120), outline=(80, 80, 90), width=1
        )
        
        # レンズ本体（鮮明な青、検査を表現）
        draw.ellipse(
            [lens_center_x - lens_width, lens_center_y - lens_height,
             lens_center_x + lens_width, lens_center_y + lens_height],
            fill=(40, 120, 180), outline=(20, 80, 140), width=2
        )
        
        # レンズの内側（絞りを表現）
        inner_lens_width = int(lens_width * 0.7)
        inner_lens_height = int(lens_height * 0.7)
        draw.ellipse(
            [lens_center_x - inner_lens_width, lens_center_y - inner_lens_height,
             lens_center_x + inner_lens_width, lens_center_y + inner_lens_height],
            fill=(25, 90, 150), outline=(15, 60, 110), width=1
        )
        
        # レンズのハイライト（光の反射）
        highlight_size = max(4, size[0] // 45)
        highlight_offset_x = int(lens_width * 0.3)
        highlight_offset_y = -int(lens_height * 0.25)
        draw.ellipse(
            [lens_center_x + highlight_offset_x - highlight_size // 2,
             lens_center_y + highlight_offset_y - highlight_size // 2,
             lens_center_x + highlight_offset_x + highlight_size // 2,
             lens_center_y + highlight_offset_y + highlight_size // 2],
            fill=(255, 255, 255), outline=None
        )
        
        # シャッターボタン（上面、右側）
        shutter_x = front_x + int(front_width * 0.3)
        shutter_y = camera_y - camera_base_height - int(camera_base_height * 0.15)
        shutter_size = max(3, size[0] // 55)
        draw.ellipse(
            [shutter_x - shutter_size, shutter_y - shutter_size,
             shutter_x + shutter_size, shutter_y + shutter_size],
            fill=(70, 70, 75), outline=(50, 50, 55), width=1
        )
        
        # カメラからワッシャーへの検査ビーム（太く明確に）
        line_start_x = lens_center_x
        line_start_y = camera_y
        line_end_x = center_x
        line_end_y = center_y - washer_radius
        line_width = max(4, size[0] // 50)
        draw.line(
            [(line_start_x, line_start_y), (line_end_x, line_end_y)],
            fill='white', width=line_width
        )
    
    # === メイン要素：ワッシャー（カメラの後、ビームの後に描画） ===
    # ワッシャーの外側（鮮明な白、縁取りなし、検査ビームの上に重ねる）
    draw.ellipse(
        [center_x - washer_radius, center_y - washer_radius,
         center_x + washer_radius, center_y + washer_radius],
        fill='white', outline=None
    )
    
    # ワッシャーの内側（穴を大きく明確に、縁取りなし）
    washer_inner_radius = int(washer_radius * 0.5)
    inner_hole_color = (40, 40, 45)  # 暗いグレー
    draw.ellipse(
        [center_x - washer_inner_radius, center_y - washer_inner_radius,
         center_x + washer_inner_radius, center_y + washer_inner_radius],
        fill=inner_hole_color, outline=None
    )
    
    # === カメラを再描画（ワッシャーの上に表示） ===
    if size[0] >= 64:
        import math
        
        # カメラの位置（上部中央、少し下に配置）
        camera_y = center_y - int(size[0] * 0.28)
        camera_base_width = max(30, size[0] // 5.2)
        camera_base_height = max(22, size[0] // 7.5)
        perspective_offset = int(camera_base_width * 0.25)
        
        # カメラ本体の色
        camera_body_color = (50, 50, 55)
        camera_body_dark = (35, 35, 40)
        camera_outline_color = (25, 25, 30)
        
        # カメラの前面
        front_x = center_x + perspective_offset // 2
        front_width = int(camera_base_width * 0.9)
        front_left = front_x - front_width // 2
        front_right = front_x + front_width // 2
        front_top = camera_y - camera_base_height
        front_bottom = camera_y
        
        # カメラの側面
        side_x = center_x - perspective_offset
        side_width = int(camera_base_width * 0.7)
        side_left = side_x - side_width // 2
        side_right = side_x + side_width // 2
        side_y_top = camera_y - camera_base_height
        side_y_bottom = camera_y - int(camera_base_height * 0.3)
        
        # 前面を描画
        draw.rectangle(
            [front_left, front_top, front_right, front_bottom],
            fill=camera_body_color, outline=None
        )
        draw.line([(front_left, front_top), (front_right, front_top)], 
                 fill=camera_outline_color, width=2)
        draw.line([(front_right, front_top), (front_right, front_bottom)], 
                 fill=camera_outline_color, width=2)
        draw.line([(front_right, front_bottom), (front_left, front_bottom)], 
                 fill=camera_outline_color, width=2)
        
        # 側面の上部台形
        side_top_points = [
            (side_left, side_y_top),
            (side_right, side_y_top),
            (front_left, side_y_bottom),
            (side_left, side_y_bottom)
        ]
        draw.polygon(side_top_points, fill=camera_body_dark, outline=None)
        
        # 側面の下部
        side_bottom_points = [
            (side_left, side_y_bottom),
            (front_left, side_y_bottom),
            (front_left, front_bottom),
            (side_left, front_bottom)
        ]
        draw.polygon(side_bottom_points, fill=camera_body_dark, outline=None)
        
        # 側面の縁取り
        draw.line([(side_left, side_y_top), (side_right, side_y_top)], 
                 fill=camera_outline_color, width=1)
        draw.line([(side_left, side_y_top), (side_left, front_bottom)], 
                 fill=camera_outline_color, width=1)
        draw.line([(side_left, front_bottom), (front_left, front_bottom)], 
                 fill=camera_outline_color, width=1)
        
        # カメラ上面
        top_points = [
            (side_left, side_y_top),
            (side_right, side_y_top),
            (front_right, front_top),
            (front_left, front_top)
        ]
        top_color = (60, 60, 65)
        draw.polygon(top_points, fill=top_color, outline=None)
        draw.line([(side_left, side_y_top), (front_left, front_top)], 
                 fill=camera_outline_color, width=1)
        draw.line([(side_right, side_y_top), (front_right, front_top)], 
                 fill=camera_outline_color, width=1)
        
        # ビューファインダー
        viewfinder_y = side_y_top + int(camera_base_height * 0.2)
        viewfinder_size = max(6, size[0] // 35)
        draw.ellipse(
            [side_x - viewfinder_size, viewfinder_y - viewfinder_size,
             side_x + viewfinder_size, viewfinder_y + viewfinder_size],
            fill=(20, 20, 25), outline=(10, 10, 15), width=1
        )
        
        # カメラレンズ
        lens_center_x = front_x
        lens_center_y = camera_y - int(camera_base_height * 0.55)
        lens_width = max(10, size[0] // 15)
        lens_height = int(lens_width * 0.85)
        
        # レンズの外側リング
        lens_ring_width = lens_width + 3
        lens_ring_height = lens_height + 2
        draw.ellipse(
            [lens_center_x - lens_ring_width, lens_center_y - lens_ring_height,
             lens_center_x + lens_ring_width, lens_center_y + lens_ring_height],
            fill=(110, 110, 120), outline=(80, 80, 90), width=1
        )
        
        # レンズ本体
        draw.ellipse(
            [lens_center_x - lens_width, lens_center_y - lens_height,
             lens_center_x + lens_width, lens_center_y + lens_height],
            fill=(40, 120, 180), outline=(20, 80, 140), width=2
        )
        
        # レンズの内側
        inner_lens_width = int(lens_width * 0.7)
        inner_lens_height = int(lens_height * 0.7)
        draw.ellipse(
            [lens_center_x - inner_lens_width, lens_center_y - inner_lens_height,
             lens_center_x + inner_lens_width, lens_center_y + inner_lens_height],
            fill=(25, 90, 150), outline=(15, 60, 110), width=1
        )
        
        # レンズのハイライト
        highlight_size = max(4, size[0] // 45)
        highlight_offset_x = int(lens_width * 0.3)
        highlight_offset_y = -int(lens_height * 0.25)
        draw.ellipse(
            [lens_center_x + highlight_offset_x - highlight_size // 2,
             lens_center_y + highlight_offset_y - highlight_size // 2,
             lens_center_x + highlight_offset_x + highlight_size // 2,
             lens_center_y + highlight_offset_y + highlight_size // 2],
            fill=(255, 255, 255), outline=None
        )
        
        # シャッターボタン
        shutter_x = front_x + int(front_width * 0.3)
        shutter_y = camera_y - camera_base_height - int(camera_base_height * 0.15)
        shutter_size = max(3, size[0] // 55)
        draw.ellipse(
            [shutter_x - shutter_size, shutter_y - shutter_size,
             shutter_x + shutter_size, shutter_y + shutter_size],
            fill=(70, 70, 75), outline=(50, 50, 55), width=1
        )
    
    # === チェックマーク（大きく、明確に） ===
    if size[0] >= 64:
        # ワッシャーの右上に大きなチェックマーク
        check_size = max(12, size[0] // 12)
        check_x = center_x + int(washer_radius * 0.55)
        check_y = center_y - int(washer_radius * 0.55)
        
        # チェックマーク（太く明確に）
        check_thickness = max(4, size[0] // 40)
        # 左の斜め線
        check_start_x = check_x - check_size // 2
        check_start_y = check_y
        check_mid_x = check_x
        check_mid_y = check_y + check_size // 3
        # 右の斜め線
        check_end_x = check_x + check_size // 2
        check_end_y = check_y - check_size // 6
        
        draw.line(
            [(check_start_x, check_start_y), (check_mid_x, check_mid_y)],
            fill=(46, 204, 113), width=check_thickness  # 鮮明な緑
        )
        draw.line(
            [(check_mid_x, check_mid_y), (check_end_x, check_end_y)],
            fill=(46, 204, 113), width=check_thickness
        )
    
    # === デザイン要素の説明 ===
    # 1. 白いワッシャー（中央）: 検査対象の製品を表現
    # 2. 青い背景: 工業的・技術的な雰囲気を表現
    # 3. カメラアイコン（上部中央）: 外観検査カメラを表現（リアルなカメラデザイン）
    # 4. 白い線（カメラ→ワッシャー）: 検査ビームを表現
    # 5. 緑のチェックマーク（ワッシャー右上）: 検査完了・合格を表現
    
    if output_path:
        img.save(output_path, 'PNG')
        print(f"✓ アイコン作成完了: {output_path}")
    
    return img

def create_logo_banner(width=800, height=200, output_path=None):
    """バナー用ロゴを作成（モダンでかっこいい横長デザイン）"""
    import math
    
    # グラデーション背景を作成
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # 背景グラデーション（上から下へ）
    for y in range(height):
        ratio = y / height
        r = int(255 - ratio * 10)
        g = int(255 - ratio * 10)
        b = int(255 - ratio * 5)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # 左側にアイコンバージョンを配置（大きく）
    icon_size = min(width // 3, height - 30)
    icon_img = create_logo_icon(size=(icon_size, icon_size))
    icon_x = 30
    icon_y = (height - icon_size) // 2
    img.paste(icon_img, (icon_x, icon_y))
    
    # テキストを追加（Cursor風の洗練されたスタイル）
    try:
        # モダンなフォントを使用
        if sys.platform.startswith('win'):
            # Windows: Segoe UI（よりモダンなフォント）
            try:
                font_large = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 62)
                font_small = ImageFont.truetype("C:/Windows/Fonts/segoeuisl.ttf", 24)
            except:
                font_large = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 62)
                font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
        else:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 62)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        # フォント読み込み失敗時はデフォルトフォント
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # タイトルの位置
    text_x = icon_x + icon_size + 35
    text_y = height // 2 - 45
    
    # タイトル（Cursor風の洗練されたテキストエフェクト）
    title = "WasherInspection"
    # 複数のシャドウレイヤー（より洗練された影）
    shadow_offsets = [(3, 3), (2, 2), (1, 1)]
    shadow_colors = [(15, 20, 25, 40), (25, 35, 45, 60), (35, 45, 55, 80)]
    for i, (offset_x, offset_y) in enumerate(shadow_offsets):
        shadow_color = shadow_colors[i][:3]
        draw.text((text_x + offset_x, text_y + offset_y), title, 
                  fill=shadow_color, font=font_large)
    
    # メインテキスト（グラデーション効果）
    main_color = (52, 152, 219)  # Cursor風の青
    draw.text((text_x, text_y), title, fill=main_color, font=font_large)
    
    # サブタイトル（Cursor風の控えめなスタイル）
    subtitle_y = text_y + 72
    subtitle = "AI-Powered Quality Inspection System"
    
    # サブタイトルのシャドウ
    draw.text((text_x + 1, subtitle_y + 1), subtitle, 
              fill=(90, 95, 100), font=font_small)
    # サブタイトル本体（控えめなグレー）
    draw.text((text_x, subtitle_y), subtitle, 
              fill=(127, 140, 141), font=font_small)
    
    # 装飾的なライン（Cursor風の洗練されたグラデーション）
    line_y = subtitle_y + 38
    line_x_start = text_x
    line_x_end = width - 50
    line_thickness = 3
    
    # グラデーションライン（より滑らかに）
    for i in range(line_x_end - line_x_start):
        ratio = i / (line_x_end - line_x_start)
        # Cursor風のブルーグレーグラデーション
        r = int(52 + (180 - 52) * ratio)
        g = int(152 + (200 - 152) * ratio)
        b = int(219 + (220 - 219) * ratio)
        draw.rectangle(
            [(line_x_start + i, line_y), (line_x_start + i + 1, line_y + line_thickness)],
            fill=(r, g, b)
        )
    
    if output_path:
        img.save(output_path, 'PNG')
        print(f"✓ バナーロゴ作成完了: {output_path}")
    
    return img

def create_splash_screen(width=600, height=400, output_path=None):
    """スプラッシュ画面用ロゴを作成（モダンでかっこいいデザイン）"""
    import math
    
    # グラデーション背景（上から下へ）
    img = Image.new('RGB', (width, height), color=(240, 248, 255))
    draw = ImageDraw.Draw(img)
    
    # 背景グラデーション
    for y in range(height):
        ratio = y / height
        r = int(240 - ratio * 15)
        g = int(248 - ratio * 20)
        b = int(255 - ratio * 25)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # 装飾的な円形グラデーション（背景）
    center_x, center_y = width // 2, height // 2
    max_radius = int(math.sqrt(width**2 + height**2))
    for radius in range(max_radius, 0, -5):
        alpha = max(0, 30 - radius // 20)
        color = (100 + alpha, 150 + alpha, 200 + alpha)
        if alpha > 0:
            draw.ellipse(
                [center_x - radius, center_y - radius,
                 center_x + radius, center_y + radius],
                fill=None, outline=color, width=1
            )
    
    # 中央にアイコン（大きめ）
    icon_size = min(width // 2.5, height // 2.5)
    icon_img = create_logo_icon(size=(int(icon_size), int(icon_size)))
    icon_x = int((width - icon_size) // 2)
    icon_y = int((height - icon_size) // 2 - 30)
    img.paste(icon_img, (icon_x, icon_y))
    
    # アイコンの後ろに光るエフェクト
    glow_radius = int(icon_size * 0.7)
    for i in range(5):
        glow_size = glow_radius + i * 8
        glow_alpha = max(0, 40 - i * 8)
        if glow_alpha > 0:
            glow_color = (min(255, 100 + glow_alpha), min(255, 150 + glow_alpha), min(255, 200 + glow_alpha))
            draw.ellipse(
                [center_x - glow_size, center_y - glow_size + 20,
                 center_x + glow_size, center_y + glow_size + 20],
                fill=None, outline=glow_color, width=2
            )
    
    # タイトル（Cursor風の洗練されたスタイル）
    try:
        if sys.platform.startswith('win'):
            try:
                # Segoe UIを使用
                font_title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 48)
                font_subtitle = ImageFont.truetype("C:/Windows/Fonts/segoeuisl.ttf", 22)
            except:
                font_title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 48)
                font_subtitle = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
        else:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
            font_subtitle = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except:
        font_title = ImageFont.load_default()
        font_subtitle = ImageFont.load_default()
    
    title_y = int(icon_y + icon_size + 30)
    title = "WasherInspection"
    
    # 中央揃えのためテキスト幅を取得
    bbox = draw.textbbox((0, 0), title, font=font_title)
    text_width = bbox[2] - bbox[0]
    title_x = int((width - text_width) // 2)
    
    # タイトル（Cursor風の洗練されたテキストエフェクト）
    shadow_offsets = [(3, 3), (2, 2), (1, 1)]
    shadow_colors = [(15, 20, 25, 40), (25, 35, 45, 60), (35, 45, 55, 80)]
    for i, (offset_x, offset_y) in enumerate(shadow_offsets):
        shadow_color = shadow_colors[i][:3]
        draw.text((title_x + offset_x, title_y + offset_y), title, 
                  fill=shadow_color, font=font_title)
    
    # メインテキスト（Cursor風の青）
    main_color = (52, 152, 219)
    draw.text((title_x, title_y), title, fill=main_color, font=font_title)
    
    # サブタイトル（Cursor風の控えめなスタイル）
    subtitle_y = title_y + 60
    subtitle = "統合ワッシャー検査・学習システム"
    
    bbox_sub = draw.textbbox((0, 0), subtitle, font=font_subtitle)
    text_width_sub = bbox_sub[2] - bbox_sub[0]
    subtitle_x = int((width - text_width_sub) // 2)
    
    # サブタイトルのシャドウ
    draw.text((subtitle_x + 1, subtitle_y + 1), subtitle, 
              fill=(90, 95, 100), font=font_subtitle)
    # サブタイトル本体
    draw.text((subtitle_x, subtitle_y), subtitle, 
              fill=(127, 140, 141), font=font_subtitle)
    
    # 下部に装飾ライン
    line_y = height - 30
    line_x_start = width // 4
    line_x_end = width * 3 // 4
    for i in range(line_x_end - line_x_start):
        ratio = i / (line_x_end - line_x_start)
        r = int(41 + (155 - 41) * ratio)
        g = int(128 + (89 - 128) * ratio)
        b = int(185 + (182 - 185) * ratio)
        draw.rectangle(
            [(line_x_start + i, line_y), (line_x_start + i + 1, line_y + 2)],
            fill=(r, g, b)
        )
    
    if output_path:
        img.save(output_path, 'PNG')
        print(f"✓ スプラッシュ画面作成完了: {output_path}")
    
    return img

def main():
    """メイン処理"""
    # アセットディレクトリを作成
    base_dir = Path(__file__).resolve().parents[2]
    assets_dir = base_dir / 'assets'
    assets_dir.mkdir(exist_ok=True)
    
    print("ロゴ画像を生成中...")
    
    # 1. アイコン（複数サイズ）
    icon_sizes = [16, 32, 48, 64, 128, 256]
    icon_images = []
    for size in icon_sizes:
        icon_img = create_logo_icon(size=(size, size))
        icon_path = assets_dir / f'logo_icon_{size}x{size}.png'
        icon_img.save(icon_path, 'PNG')
        icon_images.append(icon_img)
    
    # ICOファイルも作成（Windows用）
    try:
        ico_path = assets_dir / 'logo_icon.ico'
        icon_images[-1].save(ico_path, 'ICO', sizes=[(s, s) for s in icon_sizes])
        print(f"✓ ICOファイル作成完了: {ico_path}")
    except Exception as e:
        print(f"⚠ ICOファイル作成スキップ: {e}")
    
    # 2. バナーロゴ
    banner_path = assets_dir / 'logo_banner.png'
    create_logo_banner(output_path=banner_path)
    
    # 3. スプラッシュ画面
    splash_path = assets_dir / 'logo_splash.png'
    create_splash_screen(output_path=splash_path)
    
    print(f"\n✓ すべてのロゴ画像を {assets_dir} に作成しました。")

if __name__ == "__main__":
    main()

