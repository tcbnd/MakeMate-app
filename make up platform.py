from __future__ import annotations

import asyncio
import base64
import tempfile
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

try:
	import cv2
	import mediapipe as mp

	HAS_MEDIAPIPE = True
except Exception:
	HAS_MEDIAPIPE = False
	mp = None

MP_SOLUTIONS_FACE_MESH = None
MP_SOLUTIONS_FACE_DETECTION = None
HAS_MEDIAPIPE_TASKS = False
MP_TASKS_VISION = None

if HAS_MEDIAPIPE:
	# Some mediapipe builds expose FaceMesh under mediapipe.python.solutions.
	try:
		if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
			MP_SOLUTIONS_FACE_MESH = mp.solutions.face_mesh
	except Exception:
		MP_SOLUTIONS_FACE_MESH = None

	try:
		if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
			MP_SOLUTIONS_FACE_DETECTION = mp.solutions.face_detection
	except Exception:
		MP_SOLUTIONS_FACE_DETECTION = None

	if MP_SOLUTIONS_FACE_MESH is None:
		try:
			from mediapipe.python import solutions as mp_python_solutions

			if hasattr(mp_python_solutions, "face_mesh"):
				MP_SOLUTIONS_FACE_MESH = mp_python_solutions.face_mesh
			if hasattr(mp_python_solutions, "face_detection"):
				MP_SOLUTIONS_FACE_DETECTION = mp_python_solutions.face_detection
		except Exception:
			MP_SOLUTIONS_FACE_MESH = None

	try:
		from mediapipe.tasks.python import vision as mp_tasks_vision

		MP_TASKS_VISION = mp_tasks_vision
		HAS_MEDIAPIPE_TASKS = True
	except Exception:
		HAS_MEDIAPIPE_TASKS = False

try:
	from gtts import gTTS
	from gtts.lang import tts_langs

	HAS_GTTS = True
except Exception:
	HAS_GTTS = False

try:
	import edge_tts

	HAS_EDGE_TTS = True
except Exception:
	HAS_EDGE_TTS = False


@dataclass
class AnalysisResult:
	face_shape: str
	undertone: str
	notes: List[str]


@dataclass
class MakeupProgress:
	step_num: int
	foundation_applied: bool
	eyes_started: bool
	cheeks_applied: bool
	lips_applied: bool
	issues: List[str]


@dataclass
class NormalizedLandmark:
	x: float
	y: float
	z: float = 0.0


@dataclass
class FaceQuality:
	"""臉部掃描品質評分"""
	is_clear: bool
	brightness_level: float  # 0-100
	face_size_ratio: float  # 臉部佔圖像的百分比
	landmarks_confidence: float  # 0-100
	issues: List[str]
	quality_score: float  # 整體評分 0-100


# 底妝推薦數據庫 - 根據膚色推薦具體產品編號
FOUNDATION_DATABASE = {
	"warm": {
		"light": ["N10", "W12", "W13"],  # 淺暖
		"medium": ["N15", "W16", "W17"],  # 中暖
		"dark": ["N20", "W22", "W23"],  # 深暖
	},
	"cool": {
		"light": ["N10", "C12", "C13"],  # 淺冷
		"medium": ["N15", "C16", "C17"],  # 中冷
		"dark": ["N20", "C22", "C23"],  # 深冷
	},
	"neutral": {
		"light": ["N10", "N11"],  # 淺中性
		"medium": ["N15", "N16"],  # 中中性
		"dark": ["N20", "N21"],  # 深中性
	},
}


LIPSTICK_DATABASE = {
	"warm": {
		"日系清透": ["L-NT03 奶茶杏桃唇釉", "L-NT05 蜜桃奶茶唇泥"],
		"韓系水光": ["L-GL02 果凍蜜桃唇釉", "L-GL04 珊瑚水光唇釉"],
		"歐美立體": ["L-MT11 焦糖棕紅唇膏", "L-MT13 肉桂紅棕唇泥"],
		"晚宴 Glam": ["L-MT15 磚紅絲絨唇膏", "L-MT17 楓糖紅棕唇釉"],
		"日常自然": ["L-NT02 杏仁奶茶唇釉", "L-NT04 裸杏豆沙唇膏"],
	},
	"cool": {
		"日系清透": ["L-NT13 玫瑰奶茶唇釉", "L-NT15 冷粉豆沙唇膏"],
		"韓系水光": ["L-GL12 櫻花水光唇釉", "L-GL14 冷莓果凍唇釉"],
		"歐美立體": ["L-MT21 梅子紅絲絨唇膏", "L-MT23 酒紅霧面唇泥"],
		"晚宴 Glam": ["L-MT25 莓果深紅唇膏", "L-MT27 紫調紅棕唇釉"],
		"日常自然": ["L-NT12 裸玫瑰唇膏", "L-NT14 粉豆沙唇釉"],
	},
	"neutral": {
		"日系清透": ["L-NT23 裸奶茶唇釉", "L-NT25 杏粉豆沙唇膏"],
		"韓系水光": ["L-GL22 蜜桃玫瑰唇釉", "L-GL24 透明紅棕唇釉"],
		"歐美立體": ["L-MT31 肉桂棕紅唇膏", "L-MT33 番茄棕唇泥"],
		"晚宴 Glam": ["L-MT35 深楓葉紅唇膏", "L-MT37 摩卡紅棕唇釉"],
		"日常自然": ["L-NT22 奶油豆沙唇釉", "L-NT24 裸粉棕唇膏"],
	},
}


FACE_SHAPE_GUIDE = {
	"round": "重點放喺拉長輪廓，同提升立體感。",
	"oval": "比例平均，適合大部分妝容路線。",
	"square": "柔化線條，強調眼妝或唇妝平衡輪廓。",
	"heart": "平衡額頭與下巴視覺比例。",
	"long": "橫向擴展視覺，減少面中拉長感。",
	"unknown": "建議再影一張正面、光線均勻嘅相提升準確度。",
}


LOOK_LIBRARY: Dict[str, Dict[str, Dict[str, List[str]]]] = {
	"japan_clean": {
		"name": "日系清透妝",
		"products": {
			"warm": ["輕薄奶油底妝", "杏桃裸色胭脂", "奶茶豆沙唇釉", "柔霧啡杏眼影"],
			"cool": ["輕薄粉調底妝", "櫻粉胭脂", "玫瑰奶茶唇釉", "灰粉奶茶眼影"],
			"neutral": ["自然貼膚底妝", "裸杏胭脂", "裸玫瑰唇釉", "米棕霧面眼影"],
		},
		"tools": ["美妝蛋", "小號眼影掃", "細節遮瑕掃", "定妝噴霧"],
	},
	"everyday_natural": {
		"name": "日常清透妝",
		"products": {
			"warm": ["暖調底妝", "蜜桃色胭脂", "奶茶色唇釉", "啡金眼影"],
			"cool": ["中性偏粉底妝", "豆沙色胭脂", "玫瑰色唇釉", "灰粉眼影"],
			"neutral": ["自然色底妝", "杏桃胭脂", "裸粉唇膏", "大地色眼影"],
		},
		"tools": ["粉底掃/美妝蛋", "暈染眼影掃", "斜角眉掃", "碎粉掃"],
	},
	"party_glam": {
		"name": "晚宴立體妝",
		"products": {
			"warm": ["啞光底妝", "古銅修容", "橘棕唇膏", "金啡珠光眼影"],
			"cool": ["半啞光底妝", "灰棕修容", "莓果唇膏", "銀灰煙燻眼影"],
			"neutral": ["持妝底妝", "中性修容", "楓葉紅唇膏", "棕金眼影"],
		},
		"tools": ["修容掃", "眼線筆", "睫毛夾", "高光掃"],
	},
	"soft_korean": {
		"name": "柔和韓系妝",
		"products": {
			"warm": ["水光底妝", "杏橘胭脂", "果汁唇釉", "奶茶粉眼影"],
			"cool": ["水潤底妝", "冷粉胭脂", "櫻花粉唇釉", "藕粉眼影"],
			"neutral": ["輕薄底妝", "蜜粉色胭脂", "裸玫瑰唇釉", "米棕眼影"],
		},
		"tools": ["細節遮瑕掃", "腮紅掃", "臥蠶掃", "定妝噴霧"],
	},
}


STYLE_LOOK_ORDER = {
	"日系清透": ["japan_clean", "everyday_natural", "soft_korean"],
	"韓系水光": ["soft_korean", "japan_clean", "everyday_natural"],
	"歐美立體": ["party_glam", "everyday_natural", "soft_korean"],
	"晚宴 Glam": ["party_glam", "soft_korean", "everyday_natural"],
	"日常自然": ["everyday_natural", "japan_clean", "soft_korean"],
}


def normalize_style_name(style: str) -> str:
	"""把常見中英風格名稱統一到系統內部風格鍵。"""
	if not style:
		return "日常自然"
	s = str(style).strip().lower()
	if any(k in s for k in ["japan", "日系", "jp"]):
		return "日系清透"
	if any(k in s for k in ["korea", "korean", "韓系", "hk"]):
		return "韓系水光"
	if any(k in s for k in ["us", "europe", "western", "歐美", "美式"]):
		return "歐美立體"
	if any(k in s for k in ["glam", "晚宴", "party"]):
		return "晚宴 Glam"
	if any(k in s for k in ["自然", "daily", "everyday", "日常"]):
		return "日常自然"
	return style if style in STYLE_LOOK_ORDER else "日常自然"


def profile_signature(profile: Dict[str, object]) -> Tuple[str, str, str, str, str]:
	"""用於判斷直播模式偏好是否有變更。"""
	style = normalize_style_name(str(profile.get("style", "日常自然")))
	return (
		style,
		str(profile.get("occasion", "日常上學/上班")),
		str(profile.get("finish", "自然貼膚")),
		str(profile.get("skin_concern", "一般")),
		str(profile.get("skill_level", "新手")),
	)


def estimate_undertone(image_np: np.ndarray) -> str:
	small = image_np[::8, ::8, :].astype(np.float32)
	r, g, b = small[:, :, 0].mean(), small[:, :, 1].mean(), small[:, :, 2].mean()
	if r - b > 12:
		return "warm"
	if b - r > 12:
		return "cool"
	return "neutral"


def check_face_quality(image_np: np.ndarray) -> FaceQuality:
	"""檢查臉部掃描品質"""
	issues = []
	brightness_score = 0.0
	face_size_score = 0.0
	landmarks_score = 0.0
	
	# 1. 檢查亮度
	gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
	brightness = np.mean(gray) / 255 * 100
	brightness_score = min(100, brightness * 1.5)  # 標準化到 0-100
	
	if brightness < 50:
		issues.append("⚠️ 太暗：請增加光線，特別是臉部照明")
	elif brightness > 200:
		issues.append("⚠️ 太亮：避免直射陽光，造成反光")
	
	# 2. 檢查臉部大小
	landmarks = get_landmarks_from_image(image_np)
	if landmarks:
		h, w = image_np.shape[:2]
		y_coords = [lm.y * h for lm in landmarks]
		x_coords = [lm.x * w for lm in landmarks]
		face_height = max(y_coords) - min(y_coords)
		face_area_ratio = (face_height / h) * 100
		face_size_score = min(100, face_area_ratio * 1.5)
		
		if face_area_ratio < 30:
			issues.append("⚠️ 臉部太小：請靠近相機")
		elif face_area_ratio > 90:
			issues.append("⚠️ 臉部超出框：請退後一點")
		else:
			landmarks_score = 90  # 成功偵測到關鍵點
	else:
		face_bbox = detect_face_bbox(image_np)
		if face_bbox:
			x, y, w, h = face_bbox
			bbox_area_ratio = (h / image_np.shape[0]) * 100
			face_size_score = min(100, bbox_area_ratio * 1.5)
			landmarks_score = 60  # 只有臉框，沒有關鍵點
			if bbox_area_ratio < 30:
				issues.append("⚠️ 臉部太小：請靠近相機")
		else:
			issues.append("❌ 無法偵測到臉部：請確保臉部完整在鏡頭內")
			landmarks_score = 0
	
	# 3. 檢查正面對向
	if landmarks and len(landmarks) > 100:
		# 檢查左右眼睛是否對稱
		left_eye = [lm.x for lm in landmarks[33:46]]
		right_eye = [lm.x for lm in landmarks[263:276]]
		eye_asymmetry = abs(np.mean(left_eye) - np.mean(right_eye))
		if eye_asymmetry > 0.2:
			issues.append("⚠️ 角度偏斜：請正對著鏡頭")
	
	# 總體評分
	quality_score = (brightness_score + face_size_score + landmarks_score) / 3
	is_clear = quality_score > 70 and len(issues) == 0
	
	return FaceQuality(
		is_clear=is_clear,
		brightness_level=brightness_score,
		face_size_ratio=face_size_score,
		landmarks_confidence=landmarks_score,
		issues=issues,
		quality_score=quality_score,
	)


def get_skin_tone_depth(image_np: np.ndarray) -> str:
	"""偵測膚色深度（淺、中、深）"""
	face_bbox = detect_face_bbox(image_np)
	if not face_bbox:
		return "medium"  # 預設中膚色
	
	x, y, w, h = face_bbox
	face_region = image_np[y:y+h, x:x+w]
	
	# 取臉頰區域（通常最能代表膚色）
	cheek_y_start = int(h * 0.35)
	cheek_y_end = int(h * 0.65)
	cheek_x_start = int(w * 0.15)
	cheek_x_end = int(w * 0.35)
	
	if cheek_y_end > cheek_y_start and cheek_x_end > cheek_x_start:
		cheek = face_region[cheek_y_start:cheek_y_end, cheek_x_start:cheek_x_end]
		# 計算亮度
		brightness = np.mean(cv2.cvtColor(cheek, cv2.COLOR_RGB2GRAY))
		if brightness < 100:
			return "dark"
		elif brightness < 150:
			return "medium"
		else:
			return "light"
	
	return "medium"


def recommend_foundation_shade(
	undertone: str,
	depth: str,
	style: str = "日常自然",
	finish: str = "自然貼膚",
	skin_concern: str = "一般",
) -> List[str]:
	"""根據膚色、風格、妝感與膚質，推薦 1-2 個最適合底妝編號。"""
	try:
		style = normalize_style_name(style)
		base = list(FOUNDATION_DATABASE.get(undertone, {}).get(depth, ["N15", "N16"]))
		candidates = base.copy()

		# 風格偏好調整：日系/韓系偏亮半階，歐美/晚宴偏完整覆蓋與原膚深
		if style in ["日系清透", "韓系水光"]:
			if depth == "medium":
				candidates = ["N14", *candidates, "W15" if undertone == "warm" else "C15"]
			elif depth == "dark":
				candidates = ["N19", *candidates]
		elif style in ["歐美立體", "晚宴 Glam"]:
			if depth == "light":
				candidates = [*candidates, "N12"]
			elif depth == "medium":
				candidates = [*candidates, "N17"]

		# 妝感偏好：高遮瑕偏原膚深，水光偏亮半階
		if finish == "水光透亮":
			candidates = [candidates[0], *candidates, "N13"]
		elif finish == "高遮瑕":
			if depth == "medium":
				candidates = [c for c in candidates if c not in ["N13", "N14"]] + ["N17"]

		# 膚質偏好：偏油/易脫妝偏穩定色階，偏乾可接受半階提亮
		if skin_concern in ["偏油", "容易脫妝"]:
			candidates = [c for c in candidates if c not in ["N13"]]
		elif skin_concern == "偏乾":
			candidates = [*candidates, "N13"]

		# 去重保序
		seen = set()
		final = []
		for c in candidates:
			if c not in seen:
				seen.add(c)
				final.append(c)

		# 僅回傳 1-2 個：主推 + 備選
		if not final:
			return [base[0]]
		if len(final) == 1:
			return [final[0]]

		primary = final[0]
		secondary = final[1]
		# 高遮瑕或晚宴，偏向較穩定原膚深；水光/日系可保留提亮備選
		if finish in ["高遮瑕", "柔霧啞光"] or style in ["歐美立體", "晚宴 Glam"]:
			return [primary]
		return [primary, secondary]
	except Exception:
		return ["N15"]


def recommend_lip_products(
	undertone: str,
	style: str,
	occasion: str = "日常上學/上班",
) -> List[str]:
	"""根據風格與膚色推薦 1-2 個唇妝選擇。"""
	style = normalize_style_name(style)
	base = LIPSTICK_DATABASE.get(undertone, LIPSTICK_DATABASE["neutral"])
	options = list(base.get(style, base["日常自然"]))

	# 場合調整：晚宴/活動偏濃，日常偏自然
	if occasion in ["拍照/活動", "晚宴"]:
		options = options[::-1]

	if not options:
		return ["L-NT22 奶油豆沙唇釉"]
	if len(options) == 1:
		return [options[0]]
	return [options[0], options[1]]


def recommend_step_tools(look_tools: List[str], inventory: List[str], style: str = "日常自然") -> Dict[int, List[str]]:
	"""按步驟推薦工具，若化妝包有對應工具會優先顯示。"""
	style = normalize_style_name(style)
	style_bias_tools = {
		"日系清透": {
			2: ["美妝蛋", "薄型粉撲"],
			3: ["小號眼影掃", "細節暈染掃"],
			5: ["唇刷", "棉花棒"],
		},
		"韓系水光": {
			2: ["氣墊粉撲", "水光妝前乳"],
			3: ["臥蠶掃", "細眼線筆"],
			5: ["水光唇釉", "透明唇蜜"],
		},
		"歐美立體": {
			2: ["粉底掃", "遮瑕掃"],
			3: ["扁平眼影掃", "眼線膠筆"],
			4: ["修容掃", "高光掃"],
			5: ["唇線筆", "霧面唇刷"],
		},
		"晚宴 Glam": {
			2: ["粉底掃", "高遮瑕遮瑕掃"],
			3: ["煙燻暈染掃", "濃密睫毛夾"],
			4: ["修容掃", "高光掃"],
			5: ["唇線筆", "絲絨唇刷"],
		},
		"日常自然": {
			2: ["美妝蛋", "粉底液/氣墊"],
			3: ["暈染眼影掃", "眼線筆"],
			5: ["唇釉/唇膏"],
		},
	}
	step_defaults = {
		1: ["洗面產品", "保濕產品", "妝前乳"],
		2: ["粉底液/氣墊", "美妝蛋", "粉底掃"],
		3: ["眼影盤", "暈染眼影掃", "眼線筆"],
		4: ["胭脂/修容", "腮紅掃", "修容掃"],
		5: ["唇線筆（可選）", "唇釉/唇膏"],
		6: ["碎粉/蜜粉餅", "大粉掃", "定妝噴霧"],
	}

	inventory_lower = [x.lower() for x in inventory]
	result: Dict[int, List[str]] = {}
	for step, defaults in step_defaults.items():
		chosen = []
		for item in inventory:
			item_lower = item.lower()
			if step == 2 and any(k in item_lower for k in ["粉底", "氣墊", "妝前", "美妝蛋", "粉撲", "粉底掃"]):
				chosen.append(item)
			elif step == 3 and any(k in item_lower for k in ["眼影", "眼線", "眉", "睫毛"]):
				chosen.append(item)
			elif step == 4 and any(k in item_lower for k in ["胭脂", "腮紅", "修容", "高光"]):
				chosen.append(item)
			elif step == 5 and any(k in item_lower for k in ["唇", "lip"]):
				chosen.append(item)
			elif step == 6 and any(k in item_lower for k in ["定妝", "蜜粉", "碎粉", "噴霧"]):
				chosen.append(item)

		if step == 1:
			# 第一步可提示保養類，無則用預設
			chosen = [item for item in inventory if any(k in item.lower() for k in ["保濕", "洗面", "潔面", "妝前"])]

		# 先補上風格偏向工具（最多 2 個）
		for style_tool in style_bias_tools.get(style, {}).get(step, []):
			if style_tool not in chosen:
				chosen.append(style_tool)

		# 再補上 look 的通用工具（最多 1 個）
		if look_tools:
			tool_pick = look_tools[(step - 1) % len(look_tools)]
			if tool_pick not in chosen:
				chosen.append(tool_pick)

		if not chosen:
			chosen = defaults[:2]
		result[step] = chosen[:3]

	return result


def verify_step_completion(
	previous_image: Optional[np.ndarray],
	current_image: np.ndarray,
	step_num: int,
) -> Dict:
	"""驗證用戶是否完成了當前步驟"""
	verification_result = {
		"completed": False,
		"confidence": 0.0,
		"feedback": "無法驗證步驟",
		"tips": [],
	}

	def _face_metrics(img: np.ndarray) -> Optional[Dict[str, float]]:
		if cv2 is None:
			return None
		bbox = detect_face_bbox(img)
		if bbox is None:
			return None
		x, y, w, h = bbox
		img_h, img_w = img.shape[:2]
		x1, y1 = max(0, x), max(0, y)
		x2, y2 = min(img_w, x + w), min(img_h, y + h)
		if x2 <= x1 or y2 <= y1:
			return None
		face = img[y1:y2, x1:x2]
		if face.size == 0:
			return None
		gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
		hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
		lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
		brightness = float(np.mean(gray))
		sat = float(np.mean(hsv[:, :, 1]))
		uniformity = float(1.0 / (np.std(gray) + 1e-6))
		return {
			"lap_var": lap_var,
			"brightness": brightness,
			"sat": sat,
			"uniformity": uniformity,
		}

	current_landmarks = get_landmarks_from_image(current_image)
	current_bbox = detect_face_bbox(current_image)
	if current_landmarks is None and current_bbox is None:
		verification_result["feedback"] = "❌ 未偵測到清晰人臉，請正面拍照再試。"
		verification_result["tips"] = ["保持正面", "鏡頭距離約 35-50cm", "光線由正前方照臉"]
		return verification_result

	regions = detect_makeup_regions(current_image, current_landmarks) if current_landmarks is not None else detect_makeup_regions_by_bbox(current_image, current_bbox)

	if step_num == 1:
		verification_result["completed"] = True
		verification_result["confidence"] = 0.7 if current_landmarks is not None else 0.55
		verification_result["feedback"] = "✅ 可以進入底妝步驟。"
		verification_result["tips"] = ["保濕後等 1-2 分鐘再上底妝", "鼻翼和嘴角可先薄上一層妝前乳"]

	elif step_num == 2:
		if previous_image is None:
			verification_result["feedback"] = "⚠️ 缺少素顏基準照，請先重新拍初始照再驗證底妝。"
			verification_result["tips"] = ["先在第一步拍一張素顏清晰照"]
			verification_result["completed"] = False
			verification_result["confidence"] = 0.2
			return verification_result

		prev_m = _face_metrics(previous_image)
		curr_m = _face_metrics(current_image)
		if prev_m is None or curr_m is None:
			verification_result["feedback"] = "⚠️ 無法穩定比較底妝前後，請保持同角度再拍一次。"
			verification_result["tips"] = ["正面拍攝", "避免頭部大幅傾斜", "保持同一光線"]
			verification_result["completed"] = False
			verification_result["confidence"] = 0.25
			return verification_result

		texture_drop = prev_m["lap_var"] - curr_m["lap_var"]
		uniform_gain = curr_m["uniformity"] - prev_m["uniformity"]
		brightness_shift = curr_m["brightness"] - prev_m["brightness"]

		score = 0.0
		if texture_drop > 8.0:
			score += 0.45
		if uniform_gain > 0.002:
			score += 0.35
		if -8.0 <= brightness_shift <= 24.0:
			score += 0.20

		completed = score >= 0.72
		verification_result["completed"] = completed
		verification_result["confidence"] = min(0.95, max(0.1, score))
		if completed:
			verification_result["feedback"] = "✅ 底妝有均勻提升！可以進入眉眼步驟。"
			verification_result["tips"] = ["邊界再用海綿輕拍 5-8 下", "鼻翼與髮際線再薄暈開一層"]
		else:
			verification_result["feedback"] = "⚠️ 目前未見到底妝『薄拍均勻』變化，請再薄薄分層拍開。"
			verification_result["tips"] = ["先少量上臉，再第二層補足", "重點拍開額頭、兩頰、下巴", "同角度再拍一張以便比較"]

	elif step_num == 3:
		verification_result["completed"] = bool(regions.get("eyes", False))
		verification_result["confidence"] = 0.8 if verification_result["completed"] else 0.4
		verification_result["feedback"] = "✅ 眼妝變化已偵測到，可以進入修容。" if verification_result["completed"] else "⚠️ 眼影或眼線變化未明顯，請加強睫毛根部與眼尾。"
		verification_result["tips"] = ["眼線貼近睫毛根部", "眼尾向外 2-3mm 即可"]

	elif step_num == 4:
		verification_result["completed"] = bool(regions.get("cheeks", False))
		verification_result["confidence"] = 0.78 if verification_result["completed"] else 0.38
		verification_result["feedback"] = "✅ 顴骨/面中立體感已出現，可以進入唇妝。" if verification_result["completed"] else "⚠️ 腮紅或修容未夠明顯，請集中顴骨上方再暈開。"
		verification_result["tips"] = ["由顴骨往太陽穴斜向掃開", "避免一次上太深"]

	elif step_num == 5:
		verification_result["completed"] = bool(regions.get("lips", False))
		verification_result["confidence"] = 0.82 if verification_result["completed"] else 0.36
		verification_result["feedback"] = "✅ 唇妝完成度不錯，可以進入定妝。" if verification_result["completed"] else "⚠️ 唇妝變化未明顯，請先中間加深再向外暈開。"
		verification_result["tips"] = ["先畫內層再抿開", "嘴角保留 10% 漸層更自然"]

	elif step_num == 6:
		completed = bool(regions.get("forehead", False) or regions.get("eyes", False) or regions.get("lips", False))
		verification_result["completed"] = completed
		verification_result["confidence"] = 0.9 if completed else 0.5
		verification_result["feedback"] = "🎉 定妝完成，整體妝感已穩定！" if completed else "⚠️ 請補定妝並檢查鼻翼、眼下、嘴角後再拍。"
		verification_result["tips"] = ["T 字位先定妝，再掃全臉", "最後噴定妝噴霧 1-2 層"]
	
	return verification_result


def render_teacher_demo_frame(image_np: np.ndarray, step_num: int, undertone: str, layer_idx: int = 1) -> np.ndarray:
	"""生成 AR 老師示範畫面（基於用戶相片）。"""
	if cv2 is None:
		return image_np

	frame = image_np.copy()
	h, w = frame.shape[:2]
	landmarks = get_landmarks_from_image(image_np)
	face_bbox = detect_face_bbox(image_np)
	x1, y1, x2, y2, cx, cy, bw, bh = get_face_anchor_box(image_np, landmarks=landmarks, face_bbox=face_bbox)

	if step_num == 2:
		# 底妝示範：分三層薄拍
		base = apply_virtual_foundation(frame, undertone)
		alpha = {1: 0.18, 2: 0.30, 3: 0.42}.get(layer_idx, 0.30)
		frame = cv2.addWeighted(frame, 1.0 - alpha, base, alpha, 0)

		# 示範拍開路線（由中向外）
		center = (cx, int(cy + 0.06 * bh))
		arrow_color = (80, 235, 255)
		motion = {1: 0, 2: 14, 3: 26}.get(layer_idx, 0)
		for end in [
			(int(x1 + 0.12 * bw), int(y1 + 0.38 * bh)),
			(int(x1 + 0.88 * bw), int(y1 + 0.38 * bh)),
			(int(x1 + 0.20 * bw), int(y1 + 0.72 * bh)),
			(int(x1 + 0.80 * bw), int(y1 + 0.72 * bh)),
		]:
			jitter_end = (end[0] + (motion if end[0] > center[0] else -motion), end[1])
			cv2.arrowedLine(frame, center, jitter_end, arrow_color, 3, tipLength=0.12)
		cv2.putText(frame, f"AR Layer {layer_idx}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, arrow_color, 2)
		cv2.putText(frame, "Tap and blend outward", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.65, arrow_color, 2)
		return frame

	# 其他步驟：顯示焦點區域 + 會動的指示點
	frame = draw_step_guide_overlay(frame, step_num, landmarks=landmarks, face_bbox=face_bbox)
	if step_num == 1:
		# 打圈清潔/保濕: 3 個循環圈
		for r in [36 + layer_idx * 5, 70 + layer_idx * 6, 104 + layer_idx * 7]:
			cv2.circle(frame, (cx, cy), r, (80, 235, 255), 2)
		cv2.putText(frame, "Prep: circular motion", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 235, 255), 2)
	elif step_num == 3:
		cv2.arrowedLine(frame, (int(x1 + 0.30 * bw), int(y1 + 0.34 * bh)), (int(x1 + 0.42 * bw), int(y1 + 0.27 * bh)), (80, 235, 255), 3, tipLength=0.15)
		cv2.arrowedLine(frame, (int(x1 + 0.70 * bw), int(y1 + 0.34 * bh)), (int(x1 + 0.58 * bw), int(y1 + 0.27 * bh)), (80, 235, 255), 3, tipLength=0.15)
		cv2.putText(frame, "Eyes: blend upward", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 235, 255), 2)
	elif step_num == 4:
		cv2.arrowedLine(frame, (int(x1 + 0.30 * bw), int(y1 + 0.60 * bh)), (int(x1 + 0.14 * bw), int(y1 + 0.45 * bh)), (80, 235, 255), 3, tipLength=0.15)
		cv2.arrowedLine(frame, (int(x1 + 0.70 * bw), int(y1 + 0.60 * bh)), (int(x1 + 0.86 * bw), int(y1 + 0.45 * bh)), (80, 235, 255), 3, tipLength=0.15)
		cv2.putText(frame, "Cheeks: sweep outward", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 235, 255), 2)
	elif step_num == 5:
		lip_center = (cx, int(y1 + 0.78 * bh))
		cv2.circle(frame, lip_center, 10 + 4 * layer_idx, (80, 235, 255), 2)
		cv2.putText(frame, "Lips: dark center then blur", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 235, 255), 2)
	elif step_num == 6:
		t_zone = np.array(
			[
				[int(x1 + 0.45 * bw), int(y1 + 0.10 * bh)],
				[int(x1 + 0.55 * bw), int(y1 + 0.10 * bh)],
				[int(x1 + 0.53 * bw), int(y1 + 0.82 * bh)],
				[int(x1 + 0.47 * bw), int(y1 + 0.82 * bh)],
			],
			dtype=np.int32,
		)
		cv2.polylines(frame, [t_zone], True, (80, 235, 255), 2)
		cv2.putText(frame, "Set: T-zone first", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 235, 255), 2)
	else:
		cv2.putText(frame, "Teacher demo", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 235, 255), 2)
	return frame


def get_face_anchor_box(
	image_np: np.ndarray,
	landmarks: Optional[List] = None,
	face_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[int, int, int, int, int, int, int, int]:
	"""取得臉部錨點與邊界，避免 AR 指引固定在畫面中心。"""
	h, w = image_np.shape[:2]

	if landmarks:
		xs = [int(lm.x * w) for lm in landmarks]
		ys = [int(lm.y * h) for lm in landmarks]
		if xs and ys:
			x1 = max(0, min(xs))
			y1 = max(0, min(ys))
			x2 = min(w - 1, max(xs))
			y2 = min(h - 1, max(ys))
			bw = max(1, x2 - x1)
			bh = max(1, y2 - y1)
			pad_x = int(bw * 0.12)
			pad_y = int(bh * 0.16)
			x1 = max(0, x1 - pad_x)
			y1 = max(0, y1 - pad_y)
			x2 = min(w - 1, x2 + pad_x)
			y2 = min(h - 1, y2 + pad_y)
			bw = max(1, x2 - x1)
			bh = max(1, y2 - y1)
			cx = x1 + bw // 2
			cy = y1 + bh // 2
			return x1, y1, x2, y2, cx, cy, bw, bh

	if face_bbox is not None:
		x, y, bw, bh = face_bbox
		x1 = max(0, x)
		y1 = max(0, y)
		x2 = min(w - 1, x + bw)
		y2 = min(h - 1, y + bh)
		bw = max(1, x2 - x1)
		bh = max(1, y2 - y1)
		cx = x1 + bw // 2
		cy = y1 + bh // 2
		return x1, y1, x2, y2, cx, cy, bw, bh

	# 找不到臉時才退回畫面中心
	bw = max(1, int(w * 0.42))
	bh = max(1, int(h * 0.56))
	cx, cy = w // 2, h // 2
	x1 = max(0, cx - bw // 2)
	y1 = max(0, cy - bh // 2)
	x2 = min(w - 1, cx + bw // 2)
	y2 = min(h - 1, cy + bh // 2)
	return x1, y1, x2, y2, cx, cy, bw, bh


def generate_teacher_demo_video(image_np: np.ndarray, step_num: int, undertone: str) -> Optional[str]:
	"""把 AR 老師示範輸出為動畫 GIF。

	為了避免瀏覽器影片編碼/MIME 相容問題，統一使用 GIF。
	"""
	if cv2 is None:
		return None

	# 生成連續分鏡：每一步都要有明顯動畫變化，避免看起來像靜態相片。
	frames: List[np.ndarray] = []
	if step_num == 2:
		for layer in [1, 2, 3, 2, 1]:
			for _ in range(4):
				frames.append(render_teacher_demo_frame(image_np, step_num, undertone, layer_idx=layer))
	else:
		for i in range(24):
			layer_idx = (i % 6) + 1
			frames.append(render_teacher_demo_frame(image_np, step_num, undertone, layer_idx=layer_idx))

	if not frames:
		return None

	fps = 8
	uid = uuid.uuid4().hex
	# 統一輸出動畫 GIF，保證畫面會「動」
	gif_path = Path(tempfile.gettempdir()) / f"ar_teacher_{step_num}_{uid}.gif"
	try:
		pil_frames = [Image.fromarray(fr.astype(np.uint8)) for fr in frames]
		if not pil_frames:
			return None
		duration_ms = int(1000 / max(fps, 1))
		pil_frames[0].save(
			gif_path,
			save_all=True,
			append_images=pil_frames[1:],
			duration=duration_ms,
			loop=0,
			format="GIF",
		)
		if gif_path.exists() and gif_path.stat().st_size > 1024:
			return str(gif_path)
		return None
	except Exception:
		return None


def render_gif_animation(gif_path: str, caption: str = "AR 老師動畫示範") -> None:
	"""穩定顯示 GIF 動畫（避免部份環境只顯示第一幀）。"""
	try:
		gif_bytes = Path(gif_path).read_bytes()
		b64 = base64.b64encode(gif_bytes).decode("ascii")
		st.markdown(
			f"<img src='data:image/gif;base64,{b64}' style='width:100%; border-radius:12px;' alt='{caption}' />",
			unsafe_allow_html=True,
		)
		st.caption(caption)
	except Exception:
		st.image(gif_path, caption=caption, use_container_width=True)


def apply_virtual_foundation(image_np: np.ndarray, undertone: str) -> np.ndarray:
	"""在臉部應用虛擬底妝 AR 效果 - 展示「薄薄分層拍開」"""
	if cv2 is None:
		return image_np
	
	landmarks = get_landmarks_from_image(image_np)
	face_bbox = detect_face_bbox(image_np)
	
	if landmarks is None and face_bbox is None:
		return image_np
	
	result = image_np.copy()
	h, w = image_np.shape[:2]
	
	# 使用所有臉部特徵點來精準覆蓋臉部區域
	if landmarks is not None:
		face_points = np.array(
			[[int(lm.x * w), int(lm.y * h)] for lm in landmarks],
			dtype=np.int32,
		)
		# 尋找凸包來定義臉部邊界
		if len(face_points) > 3:
			try:
				hull = cv2.convexHull(face_points)
			except:
				hull = None
		else:
			hull = None
	else:
		hull = None
	
	# 如果無法從 landmarks 取得，用人臉框
	if hull is None and face_bbox is not None:
		x, y, w_box, h_box = face_bbox
		hull = np.array([
			[x, y],
			[x + w_box, y],
			[x + w_box, y + h_box],
			[x, y + h_box]
		], dtype=np.int32)
	
	if hull is None:
		return image_np
	
	# 根據膚色選擇底妝顏色（更自然的顏色）
	if undertone == "warm":
		# 暖色底妝 - 淺黃色系
		foundation_color = (175, 155, 130)  # BGR 格式
		highlight_color = (200, 180, 160)    # 高光色
	elif undertone == "cool":
		# 冷色底妝 - 淺粉色系
		foundation_color = (185, 145, 150)
		highlight_color = (210, 170, 180)
	else:
		# 中性底妝
		foundation_color = (180, 150, 140)
		highlight_color = (205, 175, 165)
	
	# 創建虛擬底妝層 - 展示「薄薄分層拍開」效果
	mask = np.zeros((h, w), dtype=np.uint8)
	cv2.drawContours(mask, [hull], 0, 255, -1)
	
	# 軟化邊界（展示「邊位暈開」）
	mask_blurred = cv2.GaussianBlur(mask, (35, 35), 0)
	mask_3channel = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR) / 255.0
	
	# 底妝基礎層（30% 透明度 - 表示「薄薄」）
	foundation_layer = np.full((h, w, 3), foundation_color, dtype=np.uint8)
	result = (result * (1 - mask_3channel * 0.30) + foundation_layer * mask_3channel * 0.30).astype(np.uint8)
	
	# 在高光區域（額頭、顴骨、鼻樑）添加亮色層（展示「分層」效果）
	if landmarks is not None:
		# 額頭中央
		forehead_y = int(h * 0.2)
		forehead_points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks if int(lm.y * h) < forehead_y]
		if forehead_points:
			center_x = int(np.mean([pt[0] for pt in forehead_points]))
			cv2.circle(result, (center_x, forehead_y), 50, highlight_color, -1)
			cv2.circle(result, (center_x, forehead_y), 50, highlight_color, -1)
		
		# 顴骨
		cheek_points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks if 0.4 < lm.y < 0.6]
		if cheek_points:
			# 左顴骨
			left_cheeks = [pt for pt in cheek_points if pt[0] < w/2]
			if left_cheeks:
				left_x = int(np.mean([pt[0] for pt in left_cheeks]))
				left_y = int(np.mean([pt[1] for pt in left_cheeks]))
				cv2.circle(result, (left_x, left_y), 60, highlight_color, -1)
			
			# 右顴骨
			right_cheeks = [pt for pt in cheek_points if pt[0] > w/2]
			if right_cheeks:
				right_x = int(np.mean([pt[0] for pt in right_cheeks]))
				right_y = int(np.mean([pt[1] for pt in right_cheeks]))
				cv2.circle(result, (right_x, right_y), 60, highlight_color, -1)
	
	# 混合高光，使其更自然
	result = cv2.addWeighted(result, 0.7, result, 0.3, 0)
	
	# 模糊處理高光區域，使邊界自然
	result = cv2.GaussianBlur(result, (5, 5), 0)
	
	return result


def draw_step_guide_overlay(
	image_np: np.ndarray,
	step_num: int,
	landmarks: Optional[List] = None,
	face_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
	"""在圖像上繪製步驟指導區域"""
	if cv2 is None:
		return image_np
	
	result = image_np.copy()
	x1, y1, x2, y2, cx, cy, bw, bh = get_face_anchor_box(image_np, landmarks=landmarks, face_bbox=face_bbox)

	def _pt(rx: float, ry: float) -> List[int]:
		return [int(x1 + rx * bw), int(y1 + ry * bh)]
	
	# 定義各步驟的區域
	regions = {
		2: ("forehead", "額頭和臉頰"),  # 底妝
		3: ("eyes", "眼睛周圍"),  # 眼妝
		4: ("cheeks", "顴骨和下巴"),  # 修容/腮紅
		5: ("lips", "嘴唇"),  # 唇妝
	}
	
	if step_num in regions:
		region_name, label = regions[step_num]
		
		# 根據區域繪製高亮框
		if region_name == "forehead":
			pts = np.array(
				[
					_pt(0.10, 0.02),
					_pt(0.90, 0.02),
					_pt(0.90, 0.34),
					_pt(0.10, 0.34),
				],
				dtype=np.int32,
			)
		elif region_name == "eyes":
			pts = np.array(
				[
					_pt(0.12, 0.20),
					_pt(0.88, 0.20),
					_pt(0.88, 0.50),
					_pt(0.12, 0.50),
				],
				dtype=np.int32,
			)
		elif region_name == "cheeks":
			pts = np.array(
				[
					_pt(0.08, 0.42),
					_pt(0.92, 0.42),
					_pt(0.92, 0.78),
					_pt(0.08, 0.78),
				],
				dtype=np.int32,
			)
		elif region_name == "lips":
			pts = np.array(
				[
					_pt(0.22, 0.70),
					_pt(0.78, 0.70),
					_pt(0.78, 0.92),
					_pt(0.22, 0.92),
				],
				dtype=np.int32,
			)
		else:
			return result
		
		# 繪製黃色高亮框
		cv2.polylines(result, [pts], True, (0, 255, 255), 3)
		cv2.putText(
			result,
			f"Focus: {label}",
			(pts[0][0] + 5, pts[0][1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(0, 255, 255),
			2,
		)
	
	return result


def detect_makeup_regions(image_np: np.ndarray, landmarks: List) -> Dict[str, bool]:
	"""使用 landmarks 偵測化妝區域變化"""
	h, w = image_np.shape[:2]
	regions = {
		"eyes": [33, 46, 63, 105, 107, 133],
		"cheeks": [234, 454, 266, 486],
		"lips": [61, 291, 0, 17],
		"forehead": [10, 151, 108, 337],
	}

	detected = {}
	for region_name, point_indices in regions.items():
		if not point_indices or not landmarks:
			detected[region_name] = False
			continue

		region_pixels = []
		for idx in point_indices:
			if idx < len(landmarks):
				x = int(landmarks[idx].x * w)
				y = int(landmarks[idx].y * h)
				x = max(0, min(x, w - 1))
				y = max(0, min(y, h - 1))
				region_pixels.append(image_np[y, x])

		if region_pixels:
			region_array = np.array(region_pixels)
			color_variance = np.var(region_array)
			detected[region_name] = color_variance > 500
		else:
			detected[region_name] = False

	return detected


def detect_makeup_regions_by_bbox(image_np: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, bool]:
	"""當 landmarks 不穩定時，使用臉框估算各區域化妝變化。"""
	if cv2 is None:
		return {"forehead": False, "eyes": False, "cheeks": False, "lips": False}

	x, y, w, h = face_bbox
	img_h, img_w = image_np.shape[:2]
	x1 = max(0, x)
	y1 = max(0, y)
	x2 = min(img_w, x + w)
	y2 = min(img_h, y + h)
	if x2 <= x1 or y2 <= y1:
		return {"forehead": False, "eyes": False, "cheeks": False, "lips": False}

	face = image_np[y1:y2, x1:x2]
	fh, fw = face.shape[:2]
	if fh < 20 or fw < 20:
		return {"forehead": False, "eyes": False, "cheeks": False, "lips": False}

	# 臉框內相對區域
	forehead = face[int(0.06 * fh):int(0.25 * fh), int(0.25 * fw):int(0.75 * fw)]
	eyes = face[int(0.25 * fh):int(0.48 * fh), int(0.15 * fw):int(0.85 * fw)]
	cheeks = face[int(0.46 * fh):int(0.72 * fh), int(0.12 * fw):int(0.88 * fw)]
	lips = face[int(0.62 * fh):int(0.88 * fh), int(0.28 * fw):int(0.72 * fw)]

	def region_score(region: np.ndarray) -> float:
		if region.size == 0:
			return 0.0
		hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
		sat_mean = float(np.mean(hsv[:, :, 1]))
		value_mean = float(np.mean(hsv[:, :, 2]))
		contrast = float(np.std(region))
		# 高飽和度 + 高對比度 = 化妝區域
		return sat_mean * 0.5 + contrast * 0.4 + value_mean * 0.1

	return {
		"forehead": region_score(forehead) > 85,  # 提高閾值
		"eyes": region_score(eyes) > 95,  # 提高閾值
		"cheeks": region_score(cheeks) > 88,  # 提高閾值
		"lips": region_score(lips) > 105,  # 提高閾值
	}


def analyze_makeup_progress(
	current_image: np.ndarray,
	previous_image: Optional[np.ndarray],
	current_step: int,
	landmarks: Optional[List],
	face_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> MakeupProgress:
	"""分析用户化妝進度並檢測錯誤"""
	issues = []

	if landmarks is None and face_bbox is None:
		issues.append("⚠️ 無法偵測到清晰人臉，請穩定鏡頭。")
		return MakeupProgress(
			step_num=current_step,
			foundation_applied=False,
			eyes_started=False,
			cheeks_applied=False,
			lips_applied=False,
			issues=issues,
		)

	if landmarks is not None:
		makeup_regions = detect_makeup_regions(current_image, landmarks)
	else:
		makeup_regions = detect_makeup_regions_by_bbox(current_image, face_bbox)
		issues.append("ℹ️ 目前使用臉框估算進度（關鍵點未穩定），建議保持正面與固定距離以提升精準度。")
	
	# 根據當前步驟檢查進度
	foundation_ok = makeup_regions.get("forehead", False)
	eyes_ok = makeup_regions.get("eyes", False)
	cheeks_ok = makeup_regions.get("cheeks", False)
	lips_ok = makeup_regions.get("lips", False)
	
	# 邏輯檢查
	if current_step == 2:  # Step 2: Foundation
		if not foundation_ok:
			issues.append("💡 提示：請先確保粉底均勻覆蓋，特別係額頭、鼻翼同兩頰。")
		else:
			issues.append("✅ 粉底涂得好！現在開始眼妝。")
	
	elif current_step == 3:  # Step 3: Eyes
		if not eyes_ok:
			issues.append("👁️ 提示：眼影未偵測到，請確保眼影已均勻暈開。")
		else:
			issues.append("✅ 眼妝漂亮！繼續進行。")
	
	elif current_step == 4:  # Step 4: Contour & Blush
		if not cheeks_ok:
			issues.append("🎨 提示：胭脂或修容未充分涂開，請按圓形暈開。")
		else:
			issues.append("✅ 面部立體感出現左！")
	
	elif current_step == 5:  # Step 5: Lips
		if not lips_ok:
			issues.append("💋 提示：唇妝未偵測到，請涂上唇妝。")
		else:
			issues.append("✅ 相得益彰！")
	
	# 檢查光線問題
	h, w = current_image.shape[:2]
	brightness = np.mean(current_image)
	if brightness < 80:
		issues.insert(0, "⚠️ 現場光線不足，難以準確評估，請增加光線。")
	elif brightness > 220:
		issues.insert(0, "⚠️ 光線過亮，容易反光，建議調節光線。")
	
	return MakeupProgress(
		step_num=current_step,
		foundation_applied=foundation_ok,
		eyes_started=eyes_ok,
		cheeks_applied=cheeks_ok,
		lips_applied=lips_ok,
		issues=issues
	)


def _ensure_face_landmarker_model() -> Optional[Path]:
	model_dir = Path(tempfile.gettempdir()) / "ai_makeup_platform_models"
	model_path = model_dir / "face_landmarker.task"
	if model_path.exists() and model_path.stat().st_size > 0:
		return model_path

	model_dir.mkdir(parents=True, exist_ok=True)
	url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
	try:
		urllib.request.urlretrieve(url, str(model_path))
		if model_path.exists() and model_path.stat().st_size > 0:
			return model_path
	except Exception:
		return None
	return None


def _classify_face_shape_from_landmarks(landmarks: List, w: int, h: int) -> Tuple[str, List[str]]:
	def dist(i: int, j: int) -> float:
		xi, yi = landmarks[i].x * w, landmarks[i].y * h
		xj, yj = landmarks[j].x * w, landmarks[j].y * h
		return float(np.hypot(xi - xj, yi - yj))

	forehead_w = dist(54, 284)
	cheek_w = dist(234, 454)
	jaw_w = dist(172, 397)
	face_h = dist(10, 152)

	width_height_ratio = cheek_w / max(face_h, 1e-6)
	forehead_jaw_ratio = forehead_w / max(jaw_w, 1e-6)
	jaw_cheek_ratio = jaw_w / max(cheek_w, 1e-6)

	notes = [
		f"面寬/面長比值: {width_height_ratio:.2f}",
		f"額頭/下顎比值: {forehead_jaw_ratio:.2f}",
	]

	if width_height_ratio < 0.72:
		return "long", notes
	if forehead_jaw_ratio > 1.12:
		return "heart", notes
	if 0.88 <= width_height_ratio <= 1.05 and jaw_cheek_ratio > 0.88:
		return "square", notes
	if width_height_ratio > 0.88 and forehead_jaw_ratio < 1.08:
		return "round", notes
	return "oval", notes


def classify_face_shape_with_mediapipe_tasks(image_np: np.ndarray) -> Tuple[str, List[str]]:
	if not HAS_MEDIAPIPE_TASKS or MP_TASKS_VISION is None or mp is None:
		return "unknown", ["目前環境未啟用 MediaPipe Tasks。"]

	model_path = _ensure_face_landmarker_model()
	if model_path is None:
		return "unknown", ["無法下載 face_landmarker 模型，已自動改用其他分析方式。"]

	try:
		base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
		options = MP_TASKS_VISION.FaceLandmarkerOptions(
			base_options=base_options,
			num_faces=1,
			output_face_blendshapes=False,
			output_facial_transformation_matrixes=False,
		)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
		with MP_TASKS_VISION.FaceLandmarker.create_from_options(options) as detector:
			result = detector.detect(mp_image)
	except Exception as exc:
		return "unknown", [f"Tasks 分析未成功：{type(exc).__name__}，已自動切換。"]

	if not result.face_landmarks:
		return "unknown", ["未偵測到清晰人面，請嘗試正面自拍。"]

	h, w = image_np.shape[:2]
	landmarks = result.face_landmarks[0]
	face_shape, notes = _classify_face_shape_from_landmarks(landmarks, w, h)
	notes.insert(0, "已使用 MediaPipe Tasks 進行分析。")
	return face_shape, notes


def classify_face_shape_with_mediapipe(image_np: np.ndarray) -> Tuple[str, List[str]]:
	if MP_SOLUTIONS_FACE_MESH is None:
		return "unknown", ["目前環境未提供 FaceMesh API，已改用其他分析方式。"]

	rgb = image_np
	h, w = rgb.shape[:2]
	with MP_SOLUTIONS_FACE_MESH.FaceMesh(
		static_image_mode=True,
		max_num_faces=1,
		refine_landmarks=True,
		min_detection_confidence=0.5,
	) as face_mesh:
		result = face_mesh.process(rgb)

	if not result.multi_face_landmarks:
		return "unknown", ["未偵測到清晰人面，請嘗試正面自拍。"]

	landmarks = result.multi_face_landmarks[0].landmark
	face_shape, notes = _classify_face_shape_from_landmarks(landmarks, w, h)
	notes.insert(0, "已使用 FaceMesh 進行分析。")
	return face_shape, notes


def classify_face_shape_with_enhanced_landmarks(image_np: np.ndarray) -> Tuple[str, List[str]]:
	"""使用強化版 landmarks 擷取流程（含裁切/重試）估算面形。"""
	landmarks = get_landmarks_from_image(image_np)
	if not landmarks:
		return "unknown", ["強化 landmarks 流程未取得關鍵點。"]

	h, w = image_np.shape[:2]
	if len(landmarks) < 455:
		return "unknown", ["關鍵點數量不足，無法穩定判斷面形。"]

	face_shape, notes = _classify_face_shape_from_landmarks(landmarks, w, h)
	notes.insert(0, "已使用強化 landmarks 流程進行分析。")
	return face_shape, notes


def classify_face_shape_with_bbox(image_np: np.ndarray) -> Tuple[str, List[str]]:
	"""最後降階：用臉框比例做粗略面形估算，避免長期 unknown。"""
	bbox = detect_face_bbox(image_np)
	if not bbox:
		return "unknown", ["未偵測到臉框，請保持正面與充足光線。"]

	_, _, bw, bh = bbox
	ratio = bw / max(bh, 1)
	notes = [f"臉框寬高比: {ratio:.2f}", "已使用臉框比例進行降階估算。"]

	if ratio >= 0.95:
		return "round", notes
	if ratio <= 0.72:
		return "long", notes
	if 0.72 < ratio <= 0.82:
		return "oval", notes
	if 0.82 < ratio <= 0.90:
		return "square", notes
	return "oval", notes


def analyze_face(image: Image.Image) -> AnalysisResult:
	image_np = np.array(image.convert("RGB"))
	undertone = estimate_undertone(image_np)
	notes: List[str] = []
	face_shape = "unknown"

	if HAS_MEDIAPIPE:
		face_shape, notes = classify_face_shape_with_mediapipe_tasks(image_np)
		if face_shape == "unknown":
			fallback_shape, fallback_notes = classify_face_shape_with_mediapipe(image_np)
			if fallback_shape != "unknown":
				face_shape, notes = fallback_shape, notes + fallback_notes
			else:
				notes = notes + fallback_notes

		if face_shape == "unknown":
			enhanced_shape, enhanced_notes = classify_face_shape_with_enhanced_landmarks(image_np)
			if enhanced_shape != "unknown":
				face_shape, notes = enhanced_shape, notes + enhanced_notes
			else:
				notes = notes + enhanced_notes

		if face_shape == "unknown":
			bbox_shape, bbox_notes = classify_face_shape_with_bbox(image_np)
			if bbox_shape != "unknown":
				face_shape, notes = bbox_shape, notes + bbox_notes
			else:
				notes = notes + bbox_notes
	else:
		h, w = image_np.shape[:2]
		ratio = w / max(h, 1)
		if ratio > 0.85:
			face_shape = "round"
		elif ratio < 0.68:
			face_shape = "long"
		else:
			face_shape = "oval"
		notes = [
			"未安裝 mediapipe，使用基礎比例估算。",
			"如要更準確面形分析，請安裝 mediapipe 與 opencv-python。",
		]

	if face_shape == "unknown":
		face_shape = "oval"
		notes.append("未能穩定判讀，已先以 oval 作保守預設，建議重新拍清晰正面照。")

	return AnalysisResult(face_shape=face_shape, undertone=undertone, notes=notes)


def recommend_looks(
	face_shape: str,
	undertone: str,
	preferred_style: str = "日常自然",
	profile: Optional[Dict[str, object]] = None,
) -> List[Dict[str, List[str]]]:
	preferred_style = normalize_style_name(preferred_style)
	selected = []
	for key in STYLE_LOOK_ORDER.get(preferred_style, STYLE_LOOK_ORDER["日常自然"]):
		item = LOOK_LIBRARY[key]
		products = item["products"].get(undertone, item["products"]["neutral"])
		tools = item["tools"]

		if profile:
			skin_concern = str(profile.get("skin_concern", "一般"))
			if skin_concern == "偏乾":
				products = [p.replace("啞光", "保濕") if "底妝" in p else p for p in products]
			if skin_concern == "偏油":
				products = [p.replace("水光", "控油") if "底妝" in p else p for p in products]

		if face_shape == "round":
			shape_tip = "修容集中喺顴骨下方同下顎邊，眉峰稍提高。"
		elif face_shape == "square":
			shape_tip = "眼妝同胭脂做柔和圓形暈染，減少直角感。"
		elif face_shape == "heart":
			shape_tip = "胭脂重心放中下段，唇妝可略加存在感。"
		elif face_shape == "long":
			shape_tip = "胭脂橫向掃開，眉型可偏平衡向。"
		else:
			shape_tip = "保持整體平衡，按場合調整重點。"

		selected.append(
			{
				"look": item["name"],
				"style": preferred_style,
				"products": products,
				"tools": tools,
				"tip": shape_tip,
			}
		)
	return selected


def build_live_plan_from_profile(profile: Dict[str, object], analysis: AnalysisResult, inventory: List[str], base_image_np: np.ndarray) -> Dict[str, object]:
	"""根據目前偏好重建直播模式推薦、工具與教學。"""
	style = normalize_style_name(str(profile.get("style", "日常自然")))
	depth = get_skin_tone_depth(base_image_np)
	foundation_shades = recommend_foundation_shade(
		analysis.undertone,
		depth,
		style=style,
		finish=str(profile.get("finish", "自然貼膚")),
		skin_concern=str(profile.get("skin_concern", "一般")),
	)
	lip_products = recommend_lip_products(
		analysis.undertone,
		style,
		str(profile.get("occasion", "日常上學/上班")),
	)
	recs = recommend_looks(
		analysis.face_shape,
		analysis.undertone,
		preferred_style=style,
		profile=profile,
	)
	selected = recs[0] if recs else {
		"look": "日常清透妝",
		"products": ["自然色底妝", "杏桃胭脂", "裸粉唇膏", "大地色眼影"],
		"tools": ["美妝蛋", "暈染眼影掃", "腮紅掃", "定妝噴霧"],
		"tip": "保持整體平衡，按場合調整重點。",
	}
	step_tools = recommend_step_tools(selected.get("tools", []), inventory, style=style)
	steps = generate_tutorial(
		selected,
		inventory,
		profile={**profile, "style": style},
		foundation_shades=foundation_shades,
		lip_products=lip_products,
		step_tools=step_tools,
	)
	return {
		"style": style,
		"depth": depth,
		"foundation_shades": foundation_shades,
		"lip_products": lip_products,
		"recommendations": recs,
		"step_tools": step_tools,
		"steps": steps,
	}


def generate_tutorial(
	look: Dict[str, List[str]],
	inventory: List[str],
	profile: Optional[Dict[str, object]] = None,
	foundation_shades: Optional[List[str]] = None,
	lip_products: Optional[List[str]] = None,
	step_tools: Optional[Dict[int, List[str]]] = None,
) -> List[str]:
	inv_join = "、".join(inventory) if inventory else "你未加入現有產品，先用推薦產品示範"
	style = str((profile or {}).get("style", "日常自然"))
	finish = str((profile or {}).get("finish", "自然貼膚"))
	skin_concern = str((profile or {}).get("skin_concern", "一般"))
	occasion = str((profile or {}).get("occasion", "日常上學/上班"))
	skill_level = str((profile or {}).get("skill_level", "新手"))
	foundation_note = ""
	if foundation_shades:
		foundation_note = f"建議底妝色號：{' / '.join(foundation_shades)}。"
	lip_note = ""
	if lip_products:
		lip_note = f"建議唇妝：{' / '.join(lip_products)}。"

	def _tool_text(step_idx: int) -> str:
		if not step_tools:
			return ""
		tools = step_tools.get(step_idx, [])
		return f"工具：{'、'.join(tools)}。" if tools else ""

	prep_tip = "先清潔同保濕，再上妝前乳。"
	if skin_concern == "偏乾":
		prep_tip = "先清潔，再加強保濕 2 層，最後上保濕型妝前乳。"
	elif skin_concern == "偏油":
		prep_tip = "先清潔，再薄上控油妝前乳，T 字位用量可稍多。"

	eye_step = f"眉眼：用 {look['products'][-1]} 打底，再加內眼線提升精神。"
	if style == "日系清透":
		eye_step = f"眉眼：用 {look['products'][-1]} 薄掃上眼皮前半段，眼線只填睫毛根部，保持日系柔和感。"
	elif style == "歐美立體":
		eye_step = f"眉眼：用 {look['products'][-1]} 加深眼尾三角位，再加微上揚眼線，提升立體感。"

	lip_step = f"唇妝：選擇 {look['products'][2]}，中間加深、邊位暈開。"
	if style == "日系清透":
		lip_step = f"唇妝：選擇 {look['products'][2]}，先薄塗全唇，再於唇中心疊一層做自然漸層。"

	if skill_level == "新手":
		foundation_extra = "每次只取半粒黃豆大小，分 2-3 次上臉。"
	elif skill_level == "進階":
		foundation_extra = "可先局部遮瑕再全臉薄鋪，提升精緻度。"
	else:
		foundation_extra = "先薄鋪一層，再按需要疊加局部遮瑕。"

	steps = [
		f"{prep_tip}{_tool_text(1)}",
		f"底妝：優先使用你現有產品（{inv_join}），妝感目標：{finish}。薄薄分層拍開。{foundation_note}{foundation_extra}{_tool_text(2)}",
		f"{eye_step}{_tool_text(3)}",
		f"面部立體：按面形建議使用修容/胭脂；小提示：{look['tip']} {_tool_text(4)}",
		f"{lip_step}{lip_note}{_tool_text(5)}",
		f"最後定妝，檢查鼻翼、嘴角同眼下有無積線。場合：{occasion}，可按需要加強持妝。{_tool_text(6)}",
	]
	return steps


def make_voice_audio(text: str, voice_id: str = "zh-HK-HiuGaaiNeural") -> Optional[str]:
	# Cantonese only: never fallback to Mandarin.
	if HAS_EDGE_TTS:
		with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
			try:
				asyncio.run(edge_tts.Communicate(text, voice_id).save(f.name))
				return f.name
			except Exception:
				pass

	if HAS_GTTS and "yue" in tts_langs():
		with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
			try:
				gTTS(text=text, lang="yue").save(f.name)
				return f.name
			except Exception:
				return None

	return None


def answer_question(user_q: str, analysis: Optional[AnalysisResult], inventory: List[str]) -> str:
	raw_q = str(user_q or "").strip()
	if not raw_q:
		return "我而家會用專業化妝師教新手方式答你。你可以直接問：『我眼下卡粉點算？』、『我粉底唔貼點搞？』、『我下一步做咩？』"

	q = raw_q.lower()

	def _has_any(text: str, keys: List[str]) -> bool:
		return any(k in text for k in keys)

	def _coach_reply(topic: str, reason: str, steps: List[str], checkpoint: str, extra: str = "") -> str:
		lines = [
			f"【專業化妝師｜新手教學】{topic}",
			f"先講重點：{reason}",
			"你而家照住做（簡單 3 步）：",
		]
		for idx, step in enumerate(steps, start=1):
			lines.append(f"{idx}. {step}")
		lines.append(f"完成後檢查：{checkpoint}")
		if extra:
			lines.append(extra)
		return "\n".join(lines)

	profile = st.session_state.get("user_profile", {})
	foundation_shades = st.session_state.get("foundation_shade") or []
	lip_products = st.session_state.get("photo_lip_products") or st.session_state.get("live_lip_products") or []
	depth = st.session_state.get("photo_skin_depth") or st.session_state.get("skin_depth")

	if not foundation_shades and analysis is not None and depth:
		foundation_shades = recommend_foundation_shade(
			analysis.undertone,
			depth,
			style=str(profile.get("style", "日常自然")),
			finish=str(profile.get("finish", "自然貼膚")),
			skin_concern=str(profile.get("skin_concern", "一般")),
		)

	if not lip_products and analysis is not None:
		lip_products = recommend_lip_products(
			analysis.undertone,
			str(profile.get("style", "日常自然")),
			str(profile.get("occasion", "日常上學/上班")),
		)

	if _has_any(raw_q, ["沒什麼變化", "没什么变化", "沒有變化", "冇變化", "無變化", "唔明顯", "不明显", "看不出變化", "看不出变化"]):
		if analysis is None:
			return "你呢條問題要跟分析結果先可以答得準。我而家未有你嘅面形/底調資料，請先做一次相片分析，我再按結果精準提醒你。"

		step = st.session_state.get("current_live_step")
		steps = st.session_state.get("live_steps")
		step_focus_map = {
			1: "第 1 步（妝前）：變化唔會好大，重點係令皮膚狀態穩定、後續更貼妝。",
			2: "第 2 步（底妝）：要見到膚色更均勻、鼻翼泛紅減少，唔係追求一次遮到全無瑕。",
			3: "第 3 步（眼妝）：要見到睫毛根部更實淨、眼尾有方向感。",
			4: "第 4 步（修容/胭脂）：要見到面中至顴骨有立體過渡，而唔係一條深色界線。",
			5: "第 5 步（唇妝）：要見到唇中心更飽和、邊位更乾淨。",
			6: "第 6 步（定妝）：變化係妝面更穩定耐看，而唔係顏色突然變濃。",
		}

		step_hint = "如果你喺跟妝模式，我可以按你而家步驟逐步檢查。"
		if isinstance(step, int) and 1 <= step <= 6:
			step_hint = step_focus_map.get(step, step_hint)

		extra_parts: List[str] = []
		if isinstance(step, int) and isinstance(steps, list) and 1 <= step <= len(steps):
			extra_parts.append(f"你而家係第 {step} 步：{steps[step - 1]}")
		extra_parts.append(
			f"你目前分析：{analysis.face_shape} 面形 / {analysis.undertone} 底調。面形提示：{FACE_SHAPE_GUIDE.get(analysis.face_shape, '保持整體平衡')}"
		)
		if foundation_shades:
			extra_parts.append(f"底妝可先用：{' / '.join(foundation_shades)}")

		return _coach_reply(
			"點解睇落『冇乜變化』",
			"最常見原因係：每步上妝量太薄、前後光線角度唔一致、同埋重點區域未集中。",
			[
				"前後比較一定用同一位置、同一光線、同一角度拍照。",
				step_hint,
				"每次只加少量（半粒黃豆份量），集中一區做完再睇，唔好全臉一次加厚。",
			],
			"先檢查鼻翼、眼尾、唇周三個位：只要其中一區變得更乾淨或更立體，就代表有進步。",
			extra="\n".join(extra_parts),
		)

	# 先處理高重疊關鍵字（防曬/出油）嘅精細問題，避免被泛化分支搶先命中。
	if _has_any(raw_q, ["混合肌", "t字出油", "兩頰乾", "外油內乾"]):
		return _coach_reply(
			"混合肌上妝",
			"混合肌要分區處理，唔可以全臉同一套打法。",
			[
				"T 字位用控油妝前，兩頰用保濕妝前。",
				"底妝先上兩頰，再帶到 T 字，避免 T 字過厚。",
				"定妝只重點壓 T 字，兩頰薄掃即可。",
			],
			"半日後兩頰無起皮、T 字無嚴重融妝，就係啱路線。",
		)

	if _has_any(raw_q, ["補防曬", "上妝後防曬", "防曬點補"]):
		return _coach_reply(
			"上妝後補防曬",
			"直接厚塗會花妝，要用『少量多次』方式。",
			[
				"先用紙巾輕印油汗，再補防曬。",
				"用少量點壓式補塗（可分 2 次），避免來回推。",
				"需要時再用氣墊局部補色，最後輕定妝。",
			],
			"補完後妝面無起塊、無明顯色塊就合格。",
		)

	if _has_any(raw_q, ["搓泥", "起屑", "打架", "防曬同底妝唔夾"]):
		return _coach_reply(
			"防曬/妝前搓泥",
			"通常係前一層未成膜就疊下一層，或者一次用量過多。",
			[
				"防曬後等 3-5 分鐘成膜，再上妝前或底妝。",
				"每層改做薄塗，尤其鼻翼與面中位。",
				"底妝手法由『推開』改做『點拍』，減少把前一層推起。",
			],
			"如果不再起屑、妝面能完整貼住皮膚，就處理到。",
		)

	# 先處理「問題症狀」，避免只回產品名而無法解決你當下問題。
	if _has_any(raw_q, ["卡粉", "浮粉", "起皮"]):
		is_under_eye = _has_any(raw_q, ["眼下", "眼底"])
		if is_under_eye:
			return _coach_reply(
				"眼下卡粉",
				"通常係眼下太乾 + 遮瑕一次上太厚。",
				[
					"先用米粒大小眼霜，等 60-90 秒吸收。",
					"遮瑕分 2 次極薄點拍（只拍暗位，不要整塊塗滿）。",
					"最後只用極少量蜜粉定住眼尾和眼下中段。",
				],
				"笑一下睇細紋位：如果無積線、無結塊，就成功。",
			)
		return _coach_reply(
			"底妝卡粉/浮粉",
			"多數係保濕不足、妝前同粉底唔匹配、或者用量太多。",
			[
				"妝前先補薄保濕，再上少量妝前乳（T 字位可少量控油）。",
				"粉底由面中向外薄拍，鼻翼同嘴角用『半泵再分兩次』。",
				"出現乾紋位，用微濕海綿輕拍 6-8 下，不要再疊厚粉。",
			],
			"近鏡睇鼻翼同嘴角：無一撻撻、紋理變平滑即係啱。",
		)

	if _has_any(raw_q, ["脫妝", "唔持妝", "出油", "斑駁"]):
		return _coach_reply(
			"持妝差/容易脫妝",
			"關鍵唔係『越厚越持久』，而係分區處理同定妝次序。",
			[
				"T 字位先薄上控油妝前，兩頰保濕型妝前。",
				"底妝用薄層疊加，每層之間停 20-30 秒再上下一層。",
				"先定 T 字，再定眼下和嘴角，最後全臉輕噴定妝噴霧。",
			],
			"4-6 小時後只補易出油區；補妝前先印油，再少量補粉。",
		)

	if _has_any(raw_q, ["口罩", "沾口罩", "蹭掉", "甩色", "沾粉"]):
		return _coach_reply(
			"口罩脫妝/沾粉",
			"口鼻位置摩擦最多，要先『減量』再『鎖妝』。",
			[
				"口罩接觸區（鼻樑、鼻翼、嘴角）底妝減半用量。",
				"用小粉撲局部壓蜜粉，再薄噴定妝噴霧。",
				"唇妝改用薄塗 + 抿掉一層，降低沾色。",
			],
			"除口罩後，如果底妝只輕微轉移但妝面仍均勻，即係成功。",
		)

	if _has_any(raw_q, ["毛孔", "鼻翼毛孔", "毛孔粗大"]):
		return _coach_reply(
			"毛孔明顯",
			"毛孔位最怕來回推開，會越推越明顯。",
			[
				"妝前喺毛孔位（鼻翼、鼻側、面中）薄上控油打底。",
				"粉底改用『點拍』，不要左右拖拉。",
				"最後用小粉撲輕壓蜜粉，集中 T 字位。",
			],
			"近鏡睇鼻翼：如果毛孔邊緣冇再反光，就做啱。",
		)

	if _has_any(raw_q, ["太厚", "妝感厚", "假面", "面具感", "唔自然"]):
		return _coach_reply(
			"妝感太厚/假面感",
			"厚重感多數來自一次上太多，唔係產品本身問題。",
			[
				"先用微濕海綿全臉輕拍，帶走表面多餘粉量。",
				"需要遮瑕只補局部（黑眼圈、痘印），唔好整臉再蓋一層。",
				"定妝改做分區壓粉，兩頰少粉、T 字位稍多。",
			],
			"自然光下看面頸交界，無粉感堆積就係自然。",
		)

	if _has_any(raw_q, ["痘痘", "痘印", "泛紅", "遮瑕點遮", "遮痘"]):
		return _coach_reply(
			"痘痘/痘印遮瑕",
			"要『薄底妝 + 精準點遮』，先唔會越遮越突。",
			[
				"先全臉薄底妝，唔好先喺痘位重壓粉底。",
				"遮瑕用小掃只點喺痘印中心，邊緣用指腹輕拍開。",
				"最後局部壓粉定住痘位，避免來回磨擦。",
			],
			"側光下如果凸起感無被放大、顏色有被中和，就達標。",
		)

	if _has_any(raw_q, ["黑眼圈", "眼圈", "暗沉"]):
		return _coach_reply(
			"黑眼圈遮瑕",
			"新手常見錯誤係全眼下鋪滿，結果又厚又乾。",
			[
				"先校色：偏青紫先用少量蜜桃色校正。",
				"遮瑕只點最深陰影位，分 2 次薄拍。",
				"最後少量定妝，只壓易積線位置。",
			],
			"正面望鏡：黑眼圈邊界變柔和，但眼下唔厚粉。",
		)

	if _has_any(raw_q, ["眼線暈", "眼線會暈", "熊貓眼", "暈染到下眼皮"]):
		return _coach_reply(
			"眼線暈開（熊貓眼）",
			"眼皮出油 + 線條未鎖定，最容易半日就暈。",
			[
				"眼線前先薄壓眼皮蜜粉，減少油份。",
				"眼線畫完即刻用同色眼影薄壓一層固定。",
				"下眼尾只畫外 1/3，避免全圈包住。",
			],
			"4 小時後眼尾仍清晰、下眼皮無黑印，就做啱。",
		)

	if _has_any(raw_q, ["法令紋", "笑紋", "細紋"]):
		return _coach_reply(
			"法令紋/笑紋積線",
			"通常係該位粉量太多，表情一動就卡住。",
			[
				"紋路位只上薄層底妝，唔好重複疊加。",
				"用微濕海綿把紋路位多餘粉帶走。",
				"蜜粉只定周邊，不要壓滿整條紋。",
			],
			"講笑時紋路唔結塊，就係理想效果。",
		)

	if _has_any(raw_q, ["眉", "眉毛", "畫眉", "眉型"]):
		return _coach_reply(
			"新手畫眉",
			"眉頭太深、眉峰太高係最常見失手位。",
			[
				"先定 3 點：眉頭、眉峰、眉尾，再連線。",
				"顏色由後向前，眉頭一定最淡。",
				"最後用眉刷梳開，令線條變柔和。",
			],
			"左右眉長度接近、眉頭唔一撻深色就成功。",
		)

	if _has_any(raw_q, ["睫毛", "夾睫毛", "睫毛膏", "蟑螂腳", "結塊"]):
		return _coach_reply(
			"睫毛結塊/下塌",
			"重複厚刷會結塊，太慢夾會回彈下塌。",
			[
				"先分段夾：根部 3 秒、中段 3 秒、尾段 3 秒。",
				"睫毛膏只刷 1-2 層，第二層要趁未全乾前完成。",
				"如結塊，用乾淨睫毛梳由根向上梳開。",
			],
			"側面睇睫毛有弧度且一束束分明就啱。",
		)

	if _has_any(raw_q, ["修容", "陰影", "輪廓"]):
		return _coach_reply(
			"修容位置",
			"修容太低或太深，會令面睇落髒同下垂。",
			[
				"由顴骨下方向太陽穴斜掃，顏色先淡後深。",
				"下顎線只薄掃，唔好畫實線。",
				"每上一層都先暈開，再決定要唔要加深。",
			],
			"遠看有立體感，近看見唔到生硬邊界。",
		)

	if _has_any(raw_q, ["胭脂", "腮紅", "blush"]):
		return _coach_reply(
			"胭脂位置",
			"位置比顏色更重要，放錯位會顯面腫或顯老。",
			[
				"微笑找蘋果肌，落點放喺蘋果肌外上方。",
				"向太陽穴輕掃暈開，避免集中一點。",
				"少量多次疊色，避免一次太重。",
			],
			"左右高度一致，面中有氣色但唔一撻紅。",
		)

	if _has_any(raw_q, ["高光", "打亮", "highlight"]):
		return _coach_reply(
			"高光太誇張/顯毛孔",
			"高光落喺毛孔位會放大紋理。",
			[
				"高光主放顴骨上方、鼻樑中段、唇峰。",
				"避開鼻翼旁毛孔粗大區。",
				"用少量輕掃，唔夠先再疊。",
			],
			"轉頭有微光但唔見顆粒感即係啱。",
		)

	if _has_any(raw_q, ["防曬", "隔離", "妝前", "先後"]):
		return _coach_reply(
			"妝前先後次序",
			"次序錯會影響貼妝同持妝。",
			[
				"基本順序：保濕 -> 防曬 -> 妝前 -> 底妝。",
				"防曬後等 3-5 分鐘再上妝前。",
				"每層都薄，避免一開始就厚塗。",
			],
			"上粉底時唔起屑、唔搓泥，就代表次序同份量啱。",
		)


	if _has_any(raw_q, ["卸妝", "落妝", "清潔", "洗面"]):
		return _coach_reply(
			"卸妝流程",
			"卸唔乾淨會令第二日更難上妝，亦易生粒粒。",
			[
				"先用卸妝產品溶解彩妝（眼唇位可濕敷 10-15 秒）。",
				"加水乳化再沖淨，之後再用潔面產品。",
				"最後補保濕，特別係眼下同鼻翼。",
			],
			"化妝棉輕擦無明顯底妝殘留就合格。",
		)

	if _has_any(raw_q, ["趕時間", "10分鐘", "5分鐘", "快速", "返工趕住"]):
		return _coach_reply(
			"快速出門妝（新手版）",
			"時間少就要抓『精神位』：膚色、眉眼、唇色。",
			[
				"1 分鐘妝前 + 局部底妝（黑眼圈、鼻翼、嘴角）。",
				"眉毛先定型，再畫內眼線填睫毛根部。",
				"唇色一上，最後 T 字位快速定妝。",
			],
			"整體看起來有精神、膚色均勻，就已經夠上班上學。",
		)

	if _has_any(raw_q, ["完整流程", "全套流程", "新手流程", "由頭開始", "從零開始"]):
		return _coach_reply(
			"新手完整化妝流程",
			"先穩定底妝，再加五官重點，成功率最高。",
			[
				"保濕 -> 防曬 -> 妝前 -> 底妝（薄拍分層）。",
				"眉眼先做乾淨，再上胭脂/修容調整輪廓。",
				"最後唇妝 + 定妝；中途只修正局部，唔好全臉重來。",
			],
			"按順序做完後，妝感會更乾淨、持妝更穩。",
		)

	if _has_any(raw_q, ["補妝", "中午", "下午", "脫妝後"]):
		return _coach_reply(
			"中途補妝",
			"直接補粉會結塊，先吸油先係關鍵。",
			[
				"先用紙巾或吸油紙輕印出油位。",
				"用少量粉底/氣墊補局部，不要全臉重上。",
				"最後再薄定妝，重點壓 T 字和鼻翼。",
			],
			"補完後近睇無粉痕、遠睇膚色均勻就可以。",
		)

	if _has_any(raw_q, ["工具", "美妝蛋", "化妝掃", "刷具", "清洗"]):
		return _coach_reply(
			"工具使用與清潔",
			"工具太髒會令妝面斑駁同易生暗粒。",
			[
				"粉底類用微濕海綿或乾淨粉底掃，避免混用過舊產品。",
				"眼影掃至少每週清一次，底妝工具 2-3 天清一次。",
				"清潔後完全風乾先收起，避免發霉。",
			],
			"工具毛感柔軟、上妝唔起條紋就代表狀態良好。",
		)

	if _has_any(raw_q, ["色差", "太白", "太灰", "氧化", "暗沉得快"]):
		return _coach_reply(
			"底妝色差/氧化",
			"色差通常來自底調錯配或底妝氧化。",
			[
				"優先跟底調選色，再調明暗，不要只看白不白。",
				"先薄上一層，等 5 分鐘再看有無變暗。",
				"若容易氧化，可在易暗沉區先薄上控油打底。",
			],
			"自然光下看面頸交界，無灰感無假白就正確。",
		)

	if _has_any(raw_q, ["眼影顏色", "唇色點揀", "顏色搭配", "配色", "今日著咩色妝"]):
		if analysis is None:
			return "配色我可以做得好準，但而家未有你嘅底調分析。先做一次相片分析，我再按你底調同風格幫你配眼唇色。"
		style = str(profile.get("style", "日常自然"))
		return _coach_reply(
			"眼唇配色",
			f"你係 {analysis.undertone} 底調，建議按 {style} 走同色系深淺層次。",
			[
				"眼影先用中性色打底，再用同色系深一階加眼尾。",
				"唇色同眼妝保持同冷暖方向，避免一冷一暖打架。",
				"新手先用『一深一淺』兩色就夠，不要一次混太多色。",
			],
			"全臉看起來協調、眼唇唔互相搶戲就成功。",
		)

	if _has_any(raw_q, ["敏感", "過敏", "刺痛", "紅腫"]):
		return "如果你有刺痛、紅腫或過敏反應，請即刻停用產品並先做皮膚測試。我可以提供化妝流程建議，但唔可以替代醫療診斷。"

	if _has_any(raw_q, ["我有咩", "化妝包", "inventory"]) or _has_any(q, ["makeup bag", "inventory"]):
		if inventory:
			return f"你目前化妝包有：{'、'.join(inventory)}"
		return "你而家仲未加入化妝包；可以喺側邊欄『💼 我的化妝包』加入。"

	if _has_any(raw_q, ["面形", "臉型", "底調", "膚色底調"]) or _has_any(q, ["face shape", "undertone"]):
		if analysis is None:
			return "我而家未有你嘅分析數據，所以唔會亂估面形/底調。請先上傳清晰正面相，我先可以準確答你。"
		return _coach_reply(
			"面形 + 底調解讀",
			f"你目前係「{analysis.face_shape} 面形 / {analysis.undertone} 底調」，重點係按比例去平衡。",
			[
				"底妝色號先跟底調揀，再微調明暗，不要只睇手背試色。",
				f"修飾方向：{FACE_SHAPE_GUIDE.get(analysis.face_shape, '保持整體平衡')}。",
				"眼唇只揀一個做主角，新手先易成功。",
			],
			"完成後自拍：面中乾淨、輪廓自然、唔會灰白反差太大。",
		)

	if _has_any(raw_q, ["粉底", "底妝", "色號"]) or _has_any(q, ["foundation", "shade"]):
		if analysis is None or not depth:
			return "要準確揀粉底色號，我需要你先完成相片分析（面形/底調 + 膚色深度）。目前資料不足，我唔會亂推色號。"
		if foundation_shades:
			return _coach_reply(
				"粉底色號 + 新手上妝",
				f"按你目前資料，最穩陣色號係：{' / '.join(foundation_shades)}。",
				[
					"先用第一個色號做主色，全臉薄拍一層。",
					"若頸部略深，可用第二個色號只補輪廓外圈。",
					"鼻翼、嘴角、眼下只用海綿餘粉帶過，避免厚重。",
				],
				"落自然光睇下巴線：面頸無分界線就係正確色號。",
			)
		return "我有底調資料，但暫時未生成到色號名單。你可以再拍一張光線均勻嘅正面照，我再幫你重算。"

	if _has_any(raw_q, ["唇", "口紅", "唇釉"]) or _has_any(q, ["lip", "lipstick"]):
		if lip_products:
			return _coach_reply(
				"新手唇妝",
				f"按你嘅風格同底調，建議先用：{' / '.join(lip_products)}。",
				[
					"先薄塗全唇一層，唔好一開始就塗好厚。",
					"第二層只放喺唇中間，再向外拍開做漸層。",
					"如果嘴角暗沉，先遮少少再上色會更乾淨。",
				],
				"望鏡講兩句話：邊位唔積線、顏色均勻就完成。",
			)
		return "如果你想要更準確唇妝推薦，我需要先有你嘅底調/風格資料（可先做一次相片分析）。"

	if _has_any(raw_q, ["眼線", "眼影", "眼下卡粉", "眼妝"]) or _has_any(q, ["eyeliner", "eye shadow", "eye makeup"]):
		if _has_any(raw_q, ["眼下卡粉", "卡粉"]):
			return _coach_reply(
				"眼下卡粉",
				"眼下皮膚薄，新手最常見係遮瑕太厚。",
				[
					"眼下先保濕，等待吸收。",
					"遮瑕只點黑眼圈最深位，唔好整片鋪開。",
					"用指腹或細海綿輕拍，不要來回推。",
				],
				"笑紋位無明顯積線，就代表用量剛好。",
			)
		return _coach_reply(
			"新手眼妝",
			"先穩定線條，再做層次，成功率最高。",
			[
				"眼線先填睫毛根部空隙，再拉長眼尾 2-3mm。",
				"眼影先淺後深，每次少量，逐層暈開。",
				"如果畫歪咗，用棉花棒沾少量遮瑕修邊。",
			],
			"左右眼高度接近、眼尾方向一致，就已經好靚。",
		)

	if _has_any(raw_q, ["下一步", "而家做咩", "步驟", "step"]):
		step = st.session_state.get("current_live_step")
		steps = st.session_state.get("live_steps")
		if isinstance(step, int) and isinstance(steps, list) and 1 <= step <= len(steps):
			return _coach_reply(
				f"你而家做第 {step} 步",
				"我會用新手節奏帶你做，唔使急。",
				[
					f"先做呢步核心：{steps[step - 1]}",
					"完成後影一張同角度相片，方便我幫你檢查。",
					"我確認過關先再去下一步，避免越做越亂。",
				],
				"你見到妝面均勻、邊界自然，就可以按下一步。",
			)
		return "我未見到你進入直播步驟資料；你可以去『🎥 跟妝教學模式』，我就可以逐步提示下一步。"

	return (
		"我收到你問題，但想答到最準：請你用呢個格式問我一次\n"
		"部位（底妝/眼妝/修容/唇妝）+ 問題（卡粉/脫妝/太乾/太厚）+ 場景（日常/約會/活動）。\n"
		"例如：『底妝卡粉，鼻翼最明顯，日常返工妝，點救？』"
	)


def init_state() -> None:
	st.session_state.setdefault("inventory", [])
	st.session_state.setdefault("analysis", None)
	st.session_state.setdefault("recommendations", [])
	st.session_state.setdefault("chat_history", [])
	st.session_state.setdefault("current_step", 0)
	st.session_state.setdefault("makeup_progress", None)
	st.session_state.setdefault("camera_on", False)
	st.session_state.setdefault("photo_skin_depth", "medium")
	st.session_state.setdefault("photo_lip_products", [])
	st.session_state.setdefault("photo_step_tools", {})
	st.session_state.setdefault("cantonese_voice", "女聲（粵語）")
	st.session_state.setdefault(
		"user_profile",
		{
			"style": "日系清透",
			"occasion": "日常上學/上班",
			"finish": "自然貼膚",
			"skin_concern": "一般",
			"skill_level": "新手",
		},
	)


def render_makeup_bag_sidebar() -> None:
	"""共用側邊欄：我的化妝包（兩個模式都可用）。"""
	st.sidebar.markdown("---")
	st.sidebar.markdown("### 💼 我的化妝包")
	new_item = st.sidebar.text_input("輸入化妝品或工具", key="global_inventory_input")
	if st.sidebar.button("加入化妝包", key="global_add_item"):
		item = new_item.strip()
		if item:
			if item not in st.session_state.inventory:
				st.session_state.inventory.append(item)
			else:
				st.sidebar.info("已存在於化妝包。")

	if st.session_state.inventory:
		st.sidebar.caption("目前化妝包")
		for idx, item in enumerate(st.session_state.inventory, start=1):
			st.sidebar.write(f"{idx}. {item}")
	else:
		st.sidebar.caption("目前未加入任何產品或工具。")

	if st.sidebar.button("清空化妝包", key="global_clear_inventory"):
		st.session_state.inventory = []
		st.sidebar.success("已清空化妝包。")


def _load_css() -> None:
	css_path = Path(__file__).parent / "style.css"
	if css_path.exists():
		css = css_path.read_text(encoding="utf-8")
		st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _render_hero() -> None:
	img_path = Path(__file__).parent / "assets" / "cover image.png"
	if img_path.exists():
		b64 = base64.b64encode(img_path.read_bytes()).decode()
		st.markdown(
			f"""
			<div class="hero-banner">
				<img src="data:image/png;base64,{b64}" alt="cover" />
				<div class="hero-overlay">
					<h1>MakeMate - 個人化化妝助手</h1>
					<p>上傳相片 → 面形分析 → 推薦妝容／產品工具 → 用你現有化妝品生成教學 → 語音 + 即時問答</p>
				</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
	else:
		st.title("MakeMate - 個人化化妝助手")
		st.caption("上傳相片 -> 面形分析 -> 推薦妝容/產品工具 -> 用你現有化妝品生成教學 -> 語音 + 即時問答")


def get_landmarks_from_image(image_np: np.ndarray) -> Optional[List]:
	"""從圖像中提取人臉關鍵點"""
	if MP_SOLUTIONS_FACE_MESH is None or not HAS_MEDIAPIPE or cv2 is None:
		return None

	if image_np is None or image_np.size == 0:
		return None

	# 修正 dtype / channel，避免 camera_input 回傳格式不一致造成偵測失敗
	if image_np.dtype != np.uint8:
		if image_np.max() <= 1:
			image_np = (image_np * 255).astype(np.uint8)
		else:
			image_np = image_np.astype(np.uint8)

	if len(image_np.shape) == 2:
		image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
	elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
		image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

	# 為提升穩定度，限制過大輸入尺寸
	h, w = image_np.shape[:2]
	if h > 1080 or w > 1080:
		scale = 1080 / max(h, w)
		new_h, new_w = int(h * scale), int(w * scale)
		image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

	rgb = np.ascontiguousarray(image_np)

	# 先用較高精度設定；失敗再以較寬鬆設定重試
	for refine, conf in [(True, 0.35), (False, 0.2)]:
		try:
			with MP_SOLUTIONS_FACE_MESH.FaceMesh(
				static_image_mode=True,
				max_num_faces=1,
				refine_landmarks=refine,
				min_detection_confidence=conf,
			) as face_mesh:
				result = face_mesh.process(rgb)
				if result.multi_face_landmarks:
					return result.multi_face_landmarks[0].landmark
		except Exception:
			continue

	# 降階：先找臉框，再裁切臉部放大重試，最後映射回原圖座標
	face_bbox = detect_face_bbox(rgb)
	if face_bbox is not None:
		x, y, bw, bh = face_bbox
		orig_h, orig_w = rgb.shape[:2]
		pad_x = int(bw * 0.25)
		pad_y = int(bh * 0.30)
		x1 = max(0, x - pad_x)
		y1 = max(0, y - pad_y)
		x2 = min(orig_w, x + bw + pad_x)
		y2 = min(orig_h, y + bh + pad_y)

		if x2 > x1 and y2 > y1:
			crop = rgb[y1:y2, x1:x2]
			crop_h, crop_w = crop.shape[:2]
			if crop_h > 0 and crop_w > 0:
				# FaceMesh 對中等解析度較穩定
				if max(crop_h, crop_w) < 420:
					scale = 420 / max(crop_h, crop_w)
					new_w = max(1, int(crop_w * scale))
					new_h = max(1, int(crop_h * scale))
					crop_for_mesh = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
				else:
					crop_for_mesh = crop

				crop_for_mesh = np.ascontiguousarray(crop_for_mesh)
				for refine, conf in [(True, 0.2), (False, 0.1)]:
					try:
						with MP_SOLUTIONS_FACE_MESH.FaceMesh(
							static_image_mode=True,
							max_num_faces=1,
							refine_landmarks=refine,
							min_detection_confidence=conf,
						) as face_mesh:
							result = face_mesh.process(crop_for_mesh)
							if result.multi_face_landmarks:
								mapped_landmarks: List[NormalizedLandmark] = []
								for lm in result.multi_face_landmarks[0].landmark:
									global_x = (x1 + lm.x * (x2 - x1)) / orig_w
									global_y = (y1 + lm.y * (y2 - y1)) / orig_h
									mapped_landmarks.append(
										NormalizedLandmark(
											x=float(np.clip(global_x, 0.0, 1.0)),
											y=float(np.clip(global_y, 0.0, 1.0)),
											z=float(lm.z),
										)
									)
								return mapped_landmarks
					except Exception:
						continue

	return None


def detect_face_bbox(image_np: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
	"""回傳第一個偵測到的人臉框 (x, y, w, h)。"""
	if cv2 is None or image_np is None or image_np.size == 0:
		return None

	img = image_np
	if img.dtype != np.uint8:
		if img.max() <= 1:
			img = (img * 255).astype(np.uint8)
		else:
			img = img.astype(np.uint8)

	if len(img.shape) == 3 and img.shape[2] == 4:
		img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

	# 優先用 MediaPipe FaceDetection
	if MP_SOLUTIONS_FACE_DETECTION is not None:
		try:
			h, w = img.shape[:2]
			with MP_SOLUTIONS_FACE_DETECTION.FaceDetection(
				model_selection=0,
				min_detection_confidence=0.3,
			) as detector:
				result = detector.process(img)
				if result.detections:
					bbox = result.detections[0].location_data.relative_bounding_box
					x = max(0, int(bbox.xmin * w))
					y = max(0, int(bbox.ymin * h))
					bw = max(1, int(bbox.width * w))
					bh = max(1, int(bbox.height * h))
					return (x, y, bw, bh)
		except Exception:
			pass

	# 再用 OpenCV Haar 作最後 fallback
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
	cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
	face_cascade = cv2.CascadeClassifier(cascade_path)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
	if len(faces) > 0:
		x, y, w, h = faces[0]
		return int(x), int(y), int(w), int(h)

	return None


def draw_face_landmarks(image_np: np.ndarray, landmarks: List) -> np.ndarray:
	"""在圖像上繪製人臉關鍵點和區域"""
	if cv2 is None or landmarks is None:
		return image_np

	h, w = image_np.shape[:2]
	output = image_np.copy()

	# 只畫主要區域，保持畫面清晰
	key_groups = {
		"left_eye": [33, 133, 159, 145],
		"right_eye": [263, 362, 386, 374],
		"lips": [61, 291, 0, 17, 78, 308],
		"cheeks": [234, 454],
	}

	for _, indices in key_groups.items():
		pts = []
		for idx in indices:
			if idx < len(landmarks):
				x = int(landmarks[idx].x * w)
				y = int(landmarks[idx].y * h)
				pts.append((x, y))
				cv2.circle(output, (x, y), 2, (80, 230, 120), -1)
		if len(pts) >= 2:
			for i in range(len(pts) - 1):
				cv2.line(output, pts[i], pts[i + 1], (60, 180, 255), 1)

	return output


def photo_mode() -> None:
	"""📸 妝前分析模式"""
	st.subheader("📸 妝前分析")
	profile = st.session_state.user_profile
	st.caption(
		f"目前偏好：{profile['style']}｜{profile['finish']}｜{profile['occasion']}｜{profile['skin_concern']}"
	)
	uploaded = st.file_uploader("上傳清晰正面自拍", type=["jpg", "jpeg", "png"], key="photo_upload")

	if uploaded:
		image = Image.open(uploaded).convert("RGB")
		st.image(image, caption="已上傳照片", use_container_width=True)

		with st.spinner("分析中..."):
			analysis = analyze_face(image)
			st.session_state.analysis = analysis
			image_np = np.array(image.convert("RGB"))
			st.session_state.photo_skin_depth = get_skin_tone_depth(image_np)
			st.session_state.photo_lip_products = recommend_lip_products(
				analysis.undertone,
				profile["style"],
				profile["occasion"],
			)
			st.session_state.recommendations = recommend_looks(
				analysis.face_shape,
				analysis.undertone,
				preferred_style=profile["style"],
				profile=profile,
			)

		if st.session_state.analysis:
			analysis = st.session_state.analysis
			shape_guide = FACE_SHAPE_GUIDE.get(analysis.face_shape, "保持整體平衡")

			st.markdown('<div class="section-title">✨ 面形分析結果</div>', unsafe_allow_html=True)
			st.write(f"面形：{analysis.face_shape}")
			st.write(f"膚色底調：{analysis.undertone}")
			st.info(f"建議：{shape_guide}")

			if analysis.notes:
				with st.expander("分析備註"):
					for note in analysis.notes:
						st.write(f"- {note}")

			recs = st.session_state.recommendations
			if recs:
				look_names = [f"{idx+1}. {r['look']}" for idx, r in enumerate(recs)]
				choice = st.selectbox(
					"選擇妝容",
					options=list(range(len(recs))),
					format_func=lambda i: look_names[i],
					key="photo_look_choice",
				)
				selected = recs[choice]
				st.session_state.photo_step_tools = recommend_step_tools(
					selected.get("tools", []),
					st.session_state.inventory,
					style=profile.get("style", "日常自然"),
				)

				product_chips = "".join(f'<span class="chip">{p}</span>' for p in selected["products"])
				tool_chips = "".join(f'<span class="chip">{t}</span>' for t in selected["tools"])
				st.markdown(
					f"""
					<div class="look-card">
						<h4>🌸 {selected['look']}</h4>
						<div class="section-label">推薦產品</div>
						<div class="chip-row">{product_chips}</div>
						<div class="section-label">推薦工具</div>
						<div class="chip-row">{tool_chips}</div>
					</div>
					""",
					unsafe_allow_html=True,
				)
				st.info(f"💡 面形化妝提示：{selected['tip']}")

				st.markdown('<div class="section-title">📖 一步步教學 + 語音指導（廣東話）</div>', unsafe_allow_html=True)
				depth = st.session_state.photo_skin_depth
				foundation_shades = recommend_foundation_shade(
					analysis.undertone,
					depth,
					style=profile["style"],
					finish=profile["finish"],
					skin_concern=profile["skin_concern"],
				)
				lip_products = st.session_state.get("photo_lip_products", [])
				if foundation_shades:
					st.success(f"🎯 推薦底妝（精選）: {' / '.join(foundation_shades)}")
				if lip_products:
					st.success(f"💄 推薦唇妝（精選）: {' / '.join(lip_products)}")
				step_tools = st.session_state.get("photo_step_tools", {})
				if step_tools:
					with st.expander("🧰 每步建議工具"):
						for step_no in range(1, 7):
							tools = step_tools.get(step_no, [])
							if tools:
								st.write(f"步驟 {step_no}: {'、'.join(tools)}")
				steps = generate_tutorial(
					selected,
					st.session_state.inventory,
					profile=profile,
					foundation_shades=foundation_shades,
					lip_products=lip_products,
					step_tools=step_tools,
				)
				steps_html = "".join(
					f'<div class="step-item"><span class="step-num">{i}</span>{step}</div>'
					for i, step in enumerate(steps, start=1)
				)
				st.markdown(steps_html, unsafe_allow_html=True)

				voice_text = "。".join(steps)
				voice_map = {
					"女聲（粵語）": "zh-HK-HiuGaaiNeural",
					"男聲（粵語）": "zh-HK-WanLungNeural",
				}
				audio_path = make_voice_audio(voice_text, voice_map.get(st.session_state.get("cantonese_voice", "女聲（粵語）"), "zh-HK-HiuGaaiNeural"))
				if audio_path:
					st.audio(audio_path)
				else:
					st.caption("目前未安裝可用嘅廣東話語音引擎（edge-tts 或支援 yue 嘅 gTTS），暫時只顯示文字教學。")

		st.subheader("💬 即時問答")
		for role, msg in st.session_state.chat_history:
			with st.chat_message(role):
				st.write(msg)

		prompt = st.chat_input("例如：我眼下卡粉點改善？")
		if prompt:
			st.session_state.chat_history.append(("user", prompt))
			reply = answer_question(prompt, st.session_state.analysis, st.session_state.inventory)
			st.session_state.chat_history.append(("assistant", reply))
			st.rerun()


def live_mode() -> None:
	"""🎥 跟妝教學模式 - YouTube 老師教你化妝"""
	st.subheader("🎥 跟妝教學 - 像 YouTuber 一樣邊做邊學！")
	st.write("💡 我會像化妝 YouTuber 一樣，一步步教你，邊教邊監控你的進度！")

	# 初始化階段：先拍一張照進行分析
	if "live_analysis" not in st.session_state:
		st.session_state.live_analysis = None
		st.session_state.live_recommendations = None
		st.session_state.live_steps = None
		st.session_state.current_live_step = 1
		st.session_state.step_photos = {}
		st.session_state.foundation_shade = None
		st.session_state.live_lip_products = None
		st.session_state.live_step_tools = None
		st.session_state.skin_depth = None
		st.session_state.clean_reference_photo = None
		st.session_state.live_profile_signature = None

	# =========== 階段 1: 初始分析 (含品質檢查) ===========
	if st.session_state.live_analysis is None:
		profile = st.session_state.user_profile
		profile["style"] = normalize_style_name(str(profile.get("style", "日常自然")))
		st.caption(
			f"教學設定：{profile['style']}｜{profile['finish']}｜{profile['occasion']}｜{profile['skin_concern']}｜{profile['skill_level']}"
		)
		st.info("📸 **第一步：拍一張清晰的正面自拍**")
		st.write("💡 提示：最好在自然光下，不要化妝，正面對著鏡頭")
		
		init_camera = st.camera_input("拍攝正面自拍（最好不要化妝）", key="live_init_camera")
		
		if init_camera:
			init_img = Image.open(init_camera).convert("RGB")
			init_np = np.array(init_img)
			
			# 檢查臉部品質
			quality = check_face_quality(init_np)
			
			# 顯示品質診斷
			if not quality.is_clear:
				st.warning("⚠️ 臉部掃描質量未達最佳")
				with st.expander("📋 診斷詳情"):
					st.write(f"**亮度評分：** {quality.brightness_level:.0f}/100")
					st.write(f"**臉部大小：** {quality.face_size_ratio:.0f}/100")
					st.write(f"**關鍵點偵測：** {quality.landmarks_confidence:.0f}/100")
					st.write(f"**整體質量：** {quality.quality_score:.0f}/100")
					if quality.issues:
						st.markdown("**需要改善：**")
						for issue in quality.issues:
							st.markdown(f"- {issue}")
				
				if st.button("🔄 重新拍攝"):
					st.rerun()
				
				st.info("💡 如果無法改善，仍可繼續（系統會自動調整，但可能影響精準度）")
				
				if not st.button("⚠️ 繼續（已了解風險）"):
					st.stop()
			else:
				st.success(f"✅ 臉部品質優秀！({quality.quality_score:.0f}/100)")
			
			# 進行分析
			analysis = analyze_face(init_img)
			st.session_state.live_analysis = analysis
			st.session_state.clean_reference_photo = init_np
			
			plan = build_live_plan_from_profile(profile, analysis, st.session_state.inventory, init_np)
			st.session_state.skin_depth = plan["depth"]
			st.session_state.foundation_shade = plan["foundation_shades"]
			st.session_state.live_lip_products = plan["lip_products"]
			st.session_state.live_recommendations = plan["recommendations"]
			st.session_state.live_step_tools = plan["step_tools"]
			st.session_state.live_steps = plan["steps"]
			st.session_state.live_profile_signature = profile_signature(profile)
			
			st.balloons()
			st.success("✅ 分析完成，現在開始學化妝吧！")
			st.rerun()
	
	# =========== 階段 2: 逐步教學 + 實時反饋 ===========
	else:
		profile = st.session_state.user_profile
		profile["style"] = normalize_style_name(str(profile.get("style", "日常自然")))
		current_sig = profile_signature(profile)
		if (
			st.session_state.get("live_analysis") is not None
			and st.session_state.get("clean_reference_photo") is not None
			and current_sig != st.session_state.get("live_profile_signature")
		):
			plan = build_live_plan_from_profile(
				profile,
				st.session_state.live_analysis,
				st.session_state.inventory,
				st.session_state.clean_reference_photo,
			)
			st.session_state.skin_depth = plan["depth"]
			st.session_state.foundation_shade = plan["foundation_shades"]
			st.session_state.live_lip_products = plan["lip_products"]
			st.session_state.live_recommendations = plan["recommendations"]
			st.session_state.live_step_tools = plan["step_tools"]
			st.session_state.live_steps = plan["steps"]
			st.session_state.live_profile_signature = current_sig
			st.session_state.current_live_step = 1
			st.info("✅ 已按新風格重建老師教學與推薦（已由步驟 1 重新開始）。")
			st.rerun()
		if st.session_state.live_analysis and st.session_state.live_analysis.face_shape == "unknown":
			st.warning("⚠️ 面形仍未穩定辨識，建議重新掃描一次。")
			if st.button("🔄 重新掃描面形", key="live_rescan_face_shape"):
				st.session_state.live_analysis = None
				st.session_state.live_recommendations = None
				st.session_state.live_steps = None
				st.session_state.current_live_step = 1
				st.session_state.step_photos = {}
				st.rerun()

		# 顯示分析結果和個性化建議
		col_info1, col_info2, col_info3, col_info4 = st.columns(4)
		with col_info1:
			st.metric("面形", st.session_state.live_analysis.face_shape)
		with col_info2:
			st.metric("膚色", st.session_state.live_analysis.undertone)
		with col_info3:
			if st.session_state.foundation_shade:
				st.metric("推薦底妝", " / ".join(st.session_state.foundation_shade))
		with col_info4:
			if st.session_state.live_lip_products:
				st.metric("推薦唇妝", st.session_state.live_lip_products[0])
		
		# 進度指示器
		progress_steps = st.session_state.live_steps
		current_step = st.session_state.current_live_step
		
		# 進度條
		st.progress((current_step - 1) / len(progress_steps))
		st.write(f"⏳ 進度：第 {current_step} / {len(progress_steps)} 步")
		
		st.divider()
		
		# 左右分割：老師教學 + 相機監控
		col_teach, col_camera = st.columns([1, 2], gap="large")
		
		with col_teach:
			st.markdown("### 👨‍🏫 老師教你")
			st.caption(f"風格：{profile['style']}｜熟練度：{profile['skill_level']}")
			st.markdown(f"#### 📖 步驟 {current_step}")
			
			if current_step <= len(progress_steps):
				step_text = progress_steps[current_step - 1]
				st.markdown(step_text, unsafe_allow_html=True)

				step_tools = st.session_state.get("live_step_tools", {})
				if step_tools and step_tools.get(current_step):
					st.caption(f"🧰 本步建議工具：{'、'.join(step_tools[current_step])}")
				
				# 步驟 2 特別推薦
				if current_step == 2 and st.session_state.foundation_shade:
					st.info(
						f"💡 根據你的膚色，推薦使用：**{' / '.join(st.session_state.foundation_shade)}**\n\n"
						"只保留 1-2 個最適合色號，方便你直接選購與上妝。"
					)
				if current_step == 5 and st.session_state.live_lip_products:
					st.info(f"💄 唇妝建議：**{' / '.join(st.session_state.live_lip_products)}**")
				
				# 語音教學
				voice_map = {
					"女聲（粵語）": "zh-HK-HiuGaaiNeural",
					"男聲（粵語）": "zh-HK-WanLungNeural",
				}
				voice_path = make_voice_audio(step_text, voice_map.get(st.session_state.get("cantonese_voice", "女聲（粵語）"), "zh-HK-HiuGaaiNeural"))
				if voice_path:
					st.audio(voice_path)
			
			st.divider()
			
			# 用戶確認完成
			st.markdown("### ✅ 準備好了？")
			user_ready = st.button(
				f"✅ 我完成了步驟 {current_step}，拍照讓老師檢查",
				key=f"step_{current_step}_ready",
				use_container_width=True,
				type="primary",
			)
		
		with col_camera:
			st.markdown("### 📷 相機監控")
			
			# 顯示步驟指導區域視覺
			st.write(f"準備好時，拍照讓老師檢查你的步驟 {current_step} 做得如何")
			
			# 固定顯示 AR 老師示範（基於你的相片）
			preview_np = st.session_state.step_photos.get(current_step)
			if preview_np is None:
				preview_np = st.session_state.clean_reference_photo

			if preview_np is not None and st.session_state.live_analysis is not None:
				st.markdown("#### 👩‍🏫 AR 老師示範（基於你的相片）")
				st.caption("AR 老師會播放 GIF 動畫示範，你可以跟住同樣方向與力度去做。")
				demo_gif = generate_teacher_demo_video(
					preview_np,
					current_step,
					st.session_state.live_analysis.undertone,
				)
				rendered_demo = False
				if demo_gif and Path(demo_gif).suffix.lower() == ".gif":
					render_gif_animation(demo_gif, caption="AR 老師動畫示範")
					rendered_demo = True

				if not rendered_demo:
					# fallback: 若 GIF 生成失敗，就顯示分鏡圖
					demo_cols = st.columns(3)
					for idx, demo_col in enumerate(demo_cols, start=1):
						with demo_col:
							demo_frame = render_teacher_demo_frame(
								preview_np,
								current_step,
								st.session_state.live_analysis.undertone,
								layer_idx=idx,
							)
							st.image(demo_frame, caption=f"老師示範第 {idx} 段", use_container_width=True)

			if current_step == 2:
				st.info("👀 上面是老師示範；下面拍照比對你是否做到『薄薄分層拍開』。")
			
			check_camera = st.camera_input(
				f"拍照檢查步驟 {current_step}",
				key=f"step_{current_step}_camera",
			)
			
			if check_camera or user_ready:
				if check_camera:
					check_img = Image.open(check_camera).convert("RGB")
					check_np = np.array(check_img)
					st.session_state.step_photos[current_step] = check_np
				else:
					# 如果用戶點了「完成」但沒拍照，使用之前的
					check_np = st.session_state.step_photos.get(current_step)
					if check_np is None:
						st.warning("⚠️ 請先拍照來驗證步驟")
						st.stop()
				
				# 顯示臉部追蹤
				landmarks = get_landmarks_from_image(check_np)
				face_bbox = detect_face_bbox(check_np)
				
				display_frame = check_np.copy()
				
				# 繪製步驟指導區域 (黃色框)
				display_frame = draw_step_guide_overlay(display_frame, current_step, landmarks=landmarks, face_bbox=face_bbox)
				
				# 步驟 2 時顯示虛擬底妝 AR
				if current_step == 2:
					ar_frame = render_teacher_demo_frame(
						check_np,
						current_step,
						st.session_state.live_analysis.undertone,
						layer_idx=3,
					)
					
					# 並排顯示虛擬效果
					col_before, col_after = st.columns(2)
					with col_before:
						st.image(check_np, caption="你目前畫面", use_container_width=True)
					with col_after:
						st.image(ar_frame, caption="老師示範（同一張相）", use_container_width=True)
				else:
					if landmarks is not None:
						display_frame = draw_face_landmarks(display_frame, landmarks)
					if face_bbox is not None:
						x, y, w, h = face_bbox
						cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
					
					st.image(display_frame, caption="👁️ 老師正在檢查你...", use_container_width=True)
				
				# AI 檢查反饋 (驗證步驟)
				st.markdown("### 💬 老師的反饋")
				
				feedback = verify_step_completion(st.session_state.clean_reference_photo, check_np, current_step)
				
				if feedback["completed"]:
					st.success(f"✅ {feedback['feedback']}")
				else:
					st.warning(f"⚠️ {feedback['feedback']}")
				
				# 顯示小貼士
				if feedback["tips"]:
					with st.expander("💡 小貼士"):
						for tip in feedback["tips"]:
							st.write(f"• {tip}")
				
				st.divider()
				
				# 進到下一步 (只有確認完成才能進)
				if feedback["completed"]:
					if current_step < len(progress_steps):
						if st.button(f"👉 繼續步驟 {current_step + 1}", key=f"next_step_{current_step}", use_container_width=True):
							st.session_state.current_live_step += 1
							st.rerun()
					else:
						st.balloons()
						st.success("🎉 完成了！你看起來棒極了！")
						if st.button("🔄 重新開始", key="restart_live", use_container_width=True):
							st.session_state.live_analysis = None
							st.session_state.current_live_step = 1
							st.session_state.step_photos = {}
							st.rerun()
				else:
					st.info("⚠️ 請根據提示修正，然後再拍一次照")
		
		st.divider()
		
		# 顯示所有步驟教學
		with st.expander("📖 查看所有步驟教學"):
			for i, step in enumerate(progress_steps, start=1):
				st.markdown(f"### 步驟 {i}")
				st.markdown(step, unsafe_allow_html=True)


def main() -> None:
	"""主應用 - 用戶選擇模式"""
	st.set_page_config(page_title="MakeMate", page_icon="💄", layout="wide")
	init_state()
	_load_css()
	_render_hero()
	
	# 側邊欄模式選擇
	mode = st.sidebar.radio(
		"🎯 選擇模式",
		["📸 妝前分析模式", "🎥 跟妝教學模式"],
		help="妝前分析：上傳相片做基礎分析與推薦。跟妝教學：相機邊做邊教，實時監控化妝進度。"
	)

	st.sidebar.markdown("---")
	st.sidebar.markdown("### 🧾 你的妝容偏好")
	profile = st.session_state.user_profile
	profile["style"] = st.sidebar.selectbox(
		"想要的風格",
		["日系清透", "韓系水光", "歐美立體", "晚宴 Glam", "日常自然"],
		index=["日系清透", "韓系水光", "歐美立體", "晚宴 Glam", "日常自然"].index(profile.get("style", "日系清透")),
		help="先揀風格，系統會按風格生成推薦和老師教學。",
	)
	profile["occasion"] = st.sidebar.selectbox(
		"使用場合",
		["日常上學/上班", "約會", "拍照/活動", "晚宴"],
		index=["日常上學/上班", "約會", "拍照/活動", "晚宴"].index(profile.get("occasion", "日常上學/上班")),
	)
	profile["finish"] = st.sidebar.selectbox(
		"妝感偏好",
		["自然貼膚", "水光透亮", "柔霧啞光", "高遮瑕"],
		index=["自然貼膚", "水光透亮", "柔霧啞光", "高遮瑕"].index(profile.get("finish", "自然貼膚")),
	)
	profile["skin_concern"] = st.sidebar.selectbox(
		"膚質重點",
		["一般", "偏乾", "偏油", "容易脫妝"],
		index=["一般", "偏乾", "偏油", "容易脫妝"].index(profile.get("skin_concern", "一般")),
	)
	profile["skill_level"] = st.sidebar.selectbox(
		"化妝熟練度",
		["新手", "中階", "進階"],
		index=["新手", "中階", "進階"].index(profile.get("skill_level", "新手")),
	)
	st.session_state.cantonese_voice = st.sidebar.selectbox(
		"老師語音（廣東話）",
		["女聲（粵語）", "男聲（粵語）"],
		index=["女聲（粵語）", "男聲（粵語）"].index(st.session_state.get("cantonese_voice", "女聲（粵語）")),
	)
	st.session_state.user_profile = profile
	render_makeup_bag_sidebar()
	
	st.sidebar.markdown("---")
	st.sidebar.caption("💡 提示：妝前分析用來建立基礎推薦，跟妝教學用來邊做邊學。")
	
	if mode == "📸 妝前分析模式":
		photo_mode()
	else:
		live_mode()


if __name__ == "__main__":
	main()
