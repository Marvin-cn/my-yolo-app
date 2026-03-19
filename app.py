import streamlit as st
import os

# --- 1. 深度环境兼容性处理 (必须放在最前面) ---
os.environ["QT_QPA_PLATFORM"] = "offscreen"
try:
    import cv2
except ImportError:
    # 如果发生意外冲突，强制引导系统识别 headless 版本
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])

from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter

# --- 2. 你的果蔬中英文对照表 ---
FRUIT_VEG_DICT = {
    "apple": "苹果", "banana": "香蕉", "orange": "橘子", "strawberry": "草莓",
    "grape": "葡萄", "watermelon": "西瓜", "pear": "梨", "peach": "桃子",
    "lemon": "柠檬", "tomato": "西红柿/番茄", "potato": "土豆", "carrot": "胡萝卜",
    "onion": "洋葱", "broccoli": "西兰花", "cabbage": "卷心菜", "cucumber": "黄瓜",
    "pepper": "辣椒", "eggplant": "茄子", "mushroom": "蘑菇", "mango": "芒果",
    "pineapple": "菠萝", "cherry": "樱桃"
}

# 页面基础配置
st.set_page_config(page_title="AI 智能识别助手", layout="wide")
st.title("🔍 AI 图像识别系统")

# --- 3. 方案一：带状态显示的缓存加载 ---
@st.cache_resource
def load_model():
    # 确保模型文件名与仓库内完全一致
    return YOLO("yolo_fruits_and_vegetables_v3.pt")

with st.status("🛠️ 正在初始化 AI 引擎 (137MB 模型装载中)...", expanded=True) as status:
    model = load_model()
    status.update(label="✅ 系统已就绪，请上传图片进行识别！", state="complete", expanded=False)

# --- 4. 侧边栏与图片上传 ---
st.sidebar.header("识别设置")
conf_threshold = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.25)
uploaded_file = st.file_uploader("点击上传或拍照识别...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # 针对移动端优化：展示识别中的状态
    with st.spinner('🚀 AI 正在深度分析中，请耐心等待几秒...'):
        # 自动缩放大图，防止内存溢出导致手机崩溃
        image.thumbnail((1024, 1024))
        img_array = np.array(image)
        
        # 执行识别
        results = model.predict(source=img_array, conf=conf_threshold, iou=0.7)
        res_plotted = results[0].plot()
        
        # 结果统计
        names = model.names
        detected_indices = results[0].boxes.cls.cpu().numpy().astype(int)
        detected_names = [names[i] for i in detected_indices]
        counts = Counter(detected_names)

    # 展示 UI 布局
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="原始图片", use_container_width=True)
    with col2:
        st.image(res_plotted, caption="识别结果图", use_container_width=True)

    st.divider()
    st.markdown("### 📝 识别报告")
    
    if len(detected_names) > 0:
        detail_items = []
        for obj, count in counts.items():
            cn_name = FRUIT_VEG_DICT.get(obj.lower(), obj)
            detail_items.append(f"{cn_name}: {count}个")
        
        st.success(f"**成功识别 {len(detected_names)} 个目标：** " + "、".join(detail_items))
    else:
        st.warning("未能识别到目标，请尝试降低左侧侧边栏的置信度阈值。")