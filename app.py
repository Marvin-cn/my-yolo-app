import streamlit as st
import os

# 强制关闭 OpenCV 的图形界面检测
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter

# 果蔬翻译字典
FRUIT_VEG_DICT = {
    "apple": "苹果", "banana": "香蕉", "orange": "橘子", "strawberry": "草莓",
    "grape": "葡萄", "watermelon": "西瓜", "pear": "梨", "peach": "桃子",
    "lemon": "柠檬", "tomato": "西红柿", "potato": "土豆", "carrot": "胡萝卜",
    "onion": "洋葱", "broccoli": "西兰花", "cabbage": "卷心菜", "cucumber": "黄瓜",
    "pepper": "辣椒", "eggplant": "茄子", "mushroom": "蘑菇", "mango": "芒果",
    "pineapple": "菠萝", "cherry": "樱桃"
}

st.set_page_config(page_title="AI 果蔬识别助手", layout="wide")
st.title("🔍 AI 图像识别系统")

# 缓存模型加载
@st.cache_resource
def load_model():
    return YOLO("yolo_fruits_and_vegetables_v3.pt")

try:
    with st.status("🛠️ 正在装载 AI 引擎...", expanded=False) as status:
        model = load_model()
        status.update(label="✅ 系统就绪", state="complete")
except Exception as e:
    st.error(f"模型加载失败，请检查文件名。错误信息: {e}")

# 侧边栏
st.sidebar.header("识别设置")
conf_threshold = st.sidebar.slider("置信度", 0.1, 1.0, 0.25)
uploaded_file = st.file_uploader("上传果蔬照片...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    with st.spinner('🚀 AI 正在识别中...'):
        # 针对手机端进行优化：限制处理尺寸
        image.thumbnail((1024, 1024))
        img_array = np.array(image)
        
        # 运行识别
        results = model.predict(source=img_array, conf=conf_threshold)
        res_plotted = results[0].plot()
        
        # 统计结果
        names = model.names
        detected_indices = results[0].boxes.cls.cpu().numpy().astype(int)
        detected_names = [names[i] for i in detected_indices]
        counts = Counter(detected_names)

    # 结果展示
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="原始图片", use_container_width=True)
    with col2:
        st.image(res_plotted, caption="识别结果图", use_container_width=True)

    st.divider()
    if len(detected_names) > 0:
        res_list = [f"{FRUIT_VEG_DICT.get(name.lower(), name)}: {count}个" for name, count in counts.items()]
        st.success(f"**识别报告：** " + "、".join(res_list))
    else:
        st.warning("未发现目标，请尝试调低左侧的置信度。")