import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter

# --- 1. 你的果蔬中英文对照表（原样保留） ---
FRUIT_VEG_DICT = {
    "apple": "苹果",
    "banana": "香蕉",
    "orange": "橘子",
    "strawberry": "草莓",
    "grape": "葡萄",
    "watermelon": "西瓜",
    "pear": "梨",
    "peach": "桃子",
    "lemon": "柠檬",
    "tomato": "西红柿/番茄",
    "potato": "土豆",
    "carrot": "胡萝卜",
    "onion": "洋葱",
    "broccoli": "西兰花",
    "cabbage": "卷心菜",
    "cucumber": "黄瓜",
    "pepper": "辣椒",
    "eggplant": "茄子",
    "mushroom": "蘑菇",
    "mango": "芒果",
    "pineapple": "菠萝",
    "cherry": "樱桃"
}

# 页面配置
st.set_page_config(page_title="AI 智能识别助手", layout="wide")
st.title("🔍 AI 图像识别系统")

# --- 2. 方案一：带状态显示的缓存模型加载 ---
# 这样在手机端刚进去时，你会看到一个明确的进度提示
@st.cache_resource
def load_model():
    return YOLO("yolo_fruits_and_vegetables_v3.pt")

with st.status("🛠️ 正在初始化 AI 引擎 (137MB 模型装载中)...", expanded=True) as status:
    model = load_model()
    status.update(label="✅ 系统已就绪，请上传图片进行识别！", state="complete", expanded=False)

# --- 3. 侧边栏：设置识别标准 ---
st.sidebar.header("识别设置")
conf_threshold = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.25)

uploaded_file = st.file_uploader("请上传图片...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 立即展示用户上传的图片
    image = Image.open(uploaded_file)
    
    # --- 4. 方案三：识别全过程的动态反馈 ---
    with st.spinner('🚀 AI 正在深度分析中，请耐心等待几秒...'):
        # 针对手机端进行图片缩放处理（不影响精度，但提升速度并防止崩溃）
        image.thumbnail((800, 800))
        img_array = np.array(image)
        
        # 执行识别（使用你代码里的参数）
        results = model.predict(source=img_array, conf=conf_threshold, iou=0.7)
        
        # 获取识别结果图
        res_plotted = results[0].plot()
        
        # 统计逻辑（保持你的 Counter 逻辑）
        names = model.names
        detected_indices = results[0].boxes.cls.cpu().numpy().astype(int)
        detected_names = [names[i] for i in detected_indices]
        counts = Counter(detected_names)

    # --- 5. 展示结果（保持你的 UI 布局） ---
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="原始图片", use_container_width=True)
    with col2:
        st.image(res_plotted, caption="识别结果图", use_container_width=True)

    st.divider()
    st.markdown("### 📝 识别报告")
    
    if len(detected_names) > 0:
        # 构建翻译后的汇总文本
        detail_items = []
        for obj, count in counts.items():
            cn_name = FRUIT_VEG_DICT.get(obj.lower(), obj)
            detail_items.append(f"{cn_name}: {count}个")
        
        summary_text = f"**成功识别 {len(detected_names)} 个目标：** " + "、".join(detail_items)
        st.success(summary_text)
    else:
        st.warning("未能识别到目标，请尝试在左侧侧边栏‘降低’置信度。")