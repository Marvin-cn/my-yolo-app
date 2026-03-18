import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter

# --- 新增：果蔬中英文对照表 ---
# 如果发现有没翻译的单词，直接在这个大括号里加一行就行！
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

# 1. 加载模型
@st.cache_resource
def load_model():
    return YOLO("yolo_fruits_and_vegetables_v3.pt") 

model = load_model()

# 2. 侧边栏：设置识别标准
st.sidebar.header("识别设置")
conf_threshold = st.sidebar.slider("灵敏度 (Confidence)", 0.0, 1.0, 0.25)

# 3. 文件上传
uploaded_file = st.file_uploader("请上传图片...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image.thumbnail((800, 800)) 
    
    with st.spinner('AI 正在分析中...'):
        img_array = np.array(image)
        # 执行识别
        results = model.predict(source=img_array, conf=conf_threshold, iou=0.7)
        res_plotted = results[0].plot()
        
        # 获取所有识别到的物体类别名称
        names = model.names
        detected_indices = results[0].boxes.cls.cpu().numpy().astype(int)
        detected_names = [names[i] for i in detected_indices]
        
        # 统计每个物体出现的次数
        counts = Counter(detected_names)

    # 展示结果
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="原始图片", use_container_width=True)
    with col2:
        st.image(res_plotted, caption="识别结果图", use_container_width=True)

    # 文字描述区域
    st.divider()
    st.markdown("### 📝 识别报告")
    
    if len(detected_names) > 0:
        summary_text = f"**成功识别 {len(detected_names)} 个目标：** "
        
        detail_items = []
        for obj, count in counts.items():
            # --- 新增翻译逻辑 ---
            # obj.lower() 是为了把首字母大写等情况统一转成小写去字典里查
            # .get() 的第二个参数 obj 是“后备选项”：如果字典里没收录这个词，就原样输出英文
            zh_name = FRUIT_VEG_DICT.get(obj.lower(), obj) 
            
            detail_items.append(f"{zh_name} {count} 个")
        
        full_report = summary_text + "，".join(detail_items) + "。"
        st.success(full_report)
    else:
        st.warning("未识别到任何目标，请尝试调低左侧的灵敏度。")