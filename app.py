# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ======================
# 加载模型和背景数据
# ======================
@st.cache_resource
def load_model():
    model = joblib.load("diabetes_xgboost_model.pkl")
    background = pd.read_csv("background_features.csv")
    return model, background

model, background = load_model()

# ======================
# 页面标题
# ======================
st.title("老年人糖尿病风险因素识别系统系统")
st.markdown("""
基于 CLHLS 队列数据训练的 XGBoost 模型，融合个体行为与环境因素。
""")

# ======================
# 用户输入
# ======================
st.header("请输入您的信息")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("年龄", 60, 100, 75)
    bmi = st.number_input("BMI (体重kg/身高m²)", 15.0, 40.0, 24.0)
    smoking = st.selectbox("是否吸烟", ["否", "是"])
    alcohol = st.selectbox("是否饮酒", ["否", "是"])

with col2:
    exercise = st.selectbox("是否经常锻炼", ["否", "是"])
    residence = st.selectbox("居住地", ["农村", "城市"])
    pm25 = st.slider("PM2.5 年均浓度 (μg/m³)", 10, 100, 35)
    income = st.number_input("年人均可支配收入 (千元)", 5, 80, 30)

# GDP 默认值（可简化）
gdp = 80  # 单位：千元

# 编码分类变量
smoking_val = 1 if smoking == "是" else 0
alcohol_val = 1 if alcohol == "是" else 0
exercise_val = 1 if exercise == "是" else 0
residence_val = 1 if residence == "城市" else 0

# 构造输入向量（顺序必须和训练时一致！）
input_data = pd.DataFrame({
    'smoking': [smoking_val],
    'alcohol': [alcohol_val],
    'exercise': [exercise_val],
    'bmi': [bmi],
    'residence': [residence_val],
    '年均可支配收入': [gdp],
    'income': [income],
    'pm25': [pm25]
})

# ======================
# 预测
# ======================
if st.button("风险因素识别"):
    prob = model.predict_proba(input_data)[0][1]  # 患病概率
    
    st.subheader(f"您的糖尿病风险因素识别概率为：{prob:.1%}")
    
    if prob > 0.5:
        st.error("⚠️ 高风险！建议定期筛查并改善生活方式。")
    elif prob > 0.3:
        st.warning("🟡 中等风险，注意饮食和运动。")
    else:
        st.success("🟢 低风险，继续保持健康习惯！")
    
    # ======================
    # SHAP 解释
    # ======================
    st.subheader("风险因素解释（SHAP）")
    
    # 使用背景数据初始化 explainer
    explainer = shap.Explainer(model, background)
    shap_values = explainer(input_data)
    
    # 绘制 force plot（单个样本解释）
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=8, show=False)
    st.pyplot(fig)
    
    st.markdown("""
    > **解读**：红色表示增加风险的因素，蓝色表示降低风险的因素。长度越长，风险程度越高。
    """)

# ======================
# 底部说明
# ======================
st.markdown("---")
st.caption("本模型基于中国老年健康影响因素跟踪调查（CLHLS）数据训练，仅用于科研与健康参考。")
