import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Clinical reference ranges
GLUCOSE_NORMAL = (70, 100)
BMI_NORMAL = (18.5, 24.9)
BP_NORMAL = (60, 80)

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, 
                     insulin, bmi, pedigree, age):
    """
    Predicts diabetes risk based on patient health metrics.
    Returns prediction, probability, and risk assessment.
    """
    
    # Prepare input
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, pedigree, age]])
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Risk assessment
    risk_score = probability[1] * 100
    
    if risk_score < 30:
        risk_level = "🟢 Low Risk"
        recommendation = "Continue healthy lifestyle. Regular checkups recommended."
    elif risk_score < 60:
        risk_level = "🟡 Moderate Risk"
        recommendation = "Consult with healthcare provider. Consider lifestyle modifications."
    else:
        risk_level = "🔴 High Risk"
        recommendation = "⚠️ Please consult a healthcare professional immediately for proper screening."
    
    # Clinical insights
    insights = []
    if glucose > GLUCOSE_NORMAL[1]:
        insights.append(f"⚠️ Glucose level ({glucose}) is above normal range")
    if bmi > BMI_NORMAL[1]:
        insights.append(f"⚠️ BMI ({bmi:.1f}) indicates overweight/obesity")
    if blood_pressure > BP_NORMAL[1]:
        insights.append(f"⚠️ Blood pressure ({blood_pressure}) is elevated")
    
    # Format output
    result = f"""
    ### Prediction Result
    
    **Diabetes Risk: {risk_level}**
    
    **Risk Probability: {risk_score:.1f}%**
    
    ---
    
    ### Clinical Insights
    {chr(10).join(insights) if insights else "✅ All measurements within normal ranges"}
    
    ---
    
    ### Recommendation
    {recommendation}
    
    ---
    
    **⚠️ DISCLAIMER:** This is an educational tool, NOT a medical diagnostic device. 
    Always consult qualified healthcare professionals for medical advice.
    """
    
    return result

# Gradio Interface
with gr.Blocks(title="Diabetes Risk Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 Diabetes Risk Prediction Tool
    ### Educational ML Demo - Part of "Making AI Simple" Series
    
    Enter patient health metrics below to assess diabetes risk.
    This tool uses a Random Forest model trained on the Pima Indians Diabetes Database.
    """)
    
    with gr.Row():
        with gr.Column():
            pregnancies = gr.Slider(0, 17, value=1, step=1, label="Number of Pregnancies")
            glucose = gr.Slider(0, 200, value=120, step=1, label="Glucose Level (mg/dL)")
            blood_pressure = gr.Slider(0, 122, value=70, step=1, label="Blood Pressure (mm Hg)")
            skin_thickness = gr.Slider(0, 99, value=20, step=1, label="Skin Thickness (mm)")
            
        with gr.Column():
            insulin = gr.Slider(0, 846, value=79, step=1, label="Insulin Level (mu U/ml)")
            bmi = gr.Slider(0, 67, value=25.0, step=0.1, label="BMI (kg/m²)")
            pedigree = gr.Slider(0.0, 2.5, value=0.5, step=0.01, label="Diabetes Pedigree Function")
            age = gr.Slider(21, 81, value=33, step=1, label="Age (years)")
    
    predict_btn = gr.Button("Predict Diabetes Risk", variant="primary")
    output = gr.Markdown()
    
    predict_btn.click(
        fn=predict_diabetes,
        inputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### About This Tool
    - **Model:** Random Forest Classifier
    - **Dataset:** Pima Indians Diabetes Database (768 patients)
    - **Metrics:** Optimized for recall (catching diabetes cases)
    - **Creator:** [Your Name] | Part of #MakingAISimple Series
    
    📚 Learn how to build this: [Link to Day 5 & 6 LinkedIn posts]
    """)

demo.launch()
