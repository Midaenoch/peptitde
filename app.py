import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="🧬 Bioactive Peptide Prediction",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #EFF6FF, #F3E8FF);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: white;
        border-radius: 8px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("# 🧬 Bioactive Peptide Prediction System")
st.markdown("**Predict Peptides & Biological Activities from Microbial Fermentation Conditions**")
st.markdown("---")

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.rf_model = None
    st.session_state.scaler = None
    st.session_state.label_encoders = {}
    st.session_state.output_info = {}
    st.session_state.features = []

# Function to load models
@st.cache_resource
def load_models(model_path, scaler_path, encoders_path, output_info_path):
    """Load pre-trained models and preprocessors"""
    try:
        rf_model = joblib.load(bioactive_model.pkl)
        scaler = joblib.load(scaler.pkl)
        label_encoders = joblib.load(encoders.pkl)
        feautures = joblib.load(feautures.pkl)
        output_info = joblib.load(output_info.pkl)
        return rf_model, scaler, label_encoders, output_info, feautures True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, False

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Overview", 
    "🔧 Load Models",
    "🎯 Make Predictions"
])

# TAB 1: Overview
with tab1:
    st.header("📊 System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### 🎯 What This Does")
        st.markdown("""
        This application predicts bioactive peptides and biological activities 
        based on fermentation conditions using pre-trained machine learning models.
        
        **Input Parameters:**
        - Protein Source (e.g., Cow's milk, Goat's milk)
        - Microorganism (e.g., Lactobacillus species)
        - Inoculum Size (% v/v)
        - Temperature (°C)
        - Fermentation Time (hours)
        - pH Level
        - Stirring Speed (rpm)
        """)
        
        st.success("### ✅ Model Architecture")
        st.markdown("""
        - **Algorithm:** Random Forest with Multi-Output Regression
        - **Preprocessing:** Label Encoding + Standard Scaling
        - **Feature Engineering:** Interaction features (Temp × pH, Time × Temp)
        - **Outputs:** Binary predictions + confidence scores
        """)
    
    with col2:
        st.warning("### 📦 Required Model Files")
        st.markdown("""
        Place these files in a folder and upload them:
        
        1. **`rf_model.pkl`** - Trained Random Forest model
        2. **`scaler.pkl`** - StandardScaler for features
        3. **`label_encoders.pkl`** - LabelEncoders for categorical variables
        4. **`output_info.pkl`** - Output class information (activities & peptides)
        
        You can upload them individually or specify a folder path.
        """)
        
        st.info("### 🎯 Predictions")
        st.markdown("""
        **1. Biological Activities:**  
        ACE inhibitory, antioxidative, antimicrobial, etc.
        
        **2. Bioactive Peptides:**  
        Specific peptide sequences like VPP, IPP, LVYPFP, etc.
        
        Each prediction includes a confidence score.
        """)

# TAB 2: Load Models
with tab2:
    st.header("🔧 Load Pre-trained Models")
    
    st.info("💡 **Tip:** Upload all 4 required .pkl files to get started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_file = st.file_uploader("📊 Upload Model (rf_model.pkl)", type=['pkl'])
        scaler_file = st.file_uploader("📏 Upload Scaler (scaler.pkl)", type=['pkl'])
    
    with col2:
        encoders_file = st.file_uploader("🔤 Upload Label Encoders (label_encoders.pkl)", type=['pkl'])
        output_file = st.file_uploader("📋 Upload Output Info (output_info.pkl)", type=['pkl'])
    
    if st.button("🚀 Load Models", type="primary", use_container_width=True):
        if all([model_file, scaler_file, encoders_file, output_file]):
            with st.spinner("Loading models..."):
                try:
                    # Load files
                    st.session_state.rf_model = joblib.load(model_file)
                    st.session_state.scaler = joblib.load(scaler_file)
                    st.session_state.label_encoders = joblib.load(encoders_file)
                    st.session_state.output_info = joblib.load(output_file)
                    
                    # Extract feature names from scaler
                    if hasattr(st.session_state.scaler, 'feature_names_in_'):
                        st.session_state.features = list(st.session_state.scaler.feature_names_in_)
                    else:
                        # Default feature list
                        st.session_state.features = [
                            'Temperature (°C)', 'pH', 'Stirring (rpm)', 'Inoculum_Pct',
                            'Time_Hours', 'Temp_pH', 'Time_Temp', 
                            'Protein Source_Enc', 'Microorganism_Enc'
                        ]
                    
                    st.session_state.models_loaded = True
                    
                    st.success("✅ All models loaded successfully!")
                    
                    # Display model info
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Activities", len(st.session_state.output_info['activity_classes']))
                    col2.metric("Peptides", len(st.session_state.output_info['peptide_classes']))
                    col3.metric("Features", len(st.session_state.features))
                    
                    with st.expander("📋 View Model Details"):
                        st.write("**Activity Classes:**")
                        st.write(list(st.session_state.output_info['activity_classes']))
                        st.write("\n**Peptide Classes:**")
                        st.write(list(st.session_state.output_info['peptide_classes']))
                        st.write("\n**Feature Names:**")
                        st.write(st.session_state.features)
                    
                except Exception as e:
                    st.error(f"❌ Error loading models: {str(e)}")
        else:
            st.warning("⚠️ Please upload all 4 required files")

# TAB 3: Make Predictions
with tab3:
    st.header("🎯 Make Predictions")
    
    if st.session_state.models_loaded:
        st.info("### Enter Fermentation Conditions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            protein_source = st.selectbox("Protein Source", 
                ["Cow's milk", "Goat's milk", "Camel milk", "Skimmed milk", "Soy milk", "Other"])
            
            microorganism = st.text_input("Microorganism", 
                "Lactobacillus helveticus",
                help="e.g., Lactobacillus helveticus, Lactobacillus bulgaricus")
            
            inoculum_size = st.text_input("Inoculum Size", "3% (v/v)",
                help="Format: X% (v/v) or X% (w/v)")
            
            temperature = st.number_input("Temperature (°C)", 
                min_value=20.0, max_value=50.0, value=37.0, step=0.5)
        
        with col2:
            time = st.text_input("Time", "24 hr",
                help="Format: X hr or X hours")
            
            ph = st.number_input("pH", 
                min_value=2.0, max_value=10.0, value=6.5, step=0.1)
            
            stirring = st.number_input("Stirring (rpm)", 
                min_value=0, max_value=500, value=150, step=10)
        
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            with st.spinner("Making predictions..."):
                try:
                    # Create prediction dataframe
                    new_data = pd.DataFrame({
                        'Protein Source': [protein_source],
                        'Microorganism': [microorganism],
                        'Inoculum Size': [inoculum_size],
                        'Temperature (°C)': [temperature],
                        'Time': [time],
                        'pH': [ph],
                        'Stirring (rpm)': [stirring]
                    })
                    
                    # Feature engineering
                    new_data['Inoculum_Pct'] = new_data['Inoculum Size'].str.extract(r'(\d+\.?\d*)').astype(float)
                    new_data['Time_Hours'] = new_data['Time'].str.extract(r'(\d+)').astype(float)
                    new_data['Temp_pH'] = new_data['Temperature (°C)'] * new_data['pH']
                    new_data['Time_Temp'] = new_data['Time_Hours'] * new_data['Temperature (°C)']
                    
                    # Encode categorical variables
                    for col in ['Protein Source', 'Microorganism']:
                        if col in st.session_state.label_encoders:
                            le = st.session_state.label_encoders[col]
                            try:
                                new_data[col + '_Enc'] = le.transform([new_data[col].iloc[0]])
                            except ValueError:
                                # Handle unseen categories
                                st.warning(f"⚠️ '{new_data[col].iloc[0]}' not seen during training. Using default encoding.")
                                new_data[col + '_Enc'] = 0
                        else:
                            new_data[col + '_Enc'] = 0
                    
                    # Prepare features in correct order
                    new_scaled = st.session_state.scaler.transform(new_data[st.session_state.features])
                    
                    # Make prediction
                    pred = st.session_state.rf_model.predict(new_scaled)[0]
                    pred_bin = (pred > 0.5).astype(int)
                    
                    # Split predictions
                    n_act = st.session_state.output_info['n_activities']
                    pred_activities = pred_bin[:n_act]
                    pred_peptides = pred_bin[n_act:]
                    scores_activities = pred[:n_act]
                    scores_peptides = pred[n_act:]
                    
                    # Display results
                    st.markdown("---")
                    st.success("### ✅ Prediction Results")
                    
                    # Input summary
                    with st.expander("📋 Input Conditions Summary", expanded=True):
                        input_df = pd.DataFrame({
                            'Parameter': ['Protein Source', 'Microorganism', 'Inoculum Size', 
                                         'Temperature', 'Time', 'pH', 'Stirring'],
                            'Value': [protein_source, microorganism, inoculum_size, 
                                     f"{temperature}°C", time, ph, f"{stirring} rpm"]
                        })
                        st.dataframe(input_df, use_container_width=True, hide_index=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Predicted Biological Activities")
                        activities_data = []
                        for i in range(len(pred_activities)):
                            if pred_activities[i]:
                                activities_data.append({
                                    'Activity': st.session_state.output_info['activity_classes'][i],
                                    'Confidence': f"{scores_activities[i]:.1%}",
                                    'Score': scores_activities[i]
                                })
                        
                        if activities_data:
                            activities_df = pd.DataFrame(activities_data)
                            activities_df = activities_df.sort_values('Score', ascending=False)
                            st.dataframe(activities_df[['Activity', 'Confidence']], 
                                       use_container_width=True, hide_index=True)
                            
                            # Visualization
                            fig = go.Figure(go.Bar(
                                x=[d['Score'] for d in activities_data],
                                y=[d['Activity'] for d in activities_data],
                                orientation='h',
                                marker=dict(color='#3B82F6')
                            ))
                            fig.update_layout(
                                title="Activity Confidence Scores",
                                xaxis_title="Confidence",
                                yaxis_title="",
                                height=300,
                                margin=dict(l=0, r=0, t=40, b=0)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No activities predicted with high confidence (>50%)")
                    
                    with col2:
                        st.markdown("### 🧬 Predicted Bioactive Peptides")
                        peptides_data = []
                        for i in range(len(pred_peptides)):
                            if pred_peptides[i]:
                                peptides_data.append({
                                    'Peptide': st.session_state.output_info['peptide_classes'][i],
                                    'Confidence': f"{scores_peptides[i]:.1%}",
                                    'Score': scores_peptides[i]
                                })
                        
                        if peptides_data:
                            peptides_df = pd.DataFrame(peptides_data)
                            peptides_df = peptides_df.sort_values('Score', ascending=False)
                            st.dataframe(peptides_df[['Peptide', 'Confidence']], 
                                       use_container_width=True, hide_index=True)
                            
                            # Visualization
                            fig = go.Figure(go.Bar(
                                x=[d['Score'] for d in peptides_data],
                                y=[d['Peptide'] for d in peptides_data],
                                orientation='h',
                                marker=dict(color='#10B981')
                            ))
                            fig.update_layout(
                                title="Peptide Confidence Scores",
                                xaxis_title="Confidence",
                                yaxis_title="",
                                height=300,
                                margin=dict(l=0, r=0, t=40, b=0)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No peptides predicted with high confidence (>50%)")
                    
                    # Summary metrics
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Activities Predicted", len(activities_data))
                    col2.metric("Total Peptides Predicted", len(peptides_data))
                    avg_conf = np.mean([d['Score'] for d in activities_data + peptides_data]) if (activities_data or peptides_data) else 0
                    col3.metric("Average Confidence", f"{avg_conf:.1%}")
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.exception(e)
    
    else:
        st.warning("⚠️ Please load the models first in the 'Load Models' tab!")
        st.info("👈 Go to the **Load Models** tab and upload your .pkl files")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280;'>
    💡 <strong>Note:</strong> Predictions are based on pre-trained models | 
    Confidence threshold: 50% | Handles unseen categories gracefully
</div>
""", unsafe_allow_html=True)
