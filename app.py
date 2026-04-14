"""
Real Estate Price Prediction - Interactive Dashboard
Streamlit Application for CA-2 Project Exhibition
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import inspect
from real_estate_pipeline import RealEstatePricePrediction
import warnings
warnings.filterwarnings('ignore')

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_artifact(primary, fallbacks):
    for path in [primary] + fallbacks:
        if os.path.exists(path):
            return joblib.load(path), path
    return None, None

def infer_is_bengaluru_df(df):
    cols = {str(c) for c in df.columns}
    has_location = any(c.startswith("location_") for c in cols)
    has_area_type = any(c.startswith("area_type_") for c in cols)
    has_bhk = "bhk" in cols
    return has_location or has_area_type or has_bhk

def infer_price_unit(df, price_col):
    if not price_col or price_col not in df.columns:
        return "USD"
    series = pd.to_numeric(df[price_col], errors="coerce").dropna()
    if series.empty:
        return "USD"
    median = series.median()
    # Heuristic: Bengaluru dataset price is commonly in lakhs (typical range ~10-200)
    if 1 <= median <= 200:
        return "Lakhs"
    return "USD"

def price_multiplier_to_inr(price_unit, usd_to_inr):
    if price_unit == "Lakhs":
        return 100000.0
    if price_unit == "INR":
        return 1.0
    return float(usd_to_inr)

def with_price_inr(df, price_col, usd_to_inr, price_unit):
    if price_col and price_col in df.columns:
        df = df.copy()
        multiplier = price_multiplier_to_inr(price_unit, usd_to_inr)
        df["_price_inr"] = df[price_col] * multiplier
        return df, "_price_inr"
    return df, price_col

def fmt_inr(value):
    try:
        return f"₹{value:,.2f}"
    except Exception:
        return "₹0.00"


def resolve_dataset_source(uploaded_file, dataset_path):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            label = f"uploaded:{uploaded_file.name}"
            return df, label, None
        except Exception as e:
            return None, None, f"Failed to read uploaded file: {e}"
    if dataset_path:
        resolved = os.path.expanduser(dataset_path)
        if not os.path.isabs(resolved):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            resolved = os.path.abspath(os.path.join(base_dir, resolved))
        if os.path.exists(resolved):
            return resolved, resolved, None
        return None, None, f"Dataset path not found: {resolved}"
    return None, None, None

def load_data_dynamic(pipeline, dataset_source):
    if dataset_source is None:
        pipeline.load_data()
        return "default"

    load_fn = getattr(pipeline, "load_data", None)
    if load_fn is None:
        raise AttributeError("Pipeline is missing load_data().")

    sig = inspect.signature(load_fn)
    params = sig.parameters

    if isinstance(dataset_source, pd.DataFrame):
        if "df" in params:
            load_fn(df=dataset_source)
            return "dataframe"
        pipeline.df = dataset_source.copy()
        if hasattr(pipeline, "raw_df"):
            pipeline.raw_df = dataset_source.copy()
        return "dataframe"

    for name in ["data_path", "file_path", "path", "csv_path", "dataset_path"]:
        if name in params:
            load_fn(**{name: dataset_source})
            return dataset_source

    try:
        load_fn(dataset_source)
        return dataset_source
    except TypeError:
        df = pd.read_csv(dataset_source)
        pipeline.df = df.copy()
        if hasattr(pipeline, "raw_df"):
            pipeline.raw_df = df.copy()
        return dataset_source

st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: black;
        text-align: center;
        font-weight: bold;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
            color: black;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'dataset_label' not in st.session_state:
    st.session_state.dataset_label = None

# Header
st.markdown('<h1 class="main-header">🏠 Real Estate Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/real-estate.png", width=100)
    st.title("🎯 Navigation")
    
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔮 Predictions", "📈 Visualizations"]
    )
    
    st.markdown("---")
    st.markdown("### 👥 Team Members")
    st.info("1. Ayush Pandey\n\n2. Chirag Pandey")
    st.info("CA-2 Project Exhibition\n\n📅 Date: 15/04/2026\n")
    
    st.markdown("---")
    st.markdown("### 🎓 Project Info")
    st.success("**Data Science Project**\n\nReal Estate Price Prediction using Ensemble Machine Learning Models")
    st.markdown("---")
    st.markdown("### 📂 Dataset")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    dataset_path_input = st.text_input("Or enter dataset path", value="cleaned_data.csv")
    dataset_source, dataset_label, dataset_error = resolve_dataset_source(uploaded_file, dataset_path_input)
    if dataset_error:
        st.warning(dataset_error)
    elif dataset_label:
        st.caption(f"Selected: {dataset_label}")
    st.markdown("---")
    st.markdown("### 💱 Currency")
    usd_to_inr = st.number_input("USD → INR rate", min_value=50.0, max_value=120.0, value=83.0, step=0.5)
    st.markdown("### 💵 Price Unit")
    price_unit_choice = st.selectbox(
        "Prediction price unit",
        ["Auto", "USD", "INR", "Lakhs"],
        index=0
    )
    st.caption("Auto detects lakhs if price median looks like 1–200 (typical Bengaluru dataset).")

# HOME PAGE
if page == "🏠 Home":
    st.header("Welcome to Real Estate Price Prediction Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Project Objective</h3>
            <p>Predict real estate prices using advanced ensemble machine learning models with comprehensive data analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔧 Technologies Used</h3>
            <p>Python, Scikit-learn, Streamlit, Pandas, Plotly, Ensemble Models (Random Forest, Gradient Boosting)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Pipeline Steps</h3>
            <p>9-Step ML Pipeline: Data Loading → EDA → Cleaning → Feature Selection → Training → Validation → Evaluation</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Pipeline Overview
    st.subheader("📋 ML Pipeline Architecture")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Pipeline Steps:
        1. **Input Data** - Load real estate dataset
        2. **Exploratory Data Analysis** - Statistical insights
        3. **Data Engineering & Cleaning** - Handle missing values
        4. **Feature Selection** - Select relevant features
        5. **Data Split** - Train/Test split (80/20)
        6. **Model Selection** - Multiple ML algorithms
        7. **Model Training** - Train ensemble models
        8. **K-Fold Validation** - 5-fold cross-validation
        9. **Performance Metrics** - Evaluate models
        """)
    
    with col2:
        st.markdown("""
        ### Models Used:
        - 🌲 **Random Forest Regressor**
        - 📈 **Gradient Boosting Regressor**
        - 🚀 **AdaBoost Regressor**
        - 📐 **Ridge Regression**
        - 📉 **Lasso Regression**
        - 🌳 **Decision Tree Regressor**
        - ⭐ **Ensemble (Voting Regressor)**
        """)
    
    st.markdown("---")
    
    # Initialize Pipeline Button
    if st.button("🚀 Initialize ML Pipeline"):
        with st.spinner("Loading data and initializing pipeline..."):
            if dataset_error:
                st.error(dataset_error)
            else:
                try:
                    st.session_state.pipeline = RealEstatePricePrediction()
                    source_used = load_data_dynamic(st.session_state.pipeline, dataset_source)
                    st.session_state.data_loaded = True
                    st.session_state.dataset_label = dataset_label or source_used
                    st.success("✅ Pipeline initialized successfully!")
                    if st.session_state.dataset_label:
                        st.caption(f"Dataset: {st.session_state.dataset_label}")
                    st.balloons()
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")

# DATA ANALYSIS PAGE
elif page == "📊 Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please initialize the pipeline from the Home page first!")
    else:
        pipeline = st.session_state.pipeline
        price_col = pick_col(pipeline.df, ["Price", "price"])
        area_col = pick_col(pipeline.df, ["Area_sqft", "area_sqft", "total_sqft", "Total_sqft"])
        bedrooms_col = pick_col(pipeline.df, ["Bedrooms", "bedrooms", "bhk", "BHK"])
        bath_col = pick_col(pipeline.df, ["Bathrooms", "bath", "bathrooms"])
        inferred_unit = infer_price_unit(pipeline.df, price_col)
        price_unit = inferred_unit if price_unit_choice == "Auto" else price_unit_choice
        df_plot, price_inr_col = with_price_inr(pipeline.df, price_col, usd_to_inr, price_unit)
        
        # Perform EDA
        if st.button("🔍 Perform EDA"):
            with st.spinner("Analyzing data..."):
                pipeline.perform_eda()
                pipeline.clean_and_engineer()
                
                st.success("✅ EDA completed!")
                
                # Dataset Overview
                st.subheader("📋 Dataset Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Samples", pipeline.df.shape[0])
                with col2:
                    st.metric("Features", pipeline.df.shape[1])
                with col3:
                    st.metric("Missing Values", pipeline.df.isnull().sum().sum())
                with col4:
                    st.metric("Numeric Features", len(pipeline.df.select_dtypes(include=[np.number]).columns))
                
                # Display Data
                st.subheader("📄 Dataset Sample")
                st.dataframe(pipeline.df.head(10), use_container_width=True)
                
                # Statistical Summary
                st.subheader("📈 Statistical Summary")
                st.dataframe(pipeline.df.describe(), use_container_width=True)
                
                # Correlation Heatmap
                st.subheader("🔥 Correlation Heatmap")
                fig = px.imshow(
                    pipeline.correlation_matrix,
                    labels=dict(color="Correlation"),
                    x=pipeline.correlation_matrix.columns,
                    y=pipeline.correlation_matrix.columns,
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution Plots
                st.subheader("📊 Feature Distributions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if price_col:
                        # Price Distribution
                        fig = px.histogram(
                            df_plot,
                            x=price_inr_col,
                            nbins=50,
                            title="Price Distribution",
                            labels={price_inr_col: "Price (₹)"}
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Price column not found for distribution plot.")
                
                with col2:
                    if area_col:
                        # Area Distribution
                        fig = px.histogram(
                            pipeline.df,
                            x=area_col,
                            nbins=50,
                            title="Area Distribution",
                            labels={area_col: "Area (sqft)"}
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Area column not found for distribution plot.")
                
                # Box Plots
                st.subheader("📦 Box Plots - Feature Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if price_col:
                        fig = px.box(
                            df_plot,
                            y=price_inr_col,
                            title="Price Box Plot"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Price column not found for box plot.")
                
                with col2:
                    if bedrooms_col and price_col:
                        fig = px.box(
                            df_plot,
                            x=bedrooms_col,
                            y=price_inr_col,
                            title="Price vs Bedrooms"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Bedrooms/Price columns not found for box plot.")

# MODEL TRAINING PAGE
elif page == "🤖 Model Training":
    st.header("🤖 Model Training & Evaluation")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please initialize the pipeline from the Home page first!")
    else:
        pipeline = st.session_state.pipeline
        
        if st.button("🚀 Train All Models"):
            with st.spinner("Training models... This may take a moment..."):
                
                # Execute pipeline steps
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Step 1/9: Loading data...")
                progress_bar.progress(11)
                
                status_text.text("Step 2/9: Performing EDA...")
                pipeline.perform_eda()
                progress_bar.progress(22)
                
                status_text.text("Step 3/9: Cleaning data...")
                pipeline.clean_and_engineer()
                progress_bar.progress(33)
                
                status_text.text("Step 4/9: Selecting features...")
                pipeline.select_features()
                progress_bar.progress(44)
                
                status_text.text("Step 5/9: Splitting data...")
                pipeline.split_data()
                progress_bar.progress(55)
                
                status_text.text("Step 6/9: Initializing models...")
                pipeline.initialize_models()
                progress_bar.progress(66)
                
                status_text.text("Step 7/9: Training models...")
                pipeline.train_models()
                progress_bar.progress(77)
                
                status_text.text("Step 8/9: K-Fold validation...")
                pipeline.perform_kfold_validation()
                progress_bar.progress(88)
                
                status_text.text("Step 9/9: Evaluating performance...")
                pipeline.evaluate_models()
                progress_bar.progress(100)
                
                pipeline.save_models()
                
                status_text.text("✅ Training completed!")
                st.success("🎉 All models trained successfully!")
                st.balloons()
                
                # Display Results
                st.markdown("---")
                st.subheader("📊 Model Performance Comparison")
                
                # Create performance dataframe
                perf_df = pd.DataFrame(pipeline.performance).T
                perf_df = perf_df.round(4)
                
                st.dataframe(perf_df.style.highlight_max(axis=0, subset=['R2'], color='lightgreen')
                                          .highlight_min(axis=0, subset=['RMSE', 'MAE'], color='lightgreen'),
                            use_container_width=True)
                
                # Visualize Performance
                col1, col2 = st.columns(2)
                
                with col1:
                    # R2 Score Comparison
                    fig = px.bar(
                        x=list(pipeline.performance.keys()),
                        y=[v['R2'] for v in pipeline.performance.values()],
                        title="R² Score Comparison",
                        labels={"x": "Model", "y": "R² Score"},
                        color=[v['R2'] for v in pipeline.performance.values()],
                        color_continuous_scale="Viridis"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # RMSE Comparison
                    fig = px.bar(
                        x=list(pipeline.performance.keys()),
                        y=[v['RMSE'] for v in pipeline.performance.values()],
                        title="RMSE Comparison (Lower is Better)",
                        labels={"x": "Model", "y": "RMSE"},
                        color=[v['RMSE'] for v in pipeline.performance.values()],
                        color_continuous_scale="RdYlGn_r"
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cross-Validation Results
                st.subheader("🔄 K-Fold Cross-Validation Results")
                
                cv_df = pd.DataFrame({
                    'Model': list(pipeline.cv_results.keys()),
                    'Mean R²': [v['mean'] for v in pipeline.cv_results.values()],
                    'Std Dev': [v['std'] for v in pipeline.cv_results.values()]
                }).round(4)
                
                st.dataframe(cv_df, use_container_width=True)
                
                # CV Visualization
                fig = go.Figure()
                for name, results in pipeline.cv_results.items():
                    fig.add_trace(go.Box(
                        y=results['scores'],
                        name=name,
                        boxmean='sd'
                    ))
                
                fig.update_layout(
                    title="K-Fold Cross-Validation Scores Distribution",
                    yaxis_title="R² Score",
                    showlegend=True,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature Importance
                st.subheader("🎯 Feature Importance (Random Forest)")
                
                importance_df = pipeline.get_feature_importance()
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance Ranking",
                    color='Importance',
                    color_continuous_scale="Blues"
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
                st.plotly_chart(fig, use_container_width=True)

# PREDICTIONS PAGE
elif page == "🔮 Predictions":
    st.header("🔮 Make Price Predictions")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please train the models first!")
    else:
        model, model_path = load_artifact("ensemble_model.pkl", ["model.pkl"])
        scaler, scaler_path = load_artifact("scaler.pkl", [])
        feature_cols, cols_path = load_artifact("feature_cols.pkl", ["columns.pkl"])

        if model is None or scaler is None or feature_cols is None:
            st.error("Model artifacts not found. Please train the models from the Model Training page.")
        else:
            st.caption(f"Using model: {model_path}, scaler: {scaler_path}, columns: {cols_path}")
            st.subheader("Enter Property Details")

            is_bengaluru = any(str(c).startswith("location_") for c in feature_cols)
            price_col_for_unit = None
            if st.session_state.pipeline is not None and st.session_state.pipeline.df is not None:
                price_col_for_unit = pick_col(st.session_state.pipeline.df, ["Price", "price"])
                inferred_unit = infer_price_unit(st.session_state.pipeline.df, price_col_for_unit)
            else:
                inferred_unit = "USD"
            price_unit = inferred_unit if price_unit_choice == "Auto" else price_unit_choice

            if is_bengaluru:
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_sqft = st.number_input("Total Sqft", min_value=300.0, max_value=10000.0, value=1000.0, step=50.0)
                    bhk = st.slider("BHK", 1, 10, 2)
                with col2:
                    bath = st.slider("Bathrooms (bath)", 1, 10, 2)
                    balcony = st.slider("Balcony", 0, 5, 1)
                with col3:
                    area_types = [c.replace("area_type_", "") for c in feature_cols if str(c).startswith("area_type_")]
                    locations = [c.replace("location_", "") for c in feature_cols if str(c).startswith("location_")]
                    area_type = st.selectbox("Area Type", area_types if area_types else ["Other"])
                    location = st.selectbox("Location", locations if locations else ["other"])
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    area = st.number_input("Area (sqft)", min_value=500, max_value=10000, value=2000, step=100)
                    bedrooms = st.slider("Bedrooms", 1, 6, 3)
                    bathrooms = st.slider("Bathrooms", 1, 5, 2)
                with col2:
                    age = st.slider("Age (Years)", 0, 50, 5)
                    floor = st.slider("Floor", 1, 20, 5)
                    location_score = st.slider("Location Score", 1.0, 10.0, 7.0, 0.1)
                with col3:
                    parking = st.slider("Parking Spaces", 0, 4, 1)
                    garden = st.selectbox("Garden", [0, 1], format_func=lambda x: "Yes" if x else "No")
                    elevator = st.selectbox("Elevator", [0, 1], format_func=lambda x: "Yes" if x else "No")

        if st.button("💰 Predict Price"):
            try:
                if model is None or scaler is None or feature_cols is None:
                    raise FileNotFoundError("Model artifacts missing.")

                if is_bengaluru:
                    row = {c: 0 for c in feature_cols}
                    if "total_sqft" in row:
                        row["total_sqft"] = float(total_sqft)
                    if "bath" in row:
                        row["bath"] = float(bath)
                    if "balcony" in row:
                        row["balcony"] = float(balcony)
                    if "bhk" in row:
                        row["bhk"] = float(bhk)
                    area_col = f"area_type_{area_type}"
                    loc_col = f"location_{location}"
                    if area_col in row:
                        row[area_col] = 1
                    if loc_col in row:
                        row[loc_col] = 1
                    input_data = pd.DataFrame([row])
                else:
                    input_data = pd.DataFrame({
                        'Area_sqft': [area],
                        'Bedrooms': [bedrooms],
                        'Bathrooms': [bathrooms],
                        'Age_Years': [age],
                        'Floor': [floor],
                        'Location_Score': [location_score],
                        'Parking_Spaces': [parking],
                        'Garden': [garden],
                        'Elevator': [elevator],
                        'Condition_Encoded': [2],
                        'Room_Total': [bedrooms + bathrooms],
                        'Luxury_Score': [location_score * 0.4 + elevator * 2 + garden * 2]
                    })

                    # Align to training features
                    input_data = input_data.reindex(columns=feature_cols, fill_value=0)

                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                prediction_inr = prediction * price_multiplier_to_inr(price_unit, usd_to_inr)
                
                # Display prediction
                st.markdown("---")
                if price_unit == "Lakhs":
                    st.success(f"## Predicted Price: {prediction:,.2f} Lakh ({fmt_inr(prediction_inr)})")
                else:
                    st.success(f"## Predicted Price: {fmt_inr(prediction_inr)}")
                
                col1, col2, col3 = st.columns(3)
                if is_bengaluru:
                    with col1:
                        st.metric("Price per sqft", fmt_inr(prediction_inr / total_sqft))
                    with col2:
                        st.metric("BHK", bhk)
                    with col3:
                        st.metric("Bathrooms", bath)
                else:
                    with col1:
                        st.metric("Price per sqft", fmt_inr(prediction_inr / area))
                    with col2:
                        st.metric("Total Rooms", bedrooms + bathrooms)
                    with col3:
                        luxury = location_score * 0.4 + elevator * 2 + garden * 2
                        st.metric("Luxury Score", f"{luxury:.2f}/10")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# VISUALIZATIONS PAGE
elif page == "📈 Visualizations":
    st.header("📈 Advanced Visualizations")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please initialize the pipeline first!")
    else:
        pipeline = st.session_state.pipeline
        price_col = pick_col(pipeline.df, ["Price", "price"])
        area_col = pick_col(pipeline.df, ["Area_sqft", "area_sqft", "total_sqft", "Total_sqft"])
        bedrooms_col = pick_col(pipeline.df, ["Bedrooms", "bedrooms", "bhk", "BHK"])
        bath_col = pick_col(pipeline.df, ["Bathrooms", "bath", "bathrooms"])
        age_col = pick_col(pipeline.df, ["Age_Years", "age", "age_years", "age_years_old"])
        location_col = pick_col(pipeline.df, ["Location_Score", "location_score"])
        floor_col = pick_col(pipeline.df, ["Floor", "floor"])
        inferred_unit = infer_price_unit(pipeline.df, price_col)
        price_unit = inferred_unit if price_unit_choice == "Auto" else price_unit_choice
        df_plot, price_inr_col = with_price_inr(pipeline.df, price_col, usd_to_inr, price_unit)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Relationships", "📦 Comparisons", "🗺️ 3D Analysis"])
        
        with tab1:
            st.subheader("Feature Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if price_col:
                    # Violin Plot
                    fig = px.violin(
                        df_plot,
                        y=price_inr_col,
                        box=True,
                        title="Price Distribution (Violin Plot)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Price column not found for violin plot.")
            
            with col2:
                if bedrooms_col:
                    # Bedrooms Distribution
                    fig = px.pie(
                        pipeline.df,
                        names=bedrooms_col,
                        title="Distribution of Bedrooms",
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Bedrooms column not found for pie chart.")
        
        with tab2:
            st.subheader("Feature Relationships")
            
            # Scatter Matrix
            dims = [area_col, bedrooms_col, age_col, price_col]
            if price_col and price_inr_col:
                dims = [d for d in dims if d != price_col] + [price_inr_col]
            dims = [d for d in dims if d]
            if len(dims) >= 2:
                fig = px.scatter_matrix(
                    df_plot,
                    dimensions=dims,
                    title="Scatter Matrix - Key Features",
                    height=700
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough matching columns for scatter matrix.")
            
            # Pairwise Scatter
            col1, col2 = st.columns(2)
            
            with col1:
                if area_col and price_col:
                    fig = px.scatter(
                        df_plot,
                        x=area_col,
                        y=price_inr_col,
                        color=bedrooms_col if bedrooms_col else None,
                        size=bath_col if bath_col else None,
                        title="Price vs Area (colored by Bedrooms)",
                        trendline="ols"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Area/Price columns not found for scatter plot.")
            
            with col2:
                if location_col and price_col:
                    fig = px.scatter(
                        df_plot,
                        x=location_col,
                        y=price_inr_col,
                        color=age_col if age_col else None,
                        title="Price vs Location Score",
                        trendline="ols"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Location Score/Price columns not found for scatter plot.")
        
        with tab3:
            st.subheader("Comparative Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if bedrooms_col and price_col:
                    # Price by Bedrooms
                    fig = px.box(
                        df_plot,
                        x=bedrooms_col,
                        y=price_inr_col,
                        color=bedrooms_col,
                        title="Price Distribution by Bedrooms"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Bedrooms/Price columns not found for comparison.")
            
            with col2:
                if floor_col and price_col:
                    # Price by Floor
                    avg_price_floor = df_plot.groupby(floor_col)[price_inr_col].mean().reset_index()
                    fig = px.line(
                        avg_price_floor,
                        x=floor_col,
                        y=price_inr_col,
                        title="Average Price by Floor",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Floor/Price columns not found for line chart.")
        
        with tab4:
            st.subheader("3D Analysis")
            
            # 3D Scatter Plot
            x_3d = area_col
            y_3d = location_col if location_col else bedrooms_col if bedrooms_col else bath_col
            z_3d = price_inr_col if price_inr_col else price_col

            if not (x_3d and y_3d and z_3d):
                numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 3:
                    x_3d, y_3d, z_3d = numeric_cols[:3]

            if x_3d and y_3d and z_3d:
                fig = px.scatter_3d(
                    df_plot.sample(200),
                    x=x_3d,
                    y=y_3d,
                    z=z_3d,
                    color=bedrooms_col if bedrooms_col else None,
                    size=bath_col if bath_col else None,
                    title="3D Scatter: Area vs Feature vs Price",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric columns for 3D plot.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Real Estate Price Prediction System</strong></p>
    <p>CA-2 Project Exhibition | Data Science & Machine Learning</p>
    <p>Built with ❤️ using Streamlit, Scikit-learn & Plotly</p>
</div>
""", unsafe_allow_html=True)
