import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
MODEL_PATH = 'crop_yield_model.pkl'
FEATURE_NAMES_PATH = 'feature_names.pkl'
DATA_PATH = 'crop_yield.csv'  # Make sure this file is in your app directory

# Page config
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #388E3C;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #E8F5E8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    .prediction-result {
        background-color: #F3E5F5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #9C27B0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #7B1FA2;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Helper Functions ------------------- #
def normalize_state_name(state_name):
    """Normalize state names to match format"""
    return re.sub(r'\s+', ' ', state_name.strip().title())

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        if not os.path.exists(DATA_PATH):
            st.error(f"Dataset not found at {DATA_PATH}. Please upload your crop_yield.csv file.")
            return None
        
        df = pd.read_csv(DATA_PATH)
        required_columns = {'Crop', 'Season', 'State', 'Area', 
                           'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield'}
        
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            st.error(f"Dataset missing required columns: {missing}")
            return None
        
        # Normalize state names
        df['State'] = df['State'].apply(normalize_state_name)
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def train_model():
    """Train and cache the machine learning model"""
    df = load_data()
    if df is None:
        return None, None
        
    try:
        # Feature engineering
        X = df[['Crop', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
        y = df['Yield']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Preprocessing pipeline
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Crop', 'Season', 'State']),
            ('num', StandardScaler(), ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'])
        ])
        
        # Model pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=6,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Training
        with st.spinner('Training model...'):
            model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Save model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
            
        return model, {'mse': mse, 'r2': r2}
        
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, None

def get_model():
    """Load or train the model"""
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            return model, None
        except:
            return train_model()
    else:
        return train_model()

def create_state_insights(df, selected_state):
    """Create insights for selected state"""
    state_df = df[df['State'] == selected_state]
    
    if state_df.empty:
        return None
    
    insights = {
        'avg_yield': round(state_df['Yield'].mean(), 2),
        'max_yield': round(state_df['Yield'].max(), 2),
        'min_yield': round(state_df['Yield'].min(), 2),
        'top_crop': state_df.groupby('Crop')['Yield'].mean().idxmax(),
        'avg_rainfall': round(state_df['Annual_Rainfall'].mean(), 2),
        'crop_count': state_df['Crop'].nunique(),
        'total_records': len(state_df)
    }
    
    return insights

def create_visualizations(df, selected_state=None):
    """Create visualizations for the dashboard"""
    
    # Filter data if state is selected
    plot_df = df[df['State'] == selected_state] if selected_state else df
    
    # 1. Yield by Crop
    fig1 = px.box(plot_df, x='Crop', y='Yield', 
                  title=f'Yield Distribution by Crop {f"in {selected_state}" if selected_state else ""}')
    fig1.update_layout(xaxis_tickangle=45)
    
    # 2. Yield by Season
    fig2 = px.violin(plot_df, x='Season', y='Yield', 
                     title=f'Yield Distribution by Season {f"in {selected_state}" if selected_state else ""}')
    
    # 3. Correlation heatmap
    numeric_cols = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
    corr_matrix = plot_df[numeric_cols].corr()
    
    fig3 = px.imshow(corr_matrix, 
                     title='Feature Correlation Matrix',
                     color_continuous_scale='RdBu_r',
                     aspect='auto')
    
    # 4. Top performing crops
    top_crops = plot_df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(10)
    fig4 = px.bar(x=top_crops.values, y=top_crops.index, orientation='h',
                  title=f'Top 10 Crops by Average Yield {f"in {selected_state}" if selected_state else ""}')
    fig4.update_layout(yaxis_title='Crop', xaxis_title='Average Yield (tonnes/hectare)')
    
    return fig1, fig2, fig3, fig4

# ------------------- Main App ------------------- #
def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Crop Yield Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #666;">Predict crop yields using machine learning</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["🏠 Home", "📊 Analytics", "🔮 Prediction", "ℹ️ About"])
    
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "📊 Analytics":
        show_analytics_page(df)
    elif page == "🔮 Prediction":
        show_prediction_page(df)
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page(df):
    """Display the home page with overview"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("Crops Available", f"{df['Crop'].nunique()}")
    
    with col3:
        st.metric("States Covered", f"{df['State'].nunique()}")
    
    with col4:
        st.metric("Avg Yield", f"{df['Yield'].mean():.2f} t/ha")
    
    st.markdown("---")
    
    # Quick insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Performing Crops")
        top_crops = df.groupby('Crop')['Yield'].mean().sort_values(ascending=False).head(5)
        for crop, yield_val in top_crops.items():
            st.write(f"**{crop}**: {yield_val:.2f} tonnes/hectare")
    
    with col2:
        st.subheader("🌍 Top Performing States")
        top_states = df.groupby('State')['Yield'].mean().sort_values(ascending=False).head(5)
        for state, yield_val in top_states.items():
            st.write(f"**{state}**: {yield_val:.2f} tonnes/hectare")
    
    # Sample data preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

def show_analytics_page(df):
    """Display analytics and visualizations"""
    
    st.header("📊 Data Analytics Dashboard")
    
    # State selector
    selected_state = st.selectbox("Select State for Detailed Analysis (Optional)", 
                                ["All States"] + sorted(df['State'].unique()))
    
    if selected_state != "All States":
        # State insights
        insights = create_state_insights(df, selected_state)
        if insights:
            st.subheader(f"📈 Insights for {selected_state}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Yield", f"{insights['avg_yield']} t/ha")
            with col2:
                st.metric("Top Crop", insights['top_crop'])
            with col3:
                st.metric("Avg Rainfall", f"{insights['avg_rainfall']} mm")
            with col4:
                st.metric("Crops Count", insights['crop_count'])
    
    # Create visualizations
    state_filter = selected_state if selected_state != "All States" else None
    fig1, fig2, fig3, fig4 = create_visualizations(df, state_filter)
    
    # Display charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)

def show_prediction_page(df):
    """Display the prediction interface"""
    
    st.header("🔮 Crop Yield Prediction")
    
    # Get or train model
    model, metrics = get_model()
    if model is None:
        st.error("Failed to load or train model. Please check your dataset.")
        return
    
    # Display model performance if available
    if metrics:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model R² Score", f"{metrics['r2']:.4f}")
        with col2:
            st.metric("Mean Squared Error", f"{metrics['mse']:.4f}")
    
    st.markdown("---")
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Enter Crop Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            crop = st.selectbox("Crop", sorted(df['Crop'].unique()))
            season = st.selectbox("Season", sorted(df['Season'].unique()))
            state = st.selectbox("State", sorted(df['State'].unique()))
            area = st.number_input("Area (hectares)", min_value=0.1, value=1.0, step=0.1)
        
        with col2:
            rainfall = st.number_input("Annual Rainfall (mm)", min_value=0, value=1000, step=10)
            fertilizer = st.number_input("Fertilizer (kg/hectare)", min_value=0, value=100, step=5)
            pesticide = st.number_input("Pesticide (kg/hectare)", min_value=0, value=10, step=1)
            
        submitted = st.form_submit_button("🎯 Predict Yield", use_container_width=True)
    
    if submitted:
        try:
            # Create input dataframe
            input_data = pd.DataFrame([{
                'Crop': crop,
                'Season': season,
                'State': normalize_state_name(state),
                'Area': area,
                'Annual_Rainfall': rainfall,
                'Fertilizer': fertilizer,
                'Pesticide': pesticide
            }])
            
            # Make prediction
            with st.spinner('Making prediction...'):
                prediction = model.predict(input_data)[0]
            
            # Display result
            st.markdown("---")
            st.markdown(f'<div class="prediction-result">Predicted Yield: {prediction:.2f} tonnes/hectare</div>', 
                       unsafe_allow_html=True)
            
            # Feature importance
            if hasattr(model.named_steps['regressor'], 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                
                importances = model.named_steps['regressor'].feature_importances_
                feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                
                # Aggregate feature importance
                importance_dict = {}
                for name, score in zip(feature_names, importances):
                    if '_' in name:
                        base_feature = name.split('_', 1)[0]
                    else:
                        base_feature = name
                    
                    base_feature = base_feature.replace('num__', '').replace('cat__', '')
                    importance_dict[base_feature] = importance_dict.get(base_feature, 0) + score
                
                # Normalize and display
                total = sum(importance_dict.values())
                importance_pct = {k: (v/total)*100 for k, v in importance_dict.items()}
                
                # Create bar chart
                fig = px.bar(x=list(importance_pct.values()), 
                           y=list(importance_pct.keys()),
                           orientation='h',
                           title='Feature Importance (%)')
                fig.update_layout(yaxis_title='Features', xaxis_title='Importance (%)')
                st.plotly_chart(fig, use_container_width=True)
            
            # Show similar records
            st.subheader("📋 Similar Records in Dataset")
            similar_records = df[
                (df['Crop'] == crop) & 
                (df['Season'] == season) & 
                (df['State'] == state)
            ].head(5)
            
            if not similar_records.empty:
                st.dataframe(similar_records, use_container_width=True)
            else:
                st.info("No similar records found in the dataset.")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

def show_about_page():
    """Display information about the app"""
    
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## 🌾 Crop Yield Prediction System
    
    This application uses machine learning to predict crop yields based on various agricultural parameters.
    
    ### 🎯 Features
    - **Data Analytics**: Comprehensive analysis of crop yield data
    - **Yield Prediction**: ML-powered predictions using Random Forest algorithm
    - **Interactive Visualizations**: Charts and graphs for better insights
    - **State-wise Analysis**: Detailed analysis for individual states
    
    ### 📊 Model Details
    - **Algorithm**: Random Forest Regressor
    - **Features**: Crop type, Season, State, Area, Rainfall, Fertilizer, Pesticide usage
    - **Preprocessing**: OneHot encoding for categorical variables, StandardScaler for numerical features
    
    ### 🔧 Technical Stack
    - **Frontend**: Streamlit
    - **Machine Learning**: Scikit-learn
    - **Visualizations**: Plotly
    - **Data Processing**: Pandas, NumPy
    
    ### 📝 Usage Instructions
    1. **Home**: Overview of the dataset and key metrics
    2. **Analytics**: Explore data through interactive visualizations
    3. **Prediction**: Input parameters to get yield predictions
    
    ### 📋 Dataset Requirements
    Your dataset should contain the following columns:
    - Crop, Season, State, Area, Annual_Rainfall, Fertilizer, Pesticide, Yield
    
    ### 🚀 Getting Started
    1. Ensure your `crop_yield.csv` file is in the app directory
    2. Run the Streamlit app
    3. Navigate through different pages using the sidebar
    
    """)

if __name__ == "__main__":
    main()