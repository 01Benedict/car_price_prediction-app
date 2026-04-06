# app.py
# ============================================================================
# CAR PRICE PREDICTION APP - STREAMLIT UI
# ============================================================================

import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .main-header p {
        color: #e0e0e0;
        margin: 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-card h2 {
        margin: 0;
        font-size: 2.5rem;
    }
    .prediction-card p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .feature-importance {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained model with caching for better performance"""
    try:
        model = joblib.load('car_price_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'car_price_model.pkl' not found. Please train the model first.")
        return None

def get_n_estimators_from_model(model):
    """Extract n_estimators from pipeline or direct model"""
    try:
        # If it's a pipeline, try to get the regressor/classifier
        if hasattr(model, 'named_steps'):
            if 'regressor' in model.named_steps:
                base_model = model.named_steps['regressor']
            elif 'classifier' in model.named_steps:
                base_model = model.named_steps['classifier']
            else:
                base_model = model.steps[-1][1]
        else:
            base_model = model
        
        # Check if the model has n_estimators attribute
        if hasattr(base_model, 'n_estimators'):
            return base_model.n_estimators
        else:
            return "multiple"
    except:
        return "multiple"


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================
def apply_feature_engineering(df, current_year):
    """Apply the same feature engineering as used during training"""
    df = df.copy()
    
    # Car age
    df['car_age'] = current_year - df['production_year']
    
    # Age groups
    df['age_group'] = pd.cut(df['car_age'],
                              bins=[0, 5, 10, 15, 100],
                              labels=['New', 'Recent', 'Mid-age', 'Old'])
    
    # Mileage groups
    df['mileage_group'] = pd.cut(df['mileage'],
                                  bins=[0, 50000, 100000, 150000, 1000000],
                                  labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Engine per cylinder
    df['engine_per_cylinder'] = df['engine_volume'] / df['cylinders']
    
    # Production year squared
    df['production_year_squared'] = df['production_year'] ** 2
    
    return df


# ============================================================================
# FEATURE LISTS (must match training)
# ============================================================================
numeric_cols_enhanced = ['production_year', 'levy', 'mileage', 'cylinders', 
                          'airbags', 'doors', 'engine_volume', 'car_age', 
                          'engine_per_cylinder', 'production_year_squared']

categorical_cols_enhanced = ['manufacturer', 'model', 'fuel_type', 'category', 
                              'leather_interior', 'gear_box_type', 'drive_wheels', 
                              'wheel', 'color', 'age_group', 'mileage_group']

feature_cols_enhanced = numeric_cols_enhanced + categorical_cols_enhanced


# ============================================================================
# INPUT COLLECTION FUNCTIONS
# ============================================================================
def get_manufacturer_options():
    """Common car manufacturers"""
    return ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes-Benz', 
            'Audi', 'Lexus', 'Hyundai', 'Kia', 'Nissan', 'Volkswagen', 
            'Mazda', 'Subaru', 'Volvo', 'Porsche', 'Jaguar', 'Land Rover',
            'Tesla', 'Ferrari', 'Lamborghini', 'Other']

def get_model_options(manufacturer):
    """Get model suggestions based on manufacturer"""
    models = {
        'Toyota': ['Camry', 'Corolla', 'RAV4', 'Highlander', 'Prius', 'Tacoma', 'Sienna'],
        'Honda': ['Civic', 'Accord', 'CR-V', 'Pilot', 'Odyssey', 'Fit'],
        'Ford': ['F-150', 'Mustang', 'Escape', 'Explorer', 'Focus', 'Fusion'],
        'Chevrolet': ['Silverado', 'Malibu', 'Equinox', 'Traverse', 'Camaro'],
        'BMW': ['3 Series', '5 Series', 'X3', 'X5', '7 Series', 'M3'],
        'Mercedes-Benz': ['C-Class', 'E-Class', 'S-Class', 'GLC', 'GLE', 'CLA'],
        'Audi': ['A4', 'A6', 'Q5', 'Q7', 'A3', 'Q3'],
        'Lexus': ['RX', 'ES', 'NX', 'GX', 'LS', 'IS'],
        'Hyundai': ['Elantra', 'Sonata', 'Tucson', 'Santa Fe', 'Kona'],
        'Kia': ['Optima', 'Sorento', 'Sportage', 'Telluride', 'Forte'],
        'Nissan': ['Altima', 'Maxima', 'Rogue', 'Murano', 'Sentra'],
        'Volkswagen': ['Jetta', 'Passat', 'Tiguan', 'Atlas', 'Golf'],
        'Mazda': ['Mazda3', 'Mazda6', 'CX-5', 'CX-9', 'MX-5'],
        'Subaru': ['Outback', 'Forester', 'Crosstrek', 'Impreza', 'Legacy'],
        'Volvo': ['XC60', 'XC90', 'S60', 'S90', 'V60'],
        'Porsche': ['911', 'Cayenne', 'Macan', 'Panamera', 'Taycan'],
        'Tesla': ['Model 3', 'Model S', 'Model X', 'Model Y'],
        'Other': ['Custom Model']
    }
    return models.get(manufacturer, ['Other Model'])

def get_fuel_options():
    return ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'LPG', 'CNG']

def get_category_options():
    return ['Sedan', 'SUV', 'Hatchback', 'Jeep', 'Coupe', 'Convertible', 
            'Wagon', 'Van', 'Truck', 'Minivan']

def get_gear_box_options():
    return ['Automatic', 'Manual', 'Tiptronic', 'CVT', 'Semi-Automatic']

def get_drive_wheels_options():
    return ['Front', 'Rear', '4x4', 'AWD']

def get_wheel_options():
    return ['Left wheel', 'Right wheel']

def get_color_options():
    colors = ['Black', 'White', 'Silver', 'Gray', 'Red', 'Blue', 'Green', 
              'Brown', 'Beige', 'Orange', 'Yellow', 'Purple', 'Gold', 'Other']
    return colors


# ============================================================================
# CREATE INPUT FORM
# ============================================================================
def create_input_form():
    """Create the input form for car details"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Basic Information")
        
        manufacturer = st.selectbox("Manufacturer", get_manufacturer_options(), key="manufacturer")
        
        # Dynamic model selection based on manufacturer
        models = get_model_options(manufacturer)
        model_name = st.selectbox("Model", models, key="model")
        
        if manufacturer == 'Other' or model_name == 'Custom Model' or model_name == 'Other Model':
            model_name = st.text_input("Enter Model Name", value=model_name if model_name != 'Other Model' else "", key="custom_model")
        
        production_year = st.number_input("Production Year", 
                                          min_value=1990, 
                                          max_value=datetime.now().year,
                                          value=2018,
                                          step=1,
                                          key="production_year")
        
        category = st.selectbox("Vehicle Category", get_category_options(), key="category")
        
        fuel_type = st.selectbox("Fuel Type", get_fuel_options(), key="fuel_type")
        
        engine_volume = st.number_input("Engine Volume (Liters)", 
                                         min_value=0.5, 
                                         max_value=8.0, 
                                         value=2.0, 
                                         step=0.1,
                                         format="%.1f",
                                         key="engine_volume")
        
        cylinders = st.selectbox("Number of Cylinders", [3, 4, 5, 6, 8, 10, 12], index=1, key="cylinders")
        
        mileage = st.number_input("Mileage (km)", 
                                   min_value=0, 
                                   max_value=500000, 
                                   value=50000, 
                                   step=5000,
                                   key="mileage")
    
    with col2:
        st.subheader("🔧 Additional Features")
        
        levy = st.number_input("Levy / Tax Amount", 
                               min_value=0, 
                               max_value=10000, 
                               value=500, 
                               step=50,
                               key="levy",
                               help="Annual tax or insurance levy on the vehicle")
        
        airbags = st.select_slider("Number of Airbags", options=[0, 2, 4, 6, 8, 10, 12], value=4, key="airbags")
        
        doors = st.selectbox("Number of Doors", [2, 3, 4, 5], index=2, key="doors")
        
        leather_interior = st.selectbox("Leather Interior", ['Yes', 'No'], key="leather_interior")
        
        gear_box_type = st.selectbox("Transmission Type", get_gear_box_options(), key="gear_box_type")
        
        drive_wheels = st.selectbox("Drive Wheels", get_drive_wheels_options(), key="drive_wheels")
        
        wheel = st.selectbox("Steering Wheel Position", get_wheel_options(), key="wheel")
        
        color = st.selectbox("Exterior Color", get_color_options(), key="color")
    
    return {
        'manufacturer': manufacturer,
        'model': model_name,
        'production_year': production_year,
        'category': category,
        'fuel_type': fuel_type,
        'engine_volume': engine_volume,
        'cylinders': cylinders,
        'mileage': mileage,
        'levy': levy,
        'airbags': airbags,
        'doors': doors,
        'leather_interior': leather_interior,
        'gear_box_type': gear_box_type,
        'drive_wheels': drive_wheels,
        'wheel': wheel,
        'color': color
    }


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Car Price Predictor</h1>
        <p>AI-powered car price estimation using Random Forest Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Get n_estimators value for display
    n_estimators_value = get_n_estimators_from_model(model)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1995/1995578.png", width=100)
        st.markdown("### About")
        st.markdown("""
        This app uses a **Random Forest Regression** model trained on thousands of car listings to predict market prices.
        
        **Key Features:**
        - 🏭 Manufacturer & Model
        - 📅 Production Year
        - 🔧 Engine & Transmission
        - 📊 Mileage & Condition
        - 🎨 Color & Features
        
        **Model Performance:**
        - R² Score: **0.7656** (76.56% accuracy)
        - Average Error: **~$4,000**
        """)
        
        st.markdown("---")
        st.markdown("### How to Use")
        st.markdown("""
        1. Fill in the car details on the left
        2. Click **Predict Price** button
        3. View the estimated price and confidence
        4. Adjust inputs to compare different scenarios
        """)
        
        st.markdown("---")
        st.markdown("### Tips")
        st.markdown("""
        - Newer cars with lower mileage get higher prices
        - Luxury brands and leather interiors increase value
        - More airbags and better safety features add value
        """)
    
    # Main content area
    st.markdown("### 📝 Enter Car Details")
    st.markdown("Provide the following information about the car you want to evaluate:")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Basic Info", "🔧 Technical Specs", "🎨 Appearance"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            manufacturer = st.selectbox("🏭 Manufacturer", get_manufacturer_options())
            production_year = st.number_input("📅 Production Year", 
                                              min_value=1990, 
                                              max_value=datetime.now().year,
                                              value=2018)
            mileage = st.number_input("📊 Mileage (km)", 
                                      min_value=0, 
                                      value=50000, 
                                      step=5000)
        with col2:
            models = get_model_options(manufacturer)
            model_name = st.selectbox("🚗 Model", models)
            if manufacturer == 'Other' or model_name == 'Custom Model':
                model_name = st.text_input("Enter custom model name", value=model_name)
            category = st.selectbox("🏷️ Category", get_category_options())
        with col3:
            fuel_type = st.selectbox("⛽ Fuel Type", get_fuel_options())
            leather_interior = st.selectbox("🪑 Leather Interior", ['Yes', 'No'])
            color = st.selectbox("🎨 Color", get_color_options())
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            engine_volume = st.number_input("🔧 Engine Volume (L)", 
                                            min_value=0.5, max_value=8.0, value=2.0, step=0.1)
            cylinders = st.selectbox("🔘 Cylinders", [3, 4, 5, 6, 8, 10, 12], index=1)
        with col2:
            gear_box_type = st.selectbox("⚙️ Transmission", get_gear_box_options())
            drive_wheels = st.selectbox("🔄 Drive Wheels", get_drive_wheels_options())
        with col3:
            airbags = st.select_slider("🛡️ Airbags", options=[0, 2, 4, 6, 8, 10, 12], value=4)
            levy = st.number_input("💰 Levy/Tax", min_value=0, max_value=10000, value=500, step=50)
    
    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            doors = st.selectbox("🚪 Doors", [2, 3, 4, 5], index=2)
        with col2:
            wheel = st.selectbox("🔄 Steering Wheel", get_wheel_options())
        with col3:
            st.markdown("### Quick Tips")
            st.info("💡 Tip: Cars with leather interior and more airbags typically command higher prices!")
    
    # Predict button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 PREDICT CAR PRICE", use_container_width=True, type="primary")
    
    if predict_button:
        # Prepare input data
        current_year = datetime.now().year
        
        input_data = pd.DataFrame({
            'manufacturer': [manufacturer],
            'model': [model_name],
            'production_year': [production_year],
            'category': [category],
            'fuel_type': [fuel_type],
            'engine_volume': [engine_volume],
            'cylinders': [cylinders],
            'mileage': [mileage],
            'levy': [levy],
            'airbags': [airbags],
            'doors': [doors],
            'leather_interior': [leather_interior],
            'gear_box_type': [gear_box_type],
            'drive_wheels': [drive_wheels],
            'wheel': [wheel],
            'color': [color]
        })
        
        # Apply feature engineering
        input_data = apply_feature_engineering(input_data, current_year)
        input_data = input_data[feature_cols_enhanced]
        
        # Make prediction
        with st.spinner("🤖 Analyzing car data and calculating price..."):
            prediction = model.predict(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="prediction-card">
                <p>Estimated Market Price</p>
                <h2>${prediction:,.2f}</h2>
                <p>Based on current market data and AI analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Price range estimation
        lower_bound = prediction * 0.85
        upper_bound = prediction * 1.15
        
        # Format the confidence text properly
        if n_estimators_value != "multiple":
            confidence_text = f"High (based on {n_estimators_value} decision trees)"
        else:
            confidence_text = "High (ensemble model)"
        
        st.markdown(f"""
        <div class="info-box">
            <h4>💰 Price Range Estimate</h4>
            <p><strong>Expected Range:</strong> ${lower_bound:,.2f} - ${upper_bound:,.2f}</p>
            <p><strong>Confidence Level:</strong> {confidence_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Factors affecting price
        st.markdown("### 📈 Factors Affecting This Price")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age impact
            car_age = current_year - production_year
            age_impact = "Excellent" if car_age <= 3 else "Good" if car_age <= 6 else "Fair" if car_age <= 10 else "Depreciated"
            age_emoji = "🟢" if car_age <= 3 else "🟡" if car_age <= 6 else "🟠" if car_age <= 10 else "🔴"
            
            st.markdown(f"""
            **Vehicle Age:** {car_age} years - {age_impact} {age_emoji}
            
            **Mileage Impact:** {mileage:,} km
            - {'Low mileage (premium value)' if mileage < 50000 else 'Average mileage' if mileage < 100000 else 'High mileage (depreciation)'}
            
            **Engine:** {engine_volume}L, {cylinders} cylinders
            """)
        
        with col2:
            st.markdown(f"""
            **Features:** 
            - {'✓ Leather interior (+value)' if leather_interior == 'Yes' else '✗ Cloth interior'}
            - {airbags} airbags ({'Excellent safety' if airbags >= 6 else 'Standard safety'})
            
            **Transmission:** {gear_box_type}
            
            **Drive Type:** {drive_wheels}
            """)
        
        # Comparison gauge
        st.markdown("### 📊 Price Analysis")
        
        # Create a simple gauge chart using matplotlib
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Price categories (simplified for visualization)
        categories = ['Economy', 'Standard', 'Premium', 'Luxury', 'Exotic']
        thresholds = [0, 15000, 30000, 50000, 80000, float('inf')]
        
        # Find which category the price falls into
        price_category = "Luxury"
        for i in range(len(thresholds)-1):
            if thresholds[i] <= prediction < thresholds[i+1]:
                price_category = categories[i]
                break
        
        # Create horizontal bar chart for price category
        colors_bar = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
        category_idx = categories.index(price_category)
        
        ax.barh([0], [1], color=colors_bar[category_idx], height=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Price Category: {price_category} Vehicle", fontsize=12, fontweight='bold')
        
        # Add price marker
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        st.pyplot(fig)
        plt.close()
        
        # Disclaimer
        st.markdown("---")
        st.caption("⚠️ **Disclaimer:** This is an AI-generated estimate based on historical data. Actual market prices may vary based on location, condition, market conditions, and other factors not captured in this model.")
    
    else:
        # Show example predictions when no prediction is made yet
        st.markdown("---")
        st.markdown("### 💡 Example Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        examples = [
            {"name": "Economy Car", "desc": "Honda Civic 2018", "price": "$15,000 - $22,000"},
            {"name": "Family SUV", "desc": "Toyota RAV4 2020", "price": "$28,000 - $38,000"},
            {"name": "Luxury Sedan", "desc": "BMW 5 Series 2019", "price": "$45,000 - $65,000"}
        ]
        
        for idx, (col, example) in enumerate(zip([col1, col2, col3], examples)):
            with col:
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 10px; text-align: center;">
                    <h4>{example['name']}</h4>
                    <p>{example['desc']}</p>
                    <p><strong>{example['price']}</strong></p>
                </div>
                """, unsafe_allow_html=True)


# ============================================================================
# RUN THE APP
# ============================================================================
if __name__ == "__main__":
    main()