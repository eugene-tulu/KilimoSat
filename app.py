import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Geocoder
from shapely.geometry import shape
import geopandas as gpd
import pystac_client
import stackstac
import numpy as np
import xarray as xr
import rioxarray
from dask.diagnostics import ProgressBar
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
from datetime import datetime, timedelta
from folium import plugins
import warnings
import google.generativeai as genai
import pandas as pd
import africastalking
from scipy import ndimage
from sklearn.cluster import KMeans
import json
from dotenv import load_dotenv
import requests

warnings.filterwarnings("ignore")

# Constants
MAX_AREA_KM2 = 500
HEALTH_THRESHOLDS = {
    'excellent': 0.7,
    'good': 0.5,
    'moderate': 0.3,
    'poor': 0.1,
    'critical': 0.0
}

# Load Environment Variables
load_dotenv()

# Initialize Africa's Talking
africastalking.initialize(
    username="fadhil",  # Your Africa's Talking username
    api_key=os.getenv("AT_API_KEY")
)
sms = africastalking.SMS

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state with proper defaults
def initialize_session_state():
    """Initialize all session state variables with proper defaults"""
    if 'analysis_result' not in st.session_state:
        st.session_state['analysis_result'] = None
    if 'ai_summary' not in st.session_state:
        st.session_state['ai_summary'] = None
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = None
    if 'ndvi_history' not in st.session_state:
        st.session_state['ndvi_history'] = []
    if 'vegetation_indices' not in st.session_state:
        st.session_state['vegetation_indices'] = None
    if 'health_analysis' not in st.session_state:
        st.session_state['health_analysis'] = None
    if 'recommendations' not in st.session_state:
        st.session_state['recommendations'] = []

# Call initialization
initialize_session_state()

# Helper Functions
def send_sms_africastalking(phone_number, message):
    recipients = [phone_number]
    sender = "AFTKNG"
    print(f"Preparing to send SMS to {recipients} with sender {sender} and message: {message}")
    try:
        response = sms.send(message, recipients, sender)
        print("Africa's Talking response:", response)
        st.write("Africa's Talking response:", response)
        return True
    except africastalking.exceptions.AfricasTalkingGatewayException as e:
        print(f"Africa's Talking API error: {e}")
        st.error(f"Africa's Talking API error: {e}")
        return False
    except Exception as e:
        print(f"SMS error: {e}")
        st.error(f"SMS error: {e}")
        return False

def get_gemini_summary(stats):
    """
    Get AI summary using Gemini 2.0 API with improved error handling (Google SDK)
    """
    try:
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=(
                "You are an AI assistant for crop health monitoring. "
                "Given NDVI, EVI, NDRE, and MCARI values, briefly summarize crop health, vigor, nitrogen, and chlorophyll status "
                "in simple terms, suitable for SMS. Then, provide one or two clear, actionable recommendations or next steps for the farmer."
            )
        )
        prompt = (
            f"NDVI: Mean={stats['mean']:.3f}, Min={stats['min']:.3f}, Max={stats['max']:.3f}; "
            f"EVI: Mean={stats.get('evi_mean', 'N/A')}, NDRE: Mean={stats.get('ndre_mean', 'N/A')}, "
            f"MCARI: Mean={stats.get('mcari_mean', 'N/A')}. "
            "Give a simple summary of crop health, vigor, nitrogen, and chlorophyll status."
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=100,
                temperature=0.3,
            )
        )
        text = response.text.strip()
        if len(text) > 400:
            text = text[:397] + "..."
        return text
    except Exception as e:
        print(f"Gemini SDK error: {e}")
        st.error(f"Gemini SDK error: {e}")
        return f"NDVI Analysis: Mean={stats['mean']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}"

# App Configuration
st.set_page_config(layout="wide", page_title="🌾 KilimoSat")
st.title("🌾 KilimoSat - Smart Crop Health Monitoring")
st.markdown("""
Monitor crop health in real-time using satellite imagery  
Detect diseases, nutrient deficiencies, and get actionable insights for precision agriculture.
""")

# Sidebar Configuration
with st.sidebar:
    st.header("🛰 Monitoring Parameters")
    
    # Farm Details
    st.subheader("Farm Information")
    farm_name = st.text_input("Farm Name", value="My Farm")
    crop_type = st.selectbox("Crop Type", 
                           ["Maize/Corn", "Wheat", "Rice", "Soybeans", "Cotton", "Tomatoes", "Other"])
    
    # File Upload
    uploaded_file = st.file_uploader("Upload Field Boundary (GeoJSON)", type=["geojson", "json"])
    
    # Date Selection
    st.subheader("Analysis Period")
    end_date = st.date_input("End Date", value=datetime.now())
    start_date = st.date_input("Start Date", value=end_date - timedelta(days=90))
    
    # Analysis Parameters
    st.subheader("Analysis Settings")
    cloud_cover = st.slider("Max Cloud Cover (%)", 0, 100, 10)
    analysis_type = st.multiselect("Analysis Types", 
                                 ["Health Assessment", "Disease Detection", "Nutrient Analysis", "Stress Monitoring"],
                                 default=["Health Assessment", "Disease Detection"])
    
    # Alert Settings
    st.subheader("Alert Thresholds")
    health_threshold = st.slider("Health Alert Threshold", 0.0, 1.0, 0.4, 0.1)
    stress_threshold = st.slider("Stress Alert Threshold", 0.0, 1.0, 0.6, 0.1)
    
    run_analysis = st.button("🔍 Run Analysis", type="primary")
    phone = st.text_input("Enter your phone number(include country code):", "+254xxxxxxx")

    # Phone validation
    valid_phone = True
    if phone and (not phone.startswith('+') or not phone[1:].isdigit() or len(phone) < 10):
        st.error("Please enter a valid phone number with country code.")
        valid_phone = False

# Main Content Layout
col1, col2 = st.columns([2, 1])

with col1:
    # Interactive Map
    st.subheader("📍 Field Selection")
    m = folium.Map(
        location=[-1.3, 36.8],
        zoom_start=12,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=True
    )
    draw = plugins.Draw(export=True)
    m.add_child(draw)
    
    Geocoder(collapsed=False, add_marker=True).add_to(m)
    
    map_output = st_folium(m, width=700, height=400)

with col2:
    # Quick Stats Dashboard
    st.subheader("📊 Farm Overview")
    if st.session_state.get('analysis_results'):
        results = st.session_state.analysis_results
        
        # Health Score
        health_score = results.get('overall_health', 0.5)
        st.metric("Overall Health Score", f"{health_score:.2f}", 
                 delta=f"{results.get('health_change', 0):.2f}")
        
        # Area Coverage
        st.metric("Field Area", f"{results.get('area_ha', 0):.1f} ha")
        
    else:
        st.info("Run analysis to see farm statistics")

# Core Functions
@st.cache_data
def fetch_satellite_data(bounds, date_range, cloud_limit):
    """Fetch Sentinel-2 imagery from Planetary Computer"""
    try:
        client = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bounds,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": cloud_limit}}
        )
        items = list(search.get_items())
        return items
    except Exception as e:
        st.error(f"Error fetching satellite data: {e}")
        return []

def calculate_vegetation_indices(stack):
    """Calculate multiple vegetation indices for comprehensive analysis"""
    # Get bands
    nir = stack.sel(band="B08").astype(float)  # Near Infrared
    red = stack.sel(band="B04").astype(float)  # Red
    green = stack.sel(band="B03").astype(float)  # Green
    blue = stack.sel(band="B02").astype(float)  # Blue
    red_edge = stack.sel(band="B05").astype(float)  # Red Edge
    
    # Calculate indices
    indices = {}
    
    # NDVI - Overall vegetation health
    indices['ndvi'] = (nir - red) / (nir + red)
    
    # EVI - Enhanced Vegetation Index (better in dense vegetation)
    indices['evi'] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    
    # NDRE - Normalized Difference Red Edge (nitrogen stress)
    indices['ndre'] = (nir - red_edge) / (nir + red_edge)
    
    # GNDVI - Green NDVI (chlorophyll content)
    indices['gndvi'] = (nir - green) / (nir + green)
    
    # MCARI - Modified Chlorophyll Absorption Ratio Index
    indices['mcari'] = ((red_edge - red) - 0.2 * (red_edge - green)) * (red_edge / red)
    
    return indices

def detect_crop_health_issues(indices):
    """Analyze vegetation indices to detect health issues"""
    health_analysis = {}
    
    # Overall health assessment based on NDVI
    ndvi_values = indices['ndvi'].values
    health_analysis['overall_health'] = float(np.nanmean(ndvi_values))
    
    # Disease detection (sudden drops in NDVI)
    if st.session_state.ndvi_history:
        prev_ndvi = st.session_state.ndvi_history[-1]
        health_change = health_analysis['overall_health'] - prev_ndvi
        health_analysis['health_change'] = health_change
        
        if health_change < -0.1:
            health_analysis['disease_risk'] = 'High'
        elif health_change < -0.05:
            health_analysis['disease_risk'] = 'Medium'
        else:
            health_analysis['disease_risk'] = 'Low'
    else:
        health_analysis['disease_risk'] = 'Unknown'
        health_analysis['health_change'] = 0.0
    
    # Nutrient deficiency analysis
    ndre_mean = float(np.nanmean(indices['ndre'].values))
    if ndre_mean < 0.2:
        health_analysis['nitrogen_status'] = 'Deficient'
    elif ndre_mean < 0.3:
        health_analysis['nitrogen_status'] = 'Low'
    else:
        health_analysis['nitrogen_status'] = 'Adequate'
    
    # Chlorophyll assessment
    gndvi_mean = float(np.nanmean(indices['gndvi'].values))
    if gndvi_mean < 0.3:
        health_analysis['chlorophyll_status'] = 'Low'
    elif gndvi_mean < 0.5:
        health_analysis['chlorophyll_status'] = 'Moderate'
    else:
        health_analysis['chlorophyll_status'] = 'Good'
    
    # Water stress detection
    evi_mean = float(np.nanmean(indices['evi'].values))
    if evi_mean < 0.2:
        health_analysis['water_stress'] = 'High'
    elif evi_mean < 0.4:
        health_analysis['water_stress'] = 'Moderate'
    else:
        health_analysis['water_stress'] = 'Low'
    
    return health_analysis

def generate_recommendations(health_analysis, crop_type):
    """Generate actionable recommendations based on analysis"""
    recommendations = []
    
    # Disease recommendations
    if health_analysis.get('disease_risk') == 'High':
        recommendations.append({
            'category': 'Disease Management',
            'priority': 'High',
            'action': 'Immediate field inspection recommended. Consider fungicide application.',
            'icon': '🦠'
        })
    
    # Nutrient recommendations
    if health_analysis.get('nitrogen_status') == 'Deficient':
        recommendations.append({
            'category': 'Fertilization',
            'priority': 'Medium',
            'action': f'Apply nitrogen fertilizer suitable for {crop_type.lower()}. Consider soil testing.',
            'icon': '🌱'
        })
    
    # Water stress recommendations
    if health_analysis.get('water_stress') == 'High':
        recommendations.append({
            'category': 'Irrigation',
            'priority': 'High',
            'action': 'Increase irrigation frequency. Check soil moisture levels.',
            'icon': '💧'
        })
    
    # General health recommendations
    overall_health = health_analysis.get('overall_health', 0.5)
    if overall_health < 0.4:
        recommendations.append({
            'category': 'General Health',
            'priority': 'Medium',
            'action': 'Comprehensive field assessment needed. Consider crop protection measures.',
            'icon': '📋'
        })
    
    return recommendations

def create_health_visualization(indices, health_analysis):
    """Create comprehensive visualization of crop health"""
    
    # Main health map
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('NDVI (Overall Health)', 'Disease Risk Areas', 
                       'Nitrogen Status (NDRE)', 'Water Stress (EVI)'),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    # NDVI map
    fig.add_trace(
        go.Heatmap(z=indices['ndvi'].values, colorscale='RdYlGn', showscale=False),
        row=1, col=1
    )
    
    # Disease risk (NDVI anomalies)
    ndvi_smooth = ndimage.gaussian_filter(indices['ndvi'].values, sigma=1)
    disease_risk = indices['ndvi'].values - ndvi_smooth
    fig.add_trace(
        go.Heatmap(z=disease_risk, colorscale='Reds', showscale=False),
        row=1, col=2
    )
    
    # Nitrogen status
    fig.add_trace(
        go.Heatmap(z=indices['ndre'].values, colorscale='Blues', showscale=False),
        row=2, col=1
    )
    
    # Water stress
    fig.add_trace(
        go.Heatmap(z=indices['evi'].values, colorscale='YlOrRd', showscale=False),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Crop Health Analysis Dashboard",
        height=600,
        showlegend=False
    )
    
    return fig

# Main Analysis Workflow
if run_analysis:
    try:
        # Load field boundary
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp:
                tmp.write(uploaded_file.read())
                gdf = gpd.read_file(tmp.name)
                os.unlink(tmp.name)
        elif map_output.get("all_drawings"):
            geojson = map_output["all_drawings"][0]
            gdf = gpd.GeoDataFrame.from_features([geojson], crs="EPSG:4326")
        else:
            st.error("❌ Please upload a field boundary or draw one on the map.")
            st.stop()
        
        # Validate area
        area_km2 = gdf.to_crs("EPSG:6933").area[0] / 1e6
        area_ha = area_km2 * 100
        if area_km2 > MAX_AREA_KM2:
            st.error(f"❌ Field area ({area_km2:.1f} km²) exceeds maximum limit of {MAX_AREA_KM2} km²")
            st.stop()
        
        st.success(f"✅ Field loaded: {area_ha:.1f} hectares")
        
        # Fetch satellite data
        with st.spinner("🛰 Fetching satellite imagery..."):
            bounds = gdf.total_bounds.tolist()
            date_range = f"{start_date}/{end_date}"
            items = fetch_satellite_data(bounds, date_range, cloud_cover)
            
            if not items:
                st.error("❌ No satellite imagery found for the specified parameters.")
                st.stop()
            
            st.info(f"📡 Found {len(items)} satellite images")
        
        # Process imagery
        with st.spinner("🔄 Processing satellite data..."):
            import planetary_computer
            items = [planetary_computer.sign(item) for item in items]
            stack = stackstac.stack(
                items,
                assets=["B02", "B03", "B04", "B05", "B08"],
                resolution=10,
                epsg=6933,
                dtype="float",
                bounds_latlon=bounds
            )
            gdf_proj = gdf.to_crs(stack.rio.crs)
            aoi_geom = gdf_proj.geometry.unary_union
            clipped = stack.rio.clip([aoi_geom], crs=stack.rio.crs)
            with ProgressBar():
                clipped_computed = clipped.compute()
                indices = calculate_vegetation_indices(clipped_computed)
                for key in indices:
                    indices[key] = indices[key].median(dim="time")
        
            # Store indices in session state
            st.session_state['vegetation_indices'] = indices
            
            # Calculate statistics
            ndvi_stats = {
                "mean": float(np.nanmean(indices['ndvi'].values)),
                "min": float(np.nanmin(indices['ndvi'].values)),
                "max": float(np.nanmax(indices['ndvi'].values)),
                "evi_mean": float(np.nanmean(indices['evi'].values)),
                "ndre_mean": float(np.nanmean(indices['ndre'].values)),
                "mcari_mean": float(np.nanmean(indices['mcari'].values)),
            }
            ndvi_data = indices['ndvi'].values.tolist()
            
            # Store analysis results in session state
            st.session_state['analysis_result'] = {
                "statistics": ndvi_stats,
                "ndvi_data": ndvi_data
            }
        
        # Analyze crop health
        with st.spinner("🔬 Analyzing crop health..."):
            health_analysis = detect_crop_health_issues(indices)
            health_analysis['area_ha'] = area_ha
            health_analysis['crop_type'] = crop_type
            health_analysis['farm_name'] = farm_name
            
            # Store in session state
            st.session_state['health_analysis'] = health_analysis
            st.session_state['analysis_results'] = health_analysis
            
            # Update NDVI history
            st.session_state.ndvi_history.append(health_analysis['overall_health'])
            
            # Generate recommendations
            recommendations = generate_recommendations(health_analysis, crop_type)
            st.session_state['recommendations'] = recommendations
        
        # Display Results
        st.success("✅ Analysis completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)

# Display results if analysis has been run
if st.session_state.get('health_analysis'):
    health_analysis = st.session_state['health_analysis']
    
    st.subheader("🏥 Crop Health Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        health_score = health_analysis['overall_health']
        st.metric("Overall Health", f"{health_score:.3f}", delta=f"{health_analysis.get('health_change', 0):.3f}")

    with col2:
        st.metric("Disease Risk", health_analysis.get('disease_risk', 'Unknown'))

    with col3:
        st.metric("Nitrogen Status", health_analysis.get('nitrogen_status', 'Unknown'))

    with col4:
        st.metric("Water Stress", health_analysis.get('water_stress', 'Unknown'))

    # Visualization
    if st.session_state.get('vegetation_indices'):
        st.subheader("📊 Health Analysis Maps")
        health_viz = create_health_visualization(st.session_state['vegetation_indices'], health_analysis)
        st.plotly_chart(health_viz, use_container_width=True)

    # Recommendations
    st.subheader("💡 Actionable Recommendations")
    recommendations = st.session_state.get('recommendations', [])

    if recommendations:
        for rec in recommendations:
            st.markdown(f"""
            *{rec['icon']} {rec['category']}* - Priority: {rec['priority']}
            
            {rec['action']}
            """)
    else:
        st.success("🎉 No immediate actions needed. Crop health appears normal.")

    # Detailed Analysis
    with st.expander("📈 Detailed Analysis Results"):
        st.json(health_analysis)

    # Export Options
    st.subheader("📤 Export Results")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Download Report"):
            report_data = {
                'farm_name': health_analysis.get('farm_name', 'Unknown'),
                'analysis_date': datetime.now().isoformat(),
                'health_analysis': health_analysis,
                'recommendations': recommendations
            }
            st.download_button(
                "Download JSON Report",
                data=json.dumps(report_data, indent=2),
                file_name=f"{health_analysis.get('farm_name', 'farm')}health_report{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

    with col2:
        # AI SMS Feature - This section will now persist
        if st.session_state.get('analysis_result'):
            result = st.session_state['analysis_result']

            # Display stats
            st.subheader("NDVI Statistics")
            st.json(result["statistics"])

            # Plot
            st.subheader("NDVI Map")
            ndvi_array = np.array(result["ndvi_data"])
            fig = px.imshow(ndvi_array, color_continuous_scale='RdYlGn', origin='lower', title="NDVI Map")
            st.plotly_chart(fig, use_container_width=True)

            # AI SMS Feature
            st.markdown('#### Get AI Analysis via SMS')

            # Button to generate AI summary
            if st.button("Get AI Analysis Summary", key="generate_summary"):
                with st.spinner("Generating AI summary..."):
                    summary = get_gemini_summary(result["statistics"])
                    st.session_state['ai_summary'] = summary
                    st.success("AI summary generated!")

            # Display current AI summary if it exists
            if st.session_state.get('ai_summary'):
                st.info(f"*AI Summary:* {st.session_state['ai_summary']}")
                
                # SMS sending button - only show if phone number is valid
                if phone and valid_phone:
                    if st.button("Send AI Analysis via SMS", key="send_sms"):
                        with st.spinner("Sending SMS..."):
                            sms_status = send_sms_africastalking(phone, st.session_state["ai_summary"])
                            if sms_status:
                                st.success("✅ AI analysis sent via SMS!")
                            else:
                                st.error("❌ Failed to send SMS.")
                else:
                    st.warning("Please enter a valid phone number to send SMS.")

# Footer
st.markdown("---")
st.markdown("""
**🌾 KilimoSat** - Powered by satellite imagery
Built with Streamlit, Planetary Computer & STAC API
""")

# Sidebar footer with tips
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - *Best Results*: Use fields 1-100 hectares
    - *Cloud Cover*: Lower is better (<20%)
    - *Frequency*: Weekly monitoring recommended
    - *Growth Stage*: Most accurate during active growth
    """)