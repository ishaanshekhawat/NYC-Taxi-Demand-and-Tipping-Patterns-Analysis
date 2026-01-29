
import streamlit as st
import pandas as pd
import datetime

# Page configuration
st.set_page_config(page_title="NYC Taxi Nearby Hotspots", layout="wide")

# Load data
@st.cache_data
def load_data():
    # Load Zone Lookup
    try:
        zone_lookup = pd.read_parquet("zone_lookup")
    except Exception:
        # Fallback if it's a CSV or different path, but file list showed parquet
        # Trying to read the directory as parquet dataset
        zone_lookup = pd.read_parquet("zone_lookup")
        
    # Load Location Probabilities
    location_probs = pd.read_parquet("location_probabilities_in_borough")
    
    return zone_lookup, location_probs

try:
    zone_lookup, location_probs = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- Preprocessing Zone Data ---
# Filter out EWR, Unknown, N/A as per notebook inspiration
zone_lookup = zone_lookup[~zone_lookup['Borough'].isin(['EWR', 'Unknown', 'N/A'])]
# Create a display label: "Zone - Borough"
zone_lookup['Display Label'] = zone_lookup['Zone'] + " - " + zone_lookup['Borough']
zone_map = zone_lookup.set_index('LocationID')['Display Label'].to_dict()
location_to_borough = zone_lookup.set_index('LocationID')['Borough'].to_dict()

# --- Sidebar Inputs ---
st.sidebar.header("Driver Input")

# Location Input
# Create a sorted list of options for the dropdown
location_options = zone_lookup[['LocationID', 'Display Label']].sort_values('Display Label')
selected_label = st.sidebar.selectbox(
    "Current Location",
    location_options['Display Label']
)

# Get selected LocationID
selected_location_id = location_options[location_options['Display Label'] == selected_label]['LocationID'].values[0]

# Time Input - defaulting to current hour
current_sys_hour = datetime.datetime.now().hour
selected_hour = st.sidebar.slider("Current Hour (0-23)", 0, 23, current_sys_hour)

# --- Main Content ---
st.title("🚖 NYC Taxi Nearby Hotspots")
st.markdown(f"**Current Location:** {selected_label}")
st.markdown(f"**Target Time:** {selected_hour}:00")

# --- Logic inspired by 'Nearby Hotspots.ipynb' ---

def get_recommendations(current_loc_id, hour, top_n=5):
    # 1. Identify Borough
    borough = location_to_borough.get(current_loc_id)
    if not borough:
        st.error("Current location borough not found.")
        return None

    # 2. Filter probabilities for same borough and hour
    # Columns in location_probs: PULocationID, PU_Borough, pickup_hour, pickup_probability_in_borough_pct, avg_fare, avg_tip_percent, ...
    
    candidates = location_probs[
        (location_probs['PU_Borough'] == borough) & 
        (location_probs['pickup_hour'] == hour) &
        (location_probs['PULocationID'] != current_loc_id) # Exclude current location
    ].copy()
    
    if candidates.empty:
        return pd.DataFrame()

    # 3. Get Top 5 by Probability
    # "gives the top 5 nearby locations with the highest probability for finding next ride"
    top_candidates = candidates.sort_values(by='pickup_probability_in_borough_pct', ascending=False).head(top_n)
    
    # 4. Order by Estimated Fare, then Tip
    # "suggested hotspots will be ordered by estimated fare amount, then by tip amount"
    
    # Calculate estimated tip amount if not present, for sorting purposes?
    # Dataset has 'avg_tip_percent'. Let's calculate 'avg_tip_amount' proxy = avg_fare * avg_tip_percent / 100
    top_candidates['estimated_tip_amount'] = top_candidates['avg_fare'] * (top_candidates['avg_tip_percent'] / 100)
    
    # Sort
    # "ordered by estimated fare amount, then by tip amount"
    top_candidates = top_candidates.sort_values(
        by=['avg_fare', 'estimated_tip_amount'], 
        ascending=[False, False]
    )
    
    return top_candidates

recommendations = get_recommendations(selected_location_id, selected_hour)

if recommendations is not None and not recommendations.empty:
    st.subheader(f"Top {len(recommendations)} Recommendations")
    
    # Format for display
    # Join with Zone Names
    display_df = recommendations.merge(
        zone_lookup[['LocationID', 'Zone']], 
        left_on='PULocationID', 
        right_on='LocationID', 
        how='left'
    )
    
    # Select and Rename columns
    display_df = display_df[[
        'Zone', 
        'pickup_probability_in_borough_pct', 
        'avg_fare', 
        'estimated_tip_amount'
    ]]
    
    display_df.columns = [
        'Hotspot Zone', 
        'Probability (%)', 
        'Est. Fare ($)', 
        'Est. Tip ($)'
    ]
    
    # Formatting values
    display_df['Probability (%)'] = display_df['Probability (%)'].map('{:.2f}%'.format)
    display_df['Est. Fare ($)'] = display_df['Est. Fare ($)'].map('${:.2f}'.format)
    display_df['Est. Tip ($)'] = display_df['Est. Tip ($)'].map('${:.2f}'.format)

    # Display as a table (or stylized cards)
    st.table(display_df.reset_index(drop=True))
    
    # Optional: Display metrics using st.metric for the top 1 choice
    top_choice = display_df.iloc[0]
    st.info(f"🌟 Best Hotspot: **{top_choice['Hotspot Zone']}**")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Probability", top_choice['Probability (%)'])
    col2.metric("Est. Fare", top_choice['Est. Fare ($)'])
    col3.metric("Est. Tip", top_choice['Est. Tip ($)'])

elif recommendations is not None:
    st.warning("No nearby hotspots found for this criterion.")

