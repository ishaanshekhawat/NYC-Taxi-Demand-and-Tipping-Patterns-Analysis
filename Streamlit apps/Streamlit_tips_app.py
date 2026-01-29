import streamlit as st
import pandas as pd
import os

# Set page config
st.set_page_config(page_title="NYC Taxi Tip predictor", layout="wide")

st.title("NYC Taxi Tip Estimator")
st.markdown("Select your trip details to see historical average tip metrics.")

# Load Data
@st.cache_data
def load_data():
    tips_path = "streamlit_tips_dataset"
    zone_path = "zone_lookup"
    
    if not os.path.exists(tips_path) or not os.path.exists(zone_path):
        st.error(f"Dataset files not found in current directory. Expected '{tips_path}' and '{zone_path}'.")
        return None, None

    tips_df = pd.read_parquet(tips_path)
    zone_df = pd.read_parquet(zone_path)
    return tips_df, zone_df

tips_df, zone_df = load_data()

def get_aggregated_metrics(df):
    """Calculate weighted averages for the dataframe."""
    if df.empty:
        return None
    
    total_rides = df['rides'].sum()
    if total_rides == 0:
        return None
        
    avg_tip_amt = (df['avg_tip_amount'] * df['rides']).sum() / total_rides
    avg_tip_pct = (df['avg_tip_percent'] * df['rides']).sum() / total_rides
    avg_fare = (df['avg_fare_amount'] * df['rides']).sum() / total_rides
    
    return {
        'avg_tip_amount': avg_tip_amt,
        'avg_tip_percent': avg_tip_pct,
        'avg_fare_amount': avg_fare,
        'rides': total_rides
    }

if tips_df is not None and zone_df is not None:
    # Prepare Zone Lookup Dictionary
    zone_df['Label'] = zone_df['Borough'] + " - " + zone_df['Zone']
    zone_map = pd.Series(zone_df.LocationID.values, index=zone_df.Label).to_dict()
    sorted_labels = sorted(zone_map.keys())

    # --- User Inputs ---
    col1, col2 = st.columns(2)
    
    with col1:
        pickup_label = st.selectbox("Pickup Location", options=sorted_labels)
        pickup_id = zone_map[pickup_label]
        
        pickup_month = st.number_input("Pickup Month (1-12)", min_value=1, max_value=12, value=1, step=1)
        
        is_weekend_input = st.radio("Is it a Weekend?", options=["No", "Yes"])
        is_weekend = 1 if is_weekend_input == "Yes" else 0

    with col2:
        dropoff_label = st.selectbox("Dropoff Location", options=sorted_labels)
        dropoff_id = zone_map[dropoff_label]
        
        pickup_hour = st.slider("Pickup Hour (0-23)", min_value=0, max_value=23, value=12, step=1)

    st.divider()

    # --- Filter Logic with Fallback ---
    # 1. Exact Match
    mask_base = (
        (tips_df['PULocationID'] == pickup_id) &
        (tips_df['DOLocationID'] == dropoff_id) &
        (tips_df['is_weekend'] == is_weekend)
    )
    
    match_type = "Exact Match"
    filtered_df = tips_df[
        mask_base &
        (tips_df['pickup_hour'] == pickup_hour) &
        (tips_df['pickup_month'] == pickup_month)
    ]
    
    # 2. Fallback: All Hours (Specific Month)
    if filtered_df.empty:
        match_type = "Average for Selected Month (All Hours)"
        filtered_df = tips_df[
            mask_base & 
            (tips_df['pickup_month'] == pickup_month)
        ]
        
    # 3. Fallback: All Months (All Hours)
    if filtered_df.empty:
        match_type = "Average for All Months (All Hours)"
        filtered_df = tips_df[mask_base]
        
    # 4. Fallback: All Weekend Statuses (All Months, All Hours)
    if filtered_df.empty:
        match_type = "Average for All Weekends (All Months, All Hours)"
        mask_route_only = (
            (tips_df['PULocationID'] == pickup_id) &
            (tips_df['DOLocationID'] == dropoff_id)
        )
        filtered_df = tips_df[mask_route_only]

    # --- Display Results ---
    if not filtered_df.empty:
        metrics = get_aggregated_metrics(filtered_df)
        
        if metrics:
            st.subheader("Historical Data Found")
            
            if match_type != "Exact Match":
                st.info(f"Note: Specific data not found. Showing **{match_type}**.")
            else:
                st.success(f"Showing **{match_type}**.")

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            

            m_col1.metric("Avg Tip Amount", f"${metrics['avg_tip_amount']:.2f}")
            m_col2.metric("Avg Tip Percent", f"{metrics['avg_tip_percent']:.2f}%")
            m_col3.metric("Avg Fare Amount", f"${metrics['avg_fare_amount']:.2f}")
            m_col4.metric("Number of Rides", int(metrics['rides']))

            st.divider()
            st.subheader("Ride Trends for this Route")
            st.markdown(f"Metrics based on travel between **{pickup_label}** and **{dropoff_label}**.")

            # --- Data for Visualizations ---
            # 1. Hourly Trend (Match Route & Weekend Status, Aggregate over Month)
            trend_df = tips_df[mask_base] # Reuse route+weekend mask
            
            if not trend_df.empty:
                # Group by Hour
                hourly_data = trend_df.groupby('pickup_hour').apply(
                    lambda x: pd.Series({
                        'Avg Tip %': (x['avg_tip_percent'] * x['rides']).sum() / x['rides'].sum() if x['rides'].sum() > 0 else 0,
                        'Rides': x['rides'].sum()
                    }),
                    include_groups=False
                ).reset_index()

                # Group by Month
                monthly_data = trend_df.groupby('pickup_month').apply(
                    lambda x: pd.Series({
                        'Avg Tip %': (x['avg_tip_percent'] * x['rides']).sum() / x['rides'].sum() if x['rides'].sum() > 0 else 0
                    }),
                    include_groups=False
                ).reset_index()
                
                # Charts
                tab1, tab2 = st.tabs(["Hourly Tip Trend", "Monthly Tip Trend"])
                
                with tab1:
                    st.line_chart(hourly_data.set_index('pickup_hour')['Avg Tip %'], color="#29b5e8")
                    st.caption("Average Tip Percentage by Hour of Day")
                    
                with tab2:
                    st.line_chart(monthly_data.set_index('pickup_month')['Avg Tip %'], color="#FF6C6C")
                    st.caption("Average Tip Percentage by Month")
            else:
                st.info("Not enough data to generate trends for this specific route and weekend status.")
    else:
        st.warning("No historical data found even after averaging over hours, months, and weekend status.")
        st.info("This route combination (Pickup <-> Dropoff) has no recorded rides in the dataset.")
