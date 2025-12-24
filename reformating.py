import pandas as pd
import json
import os

# 1. Load the raw JSON file
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, 'Хронология.json')
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

moves_data = []

# 2. Extract features from the nested 'semanticSegments'
for segment in data.get('semanticSegments', []):
    
    # We focus on 'activity' (moves) which contains physics data
    if 'activity' in segment:
        activity = segment['activity']
        act_type = activity.get('topCandidate', {}).get('type', 'UNKNOWN')

        if act_type in  ['UNKNOWN_ACTIVITY_TYPE', 'IN_FERRY','FLYING']:
            continue

        if act_type in ['IN_SUBWAY', 'IN_TRAIN', 'IN_TRAM']:
                act_type = 'RAIL'
        elif act_type == 'IN_PASSENGER_VEHICLE':
                act_type = 'CAR'
        elif act_type == 'IN_BUS':
                act_type = 'BUS'        

        # Get standardized time
        start = pd.to_datetime(segment['startTime'])
        end = pd.to_datetime(segment['endTime'])
        duration = (end - start).total_seconds() / 60
        
        # Extract distance and type
        dist = activity.get('distanceMeters', 0)
        speed_kmh= dist / duration

        if dist > 1000000:
            continue
        if duration > 1440: 
                continue
        if speed_kmh > 2500:
                continue

        time_period = ''

        if 6 <= start.hour <= 11:
            time_period = 'Morning'
        elif 12 <= start.hour <= 17:
            time_period = 'Afternoon'
        elif 18 <= start.hour <= 23:
            time_period = 'Evening'
        else: # 00:00 - 05:00 arası
            time_period = 'Night'

        """if dist > 18000000: 
            print(f"\n!!! FOUND IT at Index: {index} !!!")
            print(f"Exact Value in JSON: {dist}")
            print(f"Start Time: {segment.get('startTime')}")
            print(f"Activity Type: {segment['activity'].get('topCandidate', {}).get('type')}")
            
            # Print coordinates to see if it teleported
            start_loc = segment['activity'].get('start', {}).get('latLng')
            end_loc = segment['activity'].get('end', {}).get('latLng')
            print(f"Coordinates: {start_loc} -> {end_loc}")
            break"""
        
        
        moves_data.append({
            'Activity': act_type,
            'Distance_m': dist,
            'Duration_min': duration,
            'Speed_m_min': speed_kmh if duration > 0 else 0,
            'Time_Period': time_period,
            'Day': start.day_name()
        })

# 3. Create the Mining-Ready DataFrame
df = pd.DataFrame(moves_data)
print(df.head())

# 4. (Optional) Save for use in Weka/Sklearn
df.to_csv('my_location_features.csv', index=False)