import pandas as pd
from supabase import create_client
import streamlit as st

class AttendanceService:
    def __init__(self, url, key):
        self.supabase = create_client(url, key)
        self.table = "attendance_records"

    def fetch_records(self):
        try:
            # Fetching based on your verified table name
            res = self.supabase.table(self.table).select("*").order("timestamp", desc=True).execute()
            
            if not res.data:
                return pd.DataFrame()
                
            df = pd.DataFrame(res.data)
            return self._normalize(df)
        except Exception as e:
            st.error(f"📡 Database Connection Error: {str(e)}")
            return pd.DataFrame()

    def _normalize(self, df):
        """Standardizes columns based on your specific schema"""
        # Mapping 'timestamp' from your screenshot to 'created_at' for the UI logic
        if 'timestamp' in df.columns:
            df['created_at'] = pd.to_datetime(df['timestamp'])
        
        # Ensure 'verified' is treated as string for the success checks
        if 'verified' in df.columns:
            df['verified'] = df['verified'].fillna('unknown').astype(str)
            
        return df