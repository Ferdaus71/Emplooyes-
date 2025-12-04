import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import tempfile
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import base64
import io
import sys
import json

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.sidebar.warning("⚠️ OpenCV not available - video features limited")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.sidebar.warning("⚠️ PyTorch not available - RL features limited")

# ============================================
# 1. REINFORCEMENT LEARNING MODEL (SIMPLIFIED)
# ============================================

class RuleBasedAgent:
    """Rule-based agent that simulates RL decisions"""
    def __init__(self):
        self.actions = [
            "No action needed",
            "Provide positive reinforcement",
            "Schedule wellness check",
            "Offer skill training",
            "Adjust workload",
            "Recommend promotion track",
            "Provide constructive feedback"
        ]
    
    def select_action(self, state, epsilon=0.1):
        """Rule-based action selection"""
        # Extract features from state
        if len(state) >= 2:
            work_hours = state[0] * 24  # Convert normalized hours back
            task_complexity = state[1]
        else:
            work_hours = 8
            task_complexity = 0.5
        
        # Simple rule-based logic
        if work_hours > 10:
            return 2  # Wellness check for long hours
        elif work_hours < 6:
            return 4  # Adjust workload for short hours
        elif task_complexity > 0.8:
            return 3  # Skill training for complex tasks
        elif task_complexity > 0.6:
            return 1  # Positive reinforcement
        else:
            return 0  # No action needed
    
    def get_action_description(self, action_idx):
        if 0 <= action_idx < len(self.actions):
            return self.actions[action_idx]
        return "Unknown action"

# ============================================
# 2. DATA MANAGEMENT AND PROCESSING
# ============================================

class EmployeeDatabase:
    def __init__(self):
        self.df = None
        self.employee_data = {}
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load dataset from CSV file"""
        try:
            if os.path.exists("employee_multimodal_dataset.csv"):
                self.df = pd.read_csv("employee_multimodal_dataset.csv")
                self.process_dataframe()
            else:
                self.df = pd.DataFrame(columns=[
                    'Date', 'Team Members', 'Signed In', 'Signed Out', 
                    'Completed Task', 'session_id', 'selfie_path', 
                    'session_video_path', 'emotion_label', 'engagement_level',
                    'posture_label', 'performance_label'
                ])
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            self.df = pd.DataFrame()
    
    def process_dataframe(self):
        """Clean and process the dataframe"""
        if self.df is not None and not self.df.empty:
            # Fill missing columns
            required_columns = ['Team Members', 'Signed In', 'Signed Out', 'Completed Task']
            for col in required_columns:
                if col not in self.df.columns:
                    self.df[col] = ''
            
            # Fill NaN values
            self.df = self.df.fillna('')
            
            # Build employee data dictionary
            self.employee_data = {}
            for idx, row in self.df.iterrows():
                employee_name = str(row.get('Team Members', '')).strip()
                if employee_name:
                    self.employee_data[employee_name] = {
                        'sign_in': row.get('Signed In', ''),
                        'sign_out': row.get('Signed Out', ''),
                        'task': row.get('Completed Task', ''),
                        'date': row.get('Date', ''),
                        'selfie_path': row.get('selfie_path', ''),
                        'video_path': row.get('session_video_path', '')
                    }
    
    def upload_dataset(self, uploaded_file):
        """Handle CSV file upload"""
        if uploaded_file is None:
            return "No file uploaded", pd.DataFrame()
        
        try:
            self.df = pd.read_csv(uploaded_file)
            self.process_dataframe()
            return f"✅ Uploaded {len(self.df)} records", self.df
        except Exception as e:
            return f"❌ Error: {str(e)}", pd.DataFrame()
    
    def search_employees(self, search_term):
        """Search for employees by name"""
        if not search_term or self.df is None or self.df.empty:
            return []
        
        search_term = search_term.lower().strip()
        matches = []
        
        if 'Team Members' in self.df.columns:
            for idx, row in self.df.iterrows():
                employee_name = str(row['Team Members']).strip()
                if employee_name and search_term in employee_name.lower():
                    matches.append(employee_name)
        
        return list(dict.fromkeys(matches))[:10]  # Remove duplicates, limit to 10
    
    def get_employee_details(self, employee_name):
        """Get details for a specific employee"""
        if employee_name in self.employee_data:
            details = self.employee_data[employee_name]
            return (
                details.get('sign_in', ''),
                details.get('sign_out', ''),
                details.get('task', ''),
                details.get('date', '')
            )
        return '', '', '', ''
    
    def save_record(self, employee_data):
        """Save a new employee record"""
        try:
            # Create new record
            new_record = {
                'Date': employee_data.get('date', datetime.now().strftime('%d.%m.%Y')),
                'Team Members': employee_data.get('name', ''),
                'Signed In': employee_data.get('sign_in', ''),
                'Signed Out': employee_data.get('sign_out', ''),
                'Completed Task': employee_data.get('task', ''),
                'session_id': len(self.df) + 1 if not self.df.empty else 1,
                'selfie_path': employee_data.get('selfie_path', ''),
                'session_video_path': employee_data.get('session_video_path', ''),
                'emotion_label': '',
                'engagement_level': '',
                'posture_label': '',
                'performance_label': ''
            }
            
            # Add to dataframe
            self.df = pd.concat([self.df, pd.DataFrame([new_record])], ignore_index=True)
            
            # Update employee data cache
            self.employee_data[employee_data.get('name', '')] = {
                'sign_in': employee_data.get('sign_in', ''),
                'sign_out': employee_data.get('sign_out', ''),
                'task': employee_data.get('task', ''),
                'date': employee_data.get('date', ''),
                'selfie_path': employee_data.get('selfie_path', ''),
                'video_path': employee_data.get('session_video_path', '')
            }
            
            # Save to CSV
            self.save_to_csv()
            
            return f"✅ Record saved for {employee_data.get('name', '')}"
        except Exception as e:
            return f"❌ Error saving record: {str(e)}"
    
    def save_to_csv(self):
        """Save dataframe to CSV file"""
        if self.df is not None and not self.df.empty:
            self.df.to_csv('employee_multimodal_dataset.csv', index=False)
            return True
        return False

# ============================================
# 3. PERFORMANCE ANALYZER
# ============================================

class PerformanceAnalyzer:
    def __init__(self):
        self.agent = RuleBasedAgent()
    
    def calculate_work_hours(self, sign_in, sign_out):
        """Calculate work hours from sign in/out times"""
        try:
            if not sign_in or not sign_out:
                return 8.0
            
            # Parse times - handle both 24-hour and 12-hour formats
            try:
                in_time = datetime.strptime(sign_in, "%H:%M")
            except:
                try:
                    in_time = datetime.strptime(sign_in, "%I:%M %p")
                except:
                    return 8.0
            
            try:
                out_time = datetime.strptime(sign_out, "%H:%M")
            except:
                try:
                    out_time = datetime.strptime(sign_out, "%I:%M %p")
                except:
                    return 8.0
            
            duration = (out_time - in_time).seconds / 3600
            
            # Handle overnight shifts
            if duration < 0:
                duration += 24
            
            return round(duration, 2)
        except:
            return 8.0
    
    def assess_task_complexity(self, task_description):
        """Assess task complexity based on keywords"""
        if not task_description:
            return 0.5
        
        task_lower = task_description.lower()
        complexity = 0.5  # Default medium complexity
        
        # Keyword-based complexity scoring
        complexity_keywords = {
            'design': 0.8, 'develop': 0.9, 'implement': 0.85, 'create': 0.7,
            'build': 0.75, 'analyze': 0.6, 'review': 0.5, 'update': 0.4,
            'fix': 0.3, 'test': 0.4, 'manage': 0.7, 'lead': 0.8, 'plan': 0.6
        }
        
        for keyword, score in complexity_keywords.items():
            if keyword in task_lower:
                complexity = max(complexity, score)
        
        return complexity
    
    def analyze_video_features(self, video_file):
        """Analyze video file (simplified)"""
        if video_file is None:
            return {'duration': 0, 'fps': 0, 'motion': 0.5, 'engagement_score': 50}
        
        try:
            # For demo purposes, return simulated values
            return {
                'duration': 120,  # 2 minutes
                'fps': 30,
                'motion': 0.7,
                'engagement_score': 75.5
            }
        except:
            return {'duration': 0, 'fps': 0, 'motion': 0.5, 'engagement_score': 50}
    
    def analyze_image_features(self, image_file):
        """Analyze image file (simplified)"""
        if image_file is None:
            return {'brightness': 0.5, 'contrast': 0.5, 'face_detected': False, 'quality_score': 50}
        
        try:
            img = Image.open(image_file)
            img_array = np.array(img)
            
            # Simple analysis
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            brightness = np.mean(gray) / 255
            contrast = np.std(gray) / 100
            
            # Simple face detection (placeholder)
            face_detected = img_array.shape[0] > 100 and img_array.shape[1] > 100
            
            return {
                'brightness': round(brightness, 2),
                'contrast': round(contrast, 2),
                'face_detected': face_detected,
                'quality_score': round((brightness + contrast) * 50, 1)
            }
        except:
            return {'brightness': 0.5, 'contrast': 0.5, 'face_detected': False, 'quality_score': 50}
    
    def calculate_performance_score(self, work_hours, task_complexity, engagement, image_quality):
        """Calculate overall performance score"""
        # Weighted scoring
        weights = {
            'work_hours': 0.30,
            'task_complexity': 0.25,
            'engagement': 0.30,
            'image_quality': 0.15
        }
        
        # Normalize work hours (optimal 8 hours)
        if work_hours <= 0:
            hours_score = 0
        elif work_hours <= 12:
            hours_score = min(100, (work_hours / 8) * 100)
        else:
            hours_score = 100  # Cap at 100
        
        # Calculate weighted score
        score = (
            hours_score * weights['work_hours'] +
            task_complexity * 100 * weights['task_complexity'] +
            engagement * weights['engagement'] +
            image_quality * weights['image_quality']
        )
        
        return round(min(100, score), 1)
    
    def generate_recommendations(self, performance_score, action_idx, work_hours, engagement, image_quality):
        """Generate performance recommendations"""
        recommendations = []
        
        # Add RL-based action
        recommendations.append(f"🤖 **Recommended Action:** {self.agent.get_action_description(action_idx)}")
        
        # Performance-based recommendations
        if performance_score >= 85:
            recommendations.append("🎯 **Performance:** Excellent! Consider leadership opportunities.")
        elif performance_score >= 70:
            recommendations.append("✅ **Performance:** Good. Focus on continuous improvement.")
        elif performance_score >= 50:
            recommendations.append("⚠️ **Performance:** Average. Identify areas for growth.")
        else:
            recommendations.append("❌ **Performance:** Needs improvement. Schedule coaching session.")
        
        # Work hours recommendations
        if work_hours > 10:
            recommendations.append("⏰ **Work Hours:** High hours detected. Monitor work-life balance.")
        elif work_hours < 6:
            recommendations.append("⏰ **Work Hours:** Low hours. Assess task allocation.")
        
        # Engagement recommendations
        if engagement < 40:
            recommendations.append("💡 **Engagement:** Low engagement detected. Review motivation factors.")
        
        # Image quality recommendations
        if image_quality < 40:
            recommendations.append("📸 **Workspace:** Poor image quality. Check workspace setup.")
        
        return recommendations
    
    def create_visualizations(self, performance_score, engagement, image_quality, work_hours, task_complexity):
        """Create performance visualizations"""
        charts = {}
        
        # Bar chart for metrics
        metrics = ['Performance', 'Engagement', 'Image Quality', 'Work Hours', 'Task Complexity']
        values = [
            performance_score,
            engagement,
            image_quality,
            min(100, (work_hours / 8) * 100),
            task_complexity * 100
        ]
        
        fig_bar = go.Figure(data=[go.Bar(
            x=metrics,
            y=values,
            marker_color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
        )])
        
        fig_bar.update_layout(
            title="Performance Metrics",
            xaxis_title="Metric",
            yaxis_title="Score (0-100)",
            yaxis_range=[0, 100],
            height=400
        )
        
        charts['bar'] = fig_bar
        
        return charts
    
    def analyze(self, employee_data, video_file, image_file):
        """Main analysis function"""
        # Calculate work hours
        work_hours = self.calculate_work_hours(
            employee_data.get('sign_in', ''),
            employee_data.get('sign_out', '')
        )
        
        # Assess task complexity
        task = employee_data.get('task', '')
        task_complexity = self.assess_task_complexity(task)
        
        # Analyze multimedia
        video_analysis = self.analyze_video_features(video_file)
        image_analysis = self.analyze_image_features(image_file)
        
        engagement_score = video_analysis.get('engagement_score', 50)
        image_quality_score = image_analysis.get('quality_score', 50)
        
        # Calculate performance score
        performance_score = self.calculate_performance_score(
            work_hours, task_complexity, engagement_score, image_quality_score
        )
        
        # Create state for RL agent
        state = [
            work_hours / 24.0,  # Normalized work hours
            task_complexity,
            video_analysis.get('motion', 0.5),
            image_analysis.get('brightness', 0.5)
        ]
        
        # Get RL action
        action_idx = self.agent.select_action(state)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            performance_score, action_idx, work_hours, engagement_score, image_quality_score
        )
        
        # Create visualizations
        charts = self.create_visualizations(
            performance_score, engagement_score, image_quality_score, 
            work_hours, task_complexity
        )
        
        # Generate report
        report = self.generate_report(
            employee_data, performance_score, action_idx,
            engagement_score, image_quality_score, work_hours, recommendations
        )
        
        return {
            'report': report,
            'performance_score': performance_score,
            'action_idx': action_idx,
            'action_description': self.agent.get_action_description(action_idx),
            'engagement_score': engagement_score,
            'image_quality': image_quality_score,
            'work_hours': work_hours,
            'task_complexity': task_complexity,
            'recommendations': recommendations,
            'charts': charts,
            'video_analysis': video_analysis,
            'image_analysis': image_analysis
        }
    
    def generate_report(self, employee_data, performance_score, action_idx,
                       engagement_score, image_quality, work_hours, recommendations):
        """Generate detailed report"""
        report = f"""
        {'=' * 60}
        EMPLOYEE PERFORMANCE ANALYSIS REPORT
        {'=' * 60}
        
        📋 EMPLOYEE INFORMATION:
        • Name: {employee_data.get('name', 'N/A')}
        • Date: {employee_data.get('date', 'N/A')}
        • Task: {employee_data.get('task', 'N/A')}
        • Work Hours: {work_hours:.1f} hours
        
        📊 PERFORMANCE ANALYSIS:
        • Overall Score: {performance_score}/100
        • Performance Level: {'Excellent' if performance_score >= 85 else 'Good' if performance_score >= 70 else 'Average' if performance_score >= 50 else 'Needs Improvement'}
        
        🎯 MULTIMODAL ANALYSIS:
        • Engagement Score: {engagement_score:.1f}/100
        • Image Quality Score: {image_quality:.1f}/100
        
        🤖 AI RECOMMENDATION:
        • Action: {self.agent.get_action_description(action_idx)}
        
        💡 RECOMMENDATIONS:
        """
        
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"\n{'=' * 60}"
        report += f"\n📅 Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        report += f"\n{'=' * 60}"
        
        return report

# ============================================
# 4. STREAMLIT APPLICATION
# ============================================

def main():
    st.set_page_config(
        page_title="Employee Performance Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'database' not in st.session_state:
        st.session_state.database = EmployeeDatabase()
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = PerformanceAnalyzer()
    
    if 'selected_employee' not in st.session_state:
        st.session_state.selected_employee = None
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #2ca02c;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #2ca02c;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="main-title">📊 Employee Performance Analyzer</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🔍 Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Dashboard", "📁 Dataset", "🔍 Analysis", "📊 Reports"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # System Status
        st.markdown("### ⚙️ System Status")
        
        # Dataset status
        if st.session_state.database.df is not None and not st.session_state.database.df.empty:
            st.success(f"✅ Dataset: {len(st.session_state.database.df)} records")
        else:
            st.warning("📊 No dataset loaded")
        
        # Analysis status
        if st.session_state.analysis_results:
            st.success("✅ Recent analysis available")
        else:
            st.info("🔍 No analysis yet")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Clear All", type="secondary"):
            st.session_state.selected_employee = None
            st.session_state.analysis_results = None
            st.rerun()
        
        if st.button("💾 Export Dataset"):
            if st.session_state.database.df is not None and not st.session_state.database.df.empty:
                csv = st.session_state.database.df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="employee_dataset.csv",
                    mime="text/csv"
                )
    
    # Main content based on selected page
    if "🏠 Dashboard" in page:
        display_dashboard()
    elif "📁 Dataset" in page:
        display_dataset()
    elif "🔍 Analysis" in page:
        display_analysis()
    elif "📊 Reports" in page:
        display_reports()

def display_dashboard():
    """Display dashboard page"""
    st.markdown('<div class="main-title">🏠 Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(st.session_state.database.df) if st.session_state.database.df is not None else 0
        st.metric("📊 Total Records", total_records)
    
    with col2:
        unique_employees = st.session_state.database.df['Team Members'].nunique() if st.session_state.database.df is not None else 0
        st.metric("👥 Unique Employees", unique_employees)
    
    with col3:
        if st.session_state.analysis_results:
            score = st.session_state.analysis_results['performance_score']
            st.metric("⭐ Last Score", f"{score}/100")
        else:
            st.metric("⭐ Last Score", "N/A")
    
    with col4:
        st.metric("🤖 AI Model", "Active")
    
    st.markdown("---")
    
    # Recent Analysis
    if st.session_state.analysis_results:
        st.markdown("### 📈 Recent Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance chart
            st.plotly_chart(st.session_state.analysis_results['charts']['bar'], use_container_width=True)
        
        with col2:
            # Key metrics
            st.markdown("#### 📋 Key Metrics")
            results = st.session_state.analysis_results
            
            st.markdown(f"""
            <div class="metric-card">
            <h4>🎯 Performance Score: {results['performance_score']}/100</h4>
            <p>Level: {'Excellent' if results['performance_score'] >= 85 else 'Good' if results['performance_score'] >= 70 else 'Average' if results['performance_score'] >= 50 else 'Needs Improvement'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
            <h4>⏰ Work Hours: {results['work_hours']:.1f} hours</h4>
            <p>{'Optimal' if 7 <= results['work_hours'] <= 9 else 'Too long' if results['work_hours'] > 9 else 'Too short'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
            <h4>🤖 AI Recommendation</h4>
            <p>{results['action_description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Top recommendations
        st.markdown("### 💡 Top Recommendations")
        recommendations = st.session_state.analysis_results.get('recommendations', [])
        for rec in recommendations[:3]:
            st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
    
    else:
        st.info("ℹ️ No recent analysis found. Go to the Analysis page to get started!")

def display_dataset():
    """Display dataset management page"""
    st.markdown('<div class="main-title">📁 Dataset Management</div>', unsafe_allow_html=True)
    
    # Upload section
    st.markdown("### 📤 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with employee data",
        type=['csv'],
        help="Upload your employee dataset in CSV format"
    )
    
    if uploaded_file is not None:
        if st.button("📥 Process Upload", type="primary"):
            with st.spinner("Processing dataset..."):
                message, _ = st.session_state.database.upload_dataset(uploaded_file)
                st.success(message)
                st.rerun()
    
    st.markdown("---")
    
    # Dataset preview
    if st.session_state.database.df is not None and not st.session_state.database.df.empty:
        st.markdown(f"### 👁️ Dataset Preview ({len(st.session_state.database.df)} records)")
        
        # Show data preview
        st.dataframe(
            st.session_state.database.df.head(10),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Employee search section
        st.markdown("### 🔍 Search Employee")
        
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            search_term = st.text_input(
                "Enter employee name",
                placeholder="Type name to search...",
                key="search_input"
            )
        
        with search_col2:
            st.markdown("")
            st.markdown("")
            search_clicked = st.button("🔍 Search", use_container_width=True)
        
        # Handle search
        if search_term or search_clicked:
            matches = st.session_state.database.search_employees(search_term)
            
            if matches:
                st.success(f"Found {len(matches)} employee(s)")
                
                # Employee selection
                selected = st.selectbox(
                    "Select employee",
                    matches,
                    key="employee_select"
                )
                
                if selected:
                    # Get employee details
                    sign_in, sign_out, task, date = st.session_state.database.get_employee_details(selected)
                    
                    # Display details in a nice format
                    st.markdown("### 👤 Employee Details")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.text_input("👤 Employee Name", value=selected, disabled=True)
                        st.text_input("⏰ Sign In Time", value=sign_in, disabled=True)
                    
                    with col2:
                        st.text_input("📅 Date", value=date, disabled=True)
                        st.text_input("⏰ Sign Out Time", value=sign_out, disabled=True)
                    
                    st.text_area("📝 Completed Task", value=task, disabled=True, height=100)
                    
                    # "Use for Analysis" button
                    if st.button("🚀 Use for Analysis", type="primary", use_container_width=True):
                        st.session_state.selected_employee = {
                            'name': selected,
                            'sign_in': sign_in,
                            'sign_out': sign_out,
                            'task': task,
                            'date': date
                        }
                        st.success(f"✅ Employee '{selected}' loaded for analysis!")
                        st.rerun()
            
            else:
                st.warning("No employees found matching your search.")
    
    else:
        st.warning("📭 No dataset loaded. Please upload a CSV file to get started.")

def display_analysis():
    """Display performance analysis page"""
    st.markdown('<div class="main-title">🔍 Performance Analysis</div>', unsafe_allow_html=True)
    
    # Check if employee is pre-loaded from dataset
    preloaded_employee = None
    if st.session_state.selected_employee:
        preloaded_employee = st.session_state.selected_employee
        st.success(f"👤 Using employee: {preloaded_employee['name']}")
    
    # Analysis form
    with st.form("analysis_form"):
        st.markdown("### 📋 Employee Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            employee_name = st.text_input(
                "👤 Employee Name *",
                value=preloaded_employee['name'] if preloaded_employee else "",
                placeholder="Enter employee name"
            )
            
            analysis_date = st.text_input(
                "📅 Analysis Date",
                value=preloaded_employee['date'] if preloaded_employee else datetime.now().strftime("%d.%m.%Y"),
                placeholder="DD.MM.YYYY"
            )
        
        with col2:
            # 24-hour time selection
            sign_in_time = st.selectbox(
                "⏰ Sign In Time *",
                [f"{h:02d}:{m:02d}" for h in range(0, 24) for m in [0, 30]],
                index=18,  # Default to 09:00
                help="Select sign in time (24-hour format)"
            )
            
            sign_out_time = st.selectbox(
                "⏰ Sign Out Time *",
                [f"{h:02d}:{m:02d}" for h in range(0, 24) for m in [0, 30]],
                index=34,  # Default to 17:00
                help="Select sign out time (24-hour format)"
            )
        
        # Task description
        completed_task = st.text_area(
            "📝 Completed Task *",
            value=preloaded_employee['task'] if preloaded_employee else "",
            placeholder="Describe the completed task in detail...",
            height=100,
            help="Detailed description helps in better analysis"
        )
        
        st.markdown("---")
        st.markdown("### 🎬 Multimedia Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            session_video = st.file_uploader(
                "📹 Upload Session Video",
                type=['mp4', 'avi', 'mov'],
                help="Optional: Upload work session video"
            )
            if session_video:
                st.info(f"Video: {session_video.name}")
        
        with col2:
            selfie_image = st.file_uploader(
                "📸 Upload Selfie/Workspace Image",
                type=['jpg', 'jpeg', 'png'],
                help="Optional: Upload selfie or workspace photo"
            )
            if selfie_image:
                try:
                    img = Image.open(selfie_image)
                    st.image(img, caption="Uploaded Image", width=200)
                except:
                    st.warning("Could not display image")
        
        st.markdown("---")
        
        # Analyze button
        analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
        with analyze_col2:
            analyze_submitted = st.form_submit_button(
                "🚀 ANALYZE PERFORMANCE",
                type="primary",
                use_container_width=True
            )
        
        if analyze_submitted:
            if not employee_name:
                st.error("❌ Please enter employee name!")
            elif not completed_task:
                st.error("❌ Please describe the completed task!")
            else:
                with st.spinner("🔬 Analyzing performance..."):
                    # Prepare employee data
                    employee_data = {
                        'name': employee_name,
                        'date': analysis_date,
                        'sign_in': sign_in_time,
                        'sign_out': sign_out_time,
                        'task': completed_task
                    }
                    
                    # Perform analysis
                    st.session_state.analysis_results = st.session_state.analyzer.analyze(
                        employee_data, session_video, selfie_image
                    )
                    
                    # Save to database
                    try:
                        save_data = employee_data.copy()
                        save_data['selfie_path'] = selfie_image.name if selfie_image else ''
                        save_data['session_video_path'] = session_video.name if session_video else ''
                        
                        save_result = st.session_state.database.save_record(save_data)
                        st.success(save_result)
                    except Exception as e:
                        st.error(f"Could not save record: {e}")
                    
                    st.rerun()
    
    # Display results if available
    if st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        # Key metrics
        results = st.session_state.analysis_results
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Performance",
                f"{results['performance_score']}/100",
                delta="Excellent" if results['performance_score'] >= 85 else "Good" if results['performance_score'] >= 70 else "Average"
            )
        
        with col2:
            st.metric(
                "💡 Engagement",
                f"{results['engagement_score']:.1f}/100"
            )
        
        with col3:
            st.metric(
                "⏰ Work Hours",
                f"{results['work_hours']:.1f}"
            )
        
        with col4:
            st.metric(
                "🤖 AI Action",
                results['action_description'].split()[0]
            )
        
        # Visualizations
        st.plotly_chart(results['charts']['bar'], use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        recommendations = results.get('recommendations', [])
        for rec in recommendations:
            st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
        
        # Detailed report
        with st.expander("📄 View Detailed Report", expanded=False):
            st.text(results['report'])
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                # Download report as text
                st.download_button(
                    label="📥 Download Report",
                    data=results['report'],
                    file_name=f"performance_report_{employee_name}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Download as JSON
                json_data = json.dumps({
                    'employee': employee_data if 'employee_data' in locals() else {},
                    'analysis': {
                        'performance_score': results['performance_score'],
                        'engagement_score': results['engagement_score'],
                        'work_hours': results['work_hours'],
                        'recommendations': recommendations
                    }
                }, indent=2)
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"analysis_{employee_name}_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

def display_reports():
    """Display reports page"""
    st.markdown('<div class="main-title">📊 Reports & Analytics</div>', unsafe_allow_html=True)
    
    if st.session_state.database.df is None or st.session_state.database.df.empty:
        st.warning("📭 No data available for reports. Please upload a dataset first.")
        return
    
    # Summary statistics
    st.markdown("### 📈 Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    df = st.session_state.database.df
    
    with col1:
        st.metric("📊 Total Records", len(df))
    
    with col2:
        unique_count = df['Team Members'].nunique() if 'Team Members' in df.columns else 0
        st.metric("👥 Unique Employees", unique_count)
    
    with col3:
        date_range = "N/A"
        if 'Date' in df.columns and not df['Date'].empty:
            dates = pd.to_datetime(df['Date'], errors='coerce')
            if not dates.isna().all():
                date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
        st.metric("📅 Date Range", date_range)
    
    st.markdown("---")
    
    # Data visualizations
    st.markdown("### 📊 Data Visualizations")
    
    if 'Team Members' in df.columns and not df['Team Members'].empty:
        # Employee activity chart
        employee_counts = df['Team Members'].value_counts().head(10)
        
        fig1 = px.bar(
            x=employee_counts.index,
            y=employee_counts.values,
            title="Top 10 Employees by Activity",
            labels={'x': 'Employee', 'y': 'Number of Records'},
            color=employee_counts.values,
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    if 'Completed Task' in df.columns and not df['Completed Task'].empty:
        # Task word cloud simulation
        tasks = df['Completed Task'].astype(str)
        task_words = ' '.join(tasks).lower().split()
        
        from collections import Counter
        word_counts = Counter(task_words)
        top_words = dict(word_counts.most_common(20))
        
        fig2 = go.Figure(data=[go.Bar(
            x=list(top_words.keys()),
            y=list(top_words.values()),
            marker_color='lightblue'
        )])
        
        fig2.update_layout(
            title="Most Common Task Words",
            xaxis_title="Word",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Export section
    st.markdown("### 📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Export Full Dataset", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="employee_full_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if st.button("📄 Generate Summary", use_container_width=True):
            summary = f"""
            EMPLOYEE DATASET SUMMARY
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Basic Statistics:
            - Total Records: {len(df)}
            - Unique Employees: {df['Team Members'].nunique() if 'Team Members' in df.columns else 0}
            - Date Range: {df['Date'].min() if 'Date' in df.columns else 'N/A'} to {df['Date'].max() if 'Date' in df.columns else 'N/A'}
            
            Columns: {', '.join(df.columns.tolist())}
            """
            
            st.download_button(
                label="📥 Download Summary",
                data=summary,
                file_name="dataset_summary.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    with col3:
        if st.button("📊 Export Charts", use_container_width=True):
            st.info("Chart export feature coming soon!")

# Run the app
if __name__ == "__main__":
    main()
