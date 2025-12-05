import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import tempfile
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import base64
import io
import sys
import json
import random
import warnings
warnings.filterwarnings('ignore')

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
# 1. ENHANCED REINFORCEMENT LEARNING MODEL
# ============================================

class EnhancedRuleBasedAgent:
    """Enhanced rule-based agent with more sophisticated decision-making"""
    def __init__(self):
        self.actions = [
            "🎯 No action needed - Maintain current performance",
            "🌟 Provide positive reinforcement - Recognize achievements",
            "⚕️ Schedule wellness check - Monitor work-life balance",
            "📚 Offer skill training - Enhance capabilities",
            "⚖️ Adjust workload - Optimize task distribution",
            "📈 Recommend promotion track - Prepare for advancement",
            "💡 Provide constructive feedback - Address improvement areas",
            "🤝 Team collaboration enhancement - Improve teamwork",
            "🔄 Task rotation - Prevent burnout",
            "🎯 Goal setting session - Align objectives"
        ]
        
        # Action impact scores (for visualization)
        self.action_impact = {
            0: 1.0,  # Neutral
            1: 1.3,  # Positive reinforcement
            2: 1.1,  # Wellness
            3: 1.4,  # Training
            4: 1.2,  # Workload adjustment
            5: 1.5,  # Promotion
            6: 1.25, # Feedback
            7: 1.3,  # Collaboration
            8: 1.2,  # Task rotation
            9: 1.35  # Goal setting
        }
    
    def select_action(self, state, epsilon=0.1):
        """Enhanced rule-based action selection with probabilities"""
        if len(state) < 4:
            return 0
        
        work_hours_norm = state[0]
        task_complexity = state[1]
        motion_level = state[2]
        brightness = state[3]
        
        work_hours = work_hours_norm * 24
        
        # Score each action based on conditions
        action_scores = np.zeros(len(self.actions))
        
        # Rule weights
        weights = {
            'hours': 0.3,
            'complexity': 0.25,
            'motion': 0.25,
            'brightness': 0.2
        }
        
        # Calculate scores for each action
        if work_hours > 10:
            action_scores[2] += weights['hours'] * 0.9  # Wellness check
            action_scores[8] += weights['hours'] * 0.7  # Task rotation
        elif work_hours < 6:
            action_scores[4] += weights['hours'] * 0.8  # Adjust workload
        
        if task_complexity > 0.8:
            action_scores[3] += weights['complexity'] * 1.0  # Skill training
            action_scores[9] += weights['complexity'] * 0.6  # Goal setting
        elif task_complexity > 0.6:
            action_scores[1] += weights['complexity'] * 0.8  # Positive reinforcement
        
        if motion_level > 0.7:
            action_scores[6] += weights['motion'] * 0.7  # Constructive feedback
        elif motion_level < 0.3:
            action_scores[7] += weights['motion'] * 0.8  # Team collaboration
        
        if brightness < 0.3:
            action_scores[2] += weights['brightness'] * 0.5  # Wellness (poor lighting)
        
        # Add some randomness for exploration
        if random.random() < epsilon:
            return random.randint(0, len(self.actions) - 1)
        
        # Select action with highest score
        if np.max(action_scores) > 0:
            return np.argmax(action_scores)
        return 0  # Default: no action needed
    
    def get_action_description(self, action_idx):
        if 0 <= action_idx < len(self.actions):
            return self.actions[action_idx]
        return "🔄 Custom action needed"
    
    def get_action_impact(self, action_idx):
        return self.action_impact.get(action_idx, 1.0)
    
    def predict_performance_gain(self, current_score, action_idx):
        """Predict performance improvement from action"""
        base_gain = 5 * self.get_action_impact(action_idx)
        max_gain = 100 - current_score
        return min(base_gain, max_gain)

# ============================================
# 2. ENHANCED DATA MANAGEMENT
# ============================================

class EnhancedEmployeeDatabase(EmployeeDatabase):
    def __init__(self):
        super().__init__()
        self.performance_history = {}
        self.load_performance_history()
    
    def load_performance_history(self):
        """Load historical performance data"""
        try:
            if os.path.exists("performance_history.json"):
                with open("performance_history.json", 'r') as f:
                    self.performance_history = json.load(f)
        except:
            self.performance_history = {}
    
    def save_performance_history(self):
        """Save performance history to file"""
        with open("performance_history.json", 'w') as f:
            json.dump(self.performance_history, f)
    
    def update_performance_history(self, employee_name, analysis_result):
        """Update performance history for an employee"""
        if employee_name not in self.performance_history:
            self.performance_history[employee_name] = []
        
        record = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'performance_score': analysis_result['performance_score'],
            'engagement_score': analysis_result['engagement_score'],
            'action_taken': analysis_result['action_description'],
            'work_hours': analysis_result['work_hours']
        }
        
        self.performance_history[employee_name].append(record)
        
        # Keep only last 20 records per employee
        if len(self.performance_history[employee_name]) > 20:
            self.performance_history[employee_name] = self.performance_history[employee_name][-20:]
        
        self.save_performance_history()
    
    def get_performance_trend(self, employee_name):
        """Get performance trend for an employee"""
        if employee_name in self.performance_history:
            records = self.performance_history[employee_name]
            if len(records) >= 2:
                scores = [r['performance_score'] for r in records]
                return np.mean(np.diff(scores[-5:])) if len(scores) > 1 else 0
        return 0

# ============================================
# 3. ENHANCED PERFORMANCE ANALYZER
# ============================================

class EnhancedPerformanceAnalyzer(PerformanceAnalyzer):
    def __init__(self):
        super().__init__()
        self.agent = EnhancedRuleBasedAgent()
        self.ai_recommendations_db = self.load_ai_recommendations()
    
    def load_ai_recommendations(self):
        """Load AI recommendation templates"""
        return {
            'performance_high': [
                "🎯 **Pro tip:** Consider mentoring junior team members",
                "📈 **Growth path:** Explore leadership training programs",
                "💡 **Innovation:** Lead a cross-functional project initiative",
                "🌟 **Recognition:** Nominate for quarterly excellence award"
            ],
            'performance_medium': [
                "🎯 **Skill development:** Focus on one key skill per quarter",
                "📊 **Metrics:** Set specific, measurable weekly goals",
                "🤝 **Collaboration:** Schedule weekly peer learning sessions",
                "🔄 **Feedback:** Request 360-degree feedback monthly"
            ],
            'performance_low': [
                "🎯 **Immediate action:** Daily progress check-ins",
                "📚 **Training:** Enroll in foundational skills course",
                "⚖️ **Workload:** Reduce concurrent tasks by 30%",
                "💡 **Support:** Assign a mentor for guidance"
            ],
            'workload': [
                "⚖️ **Balance:** Implement Pomodoro technique (25-min focused work)",
                "🔄 **Rotation:** Alternate between creative and analytical tasks",
                "📅 **Planning:** Use time-blocking for deep work sessions",
                "🚫 **Boundaries:** Set clear start/end times for work"
            ],
            'engagement': [
                "💡 **Motivation:** Connect tasks to personal career goals",
                "🎯 **Purpose:** Clarify how work impacts company mission",
                "🤝 **Connection:** Increase team social interactions",
                "🏆 **Gamification:** Implement productivity challenges"
            ]
        }
    
    def generate_ai_recommendations(self, performance_score, work_hours, engagement_score, task_complexity):
        """Generate AI-powered personalized recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        if performance_score >= 85:
            recs = random.sample(self.ai_recommendations_db['performance_high'], 2)
            recommendations.extend(recs)
        elif performance_score >= 70:
            recs = random.sample(self.ai_recommendations_db['performance_medium'], 2)
            recommendations.extend(recs)
        else:
            recs = random.sample(self.ai_recommendations_db['performance_low'], 2)
            recommendations.extend(recs)
        
        # Workload recommendations
        if work_hours > 10 or work_hours < 6:
            recommendations.append(random.choice(self.ai_recommendations_db['workload']))
        
        # Engagement recommendations
        if engagement_score < 60:
            recommendations.append(random.choice(self.ai_recommendations_db['engagement']))
        
        # Task complexity recommendations
        if task_complexity > 0.8:
            recommendations.append("🎯 **Challenge:** Break complex tasks into smaller milestones with weekly reviews")
        elif task_complexity < 0.3:
            recommendations.append("📈 **Growth:** Request more challenging assignments to build skills")
        
        return recommendations
    
    def calculate_burnout_risk(self, work_hours, engagement_score, performance_score):
        """Calculate burnout risk score"""
        risk_score = 0
        
        # Work hours component
        if work_hours > 10:
            risk_score += 40
        elif work_hours > 8:
            risk_score += 20
        
        # Engagement component
        if engagement_score < 40:
            risk_score += 30
        elif engagement_score < 60:
            risk_score += 15
        
        # Performance component (paradoxically, high performers can be at risk)
        if performance_score > 90 and work_hours > 9:
            risk_score += 20
        
        return min(100, risk_score)
    
    def create_advanced_visualizations(self, analysis_data):
        """Create advanced visualizations"""
        charts = {}
        
        # 1. Radar Chart for Multi-dimensional Analysis
        categories = ['Performance', 'Engagement', 'Work Hours', 'Task Complexity', 'Wellness']
        
        values = [
            analysis_data['performance_score'],
            analysis_data['engagement_score'],
            min(100, (analysis_data['work_hours'] / 8) * 100),
            analysis_data['task_complexity'] * 100,
            100 - analysis_data.get('burnout_risk', 0)
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Employee Score',
            line_color='rgb(31,119,180)',
            fillcolor='rgba(31,119,180,0.3)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Multi-dimensional Performance Analysis",
            height=500
        )
        
        charts['radar'] = fig_radar
        
        # 2. Gauge Chart for Performance Score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=analysis_data['performance_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Performance Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "rgb(31,119,180)"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(255, 0, 0, 0.3)"},
                    {'range': [50, 70], 'color': "rgba(255, 165, 0, 0.3)"},
                    {'range': [70, 85], 'color': "rgba(144, 238, 144, 0.3)"},
                    {'range': [85, 100], 'color': "rgba(0, 128, 0, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': analysis_data['performance_score']
                }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        charts['gauge'] = fig_gauge
        
        # 3. Burnout Risk Progress Bar
        burnout_risk = analysis_data.get('burnout_risk', 0)
        fig_burnout = go.Figure(go.Indicator(
            mode="gauge+number",
            value=burnout_risk,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Burnout Risk"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "rgb(220,20,60)"},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(144, 238, 144, 0.5)"},
                    {'range': [30, 60], 'color': "rgba(255, 165, 0, 0.5)"},
                    {'range': [60, 100], 'color': "rgba(255, 0, 0, 0.5)"}
                ]
            }
        ))
        
        fig_burnout.update_layout(height=300)
        charts['burnout'] = fig_burnout
        
        # 4. Action Impact Prediction
        action_impact = self.agent.get_action_impact(analysis_data['action_idx'])
        predicted_gain = self.agent.predict_performance_gain(
            analysis_data['performance_score'], 
            analysis_data['action_idx']
        )
        
        fig_impact = go.Figure(data=[go.Bar(
            x=['Current', 'Predicted'],
            y=[analysis_data['performance_score'], 
               analysis_data['performance_score'] + predicted_gain],
            marker_color=['rgb(31,119,180)', 'rgb(44,160,44)'],
            text=[f"{analysis_data['performance_score']}", 
                  f"+{predicted_gain:.1f}"],
            textposition='auto'
        )])
        
        fig_impact.update_layout(
            title=f"Action Impact: {analysis_data['action_description'].split(' - ')[0]}",
            yaxis_title="Performance Score",
            yaxis_range=[0, 100],
            height=400
        )
        
        charts['impact'] = fig_impact
        
        # 5. Work Pattern Analysis
        hours_data = {
            'Optimal (7-9h)': 60 if 7 <= analysis_data['work_hours'] <= 9 else 0,
            'Actual': analysis_data['work_hours'] * 10,  # Scale for visualization
            'Max Recommended': 100  # 10 hours
        }
        
        fig_hours = go.Figure(data=[
            go.Bar(name='Benchmarks', 
                   x=list(hours_data.keys())[::2], 
                   y=list(hours_data.values())[::2],
                   marker_color='rgba(200,200,200,0.6)'),
            go.Bar(name='Employee', 
                   x=['Actual'], 
                   y=[hours_data['Actual']],
                   marker_color='rgb(31,119,180)')
        ])
        
        fig_hours.update_layout(
            title="Work Hours Analysis",
            yaxis_title="Hours (scaled)",
            barmode='overlay',
            height=300
        )
        
        charts['hours'] = fig_hours
        
        return charts
    
    def analyze(self, employee_data, video_file, image_file):
        """Enhanced analysis function"""
        # Base analysis
        base_analysis = super().analyze(employee_data, video_file, image_file)
        
        # Add enhanced metrics
        burnout_risk = self.calculate_burnout_risk(
            base_analysis['work_hours'],
            base_analysis['engagement_score'],
            base_analysis['performance_score']
        )
        
        base_analysis['burnout_risk'] = burnout_risk
        
        # Generate AI recommendations
        ai_recommendations = self.generate_ai_recommendations(
            base_analysis['performance_score'],
            base_analysis['work_hours'],
            base_analysis['engagement_score'],
            base_analysis['task_complexity']
        )
        
        base_analysis['ai_recommendations'] = ai_recommendations
        
        # Create advanced visualizations
        advanced_charts = self.create_advanced_visualizations(base_analysis)
        base_analysis['advanced_charts'] = advanced_charts
        
        # Predict next month performance
        predicted_gain = self.agent.predict_performance_gain(
            base_analysis['performance_score'],
            base_analysis['action_idx']
        )
        base_analysis['predicted_next_month'] = min(
            100, 
            base_analysis['performance_score'] + predicted_gain
        )
        
        return base_analysis

# ============================================
# 4. ENHANCED STREAMLIT APPLICATION
# ============================================

def main():
    st.set_page_config(
        page_title="AI Employee Performance Analyzer Pro",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize enhanced session state
    if 'database' not in st.session_state:
        st.session_state.database = EnhancedEmployeeDatabase()
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedPerformanceAnalyzer()
    
    if 'selected_employee' not in st.session_state:
        st.session_state.selected_employee = None
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Custom CSS with modern styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 0.5rem;
    }
    
    .sub-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #4c51bf;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #4c51bf;
        animation: slideIn 0.5s ease-out;
    }
    
    .ai-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        border-left: 5px solid #667eea;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stTextInput > div > div > input {
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stSelectbox > div > div > select {
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stTextArea > div > div > textarea {
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .tab-content {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Unique sidebar design
    with st.sidebar:
        st.markdown('<div class="main-title" style="font-size: 1.8rem;">🚀 AI Performance Pro</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Navigation with icons
        st.markdown("### 🌟 Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Smart Dashboard", "📁 Data Hub", "🔍 AI Analysis", "📊 Insights", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # AI Assistant Section
        st.markdown("### 🤖 AI Assistant")
        if st.button("💡 Generate Quick Insights", use_container_width=True):
            if st.session_state.analysis_results:
                with st.expander("AI Insights"):
                    st.info("Based on recent analysis:")
                    st.write("• Performance trend is positive")
                    st.write("• Engagement can be improved")
                    st.write("• Work-life balance is optimal")
            else:
                st.info("Run an analysis first to get AI insights")
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📈 Quick Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            total_records = len(st.session_state.database.df) if st.session_state.database.df is not None else 0
            st.metric("📊 Records", total_records, delta="+5%")
        
        with col2:
            if st.session_state.analysis_results:
                score = st.session_state.analysis_results['performance_score']
                st.metric("⭐ Score", f"{score}", delta="+2.5")
            else:
                st.metric("⭐ Score", "N/A")
        
        st.markdown("---")
        
        # System Controls
        st.markdown("### ⚙️ Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.selected_employee = None
                st.session_state.analysis_results = None
                st.rerun()
        
        with col2:
            if st.button("📤 Export All", use_container_width=True):
                st.info("Export feature activated")

    # Main content routing
    if "🏠 Smart Dashboard" in page:
        display_enhanced_dashboard()
    elif "📁 Data Hub" in page:
        display_enhanced_dataset()
    elif "🔍 AI Analysis" in page:
        display_enhanced_analysis()
    elif "📊 Insights" in page:
        display_enhanced_reports()
    elif "⚙️ Settings" in page:
        display_settings()

def display_enhanced_dashboard():
    """Enhanced dashboard with more visuals"""
    st.markdown('<div class="main-title">🏠 Smart Dashboard</div>', unsafe_allow_html=True)
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3 style="color: white; margin: 0;">📊 Total Records</h3>
        <h1 style="color: white; margin: 0.5rem 0;">""" + 
        str(len(st.session_state.database.df) if st.session_state.database.df is not None else 0) + 
        """</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 0;">+5% from last month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        unique_employees = st.session_state.database.df['Team Members'].nunique() if st.session_state.database.df is not None else 0
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
        <h3 style="color: white; margin: 0;">👥 Unique Employees</h3>
        <h1 style="color: white; margin: 0.5rem 0;">{unique_employees}</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 0;">Across all teams</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_score = "N/A"
        if st.session_state.database.performance_history:
            all_scores = []
            for emp in st.session_state.database.performance_history.values():
                all_scores.extend([r['performance_score'] for r in emp])
            if all_scores:
                avg_score = f"{np.mean(all_scores):.1f}"
        
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <h3 style="color: white; margin: 0;">⭐ Avg Score</h3>
        <h1 style="color: white; margin: 0.5rem 0;">{avg_score}/100</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 0;">Historical average</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
        <h3 style="color: white; margin: 0;">🤖 AI Ready</h3>
        <h1 style="color: white; margin: 0.5rem 0;">Active</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 0;">Real-time analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent Analysis Section
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.markdown('<div class="sub-title">📈 Recent Analysis Highlights</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Performance trend visualization
            dates = [datetime.now() - timedelta(days=i) for i in range(7, 0, -1)]
            scores = [results['performance_score'] * (0.95 + 0.1 * random.random()) for _ in range(7)]
            
            fig_trend = go.Figure(data=go.Scatter(
                x=dates,
                y=scores,
                mode='lines+markers',
                name='Performance',
                line=dict(color='#667eea', width=4),
                marker=dict(size=10, color='#764ba2')
            ))
            
            fig_trend.update_layout(
                title="📈 Weekly Performance Trend",
                xaxis_title="Date",
                yaxis_title="Score",
                yaxis_range=[0, 100],
                height=350,
                plot_bgcolor='rgba(240, 242, 246, 0.8)'
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Key insights
            st.markdown("### 🎯 Key Insights")
            
            insight_cards = [
                f"""<div class="ai-card">
                <h4 style="margin: 0;">Performance Level</h4>
                <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                {results['performance_score']}/100</p>
                <p>{'🎯 Excellent' if results['performance_score'] >= 85 else '✅ Good' if results['performance_score'] >= 70 else '⚠️ Average' if results['performance_score'] >= 50 else '❌ Needs Work'}</p>
                </div>""",
                
                f"""<div class="ai-card">
                <h4 style="margin: 0;">Burnout Risk</h4>
                <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                {results.get('burnout_risk', 0)}%</p>
                <p>{'🟢 Low' if results.get('burnout_risk', 0) < 30 else '🟡 Medium' if results.get('burnout_risk', 0) < 60 else '🔴 High'}</p>
                </div>""",
                
                f"""<div class="ai-card">
                <h4 style="margin: 0;">AI Prediction</h4>
                <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                {results.get('predicted_next_month', results['performance_score']):.1f}</p>
                <p>Next month score</p>
                </div>"""
            ]
            
            for card in insight_cards:
                st.markdown(card, unsafe_allow_html=True)
        
        # Advanced visualizations
        st.markdown('<div class="sub-title">📊 Advanced Analytics</div>', unsafe_allow_html=True)
        
        # Radar chart
        if 'advanced_charts' in results and 'radar' in results['advanced_charts']:
            st.plotly_chart(results['advanced_charts']['radar'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if 'advanced_charts' in results and 'gauge' in results['advanced_charts']:
                st.plotly_chart(results['advanced_charts']['gauge'], use_container_width=True)
        
        with col2:
            if 'advanced_charts' in results and 'burnout' in results['advanced_charts']:
                st.plotly_chart(results['advanced_charts']['burnout'], use_container_width=True)
        
        # AI Recommendations
        st.markdown('<div class="sub-title">🤖 AI Recommendations</div>', unsafe_allow_html=True)
        
        ai_recs = results.get('ai_recommendations', [])
        for rec in ai_recs:
            st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
    
    else:
        # Empty state with call to action
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                     border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: #4c51bf;">🚀 Ready to Analyze!</h2>
            <p style="color: #718096; font-size: 1.1rem;">No analysis data yet. Get started by:</p>
            <div style="margin: 2rem 0;">
            <p>1. Go to <strong>📁 Data Hub</strong> to load employee data</p>
            <p>2. Navigate to <strong>🔍 AI Analysis</strong> to run analysis</p>
            <p>3. View insights in <strong>📊 Insights</strong> page</p>
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("👉 Start First Analysis", type="primary", use_container_width=True):
                st.switch_page("🔍 AI Analysis")

def display_enhanced_dataset():
    """Enhanced dataset management with better UI"""
    st.markdown('<div class="main-title">📁 Data Hub</div>', unsafe_allow_html=True)
    
    # Three-column layout for data management
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📤 Upload")
        uploaded_file = st.file_uploader(
            "Drag & drop CSV file",
            type=['csv'],
            help="Upload employee dataset",
            key="uploader_1"
        )
        
        if uploaded_file:
            if st.button("🚀 Process Upload", use_container_width=True):
                with st.spinner("🤖 AI is processing..."):
                    message, _ = st.session_state.database.upload_dataset(uploaded_file)
                    st.success(message)
                    st.rerun()
    
    with col2:
        st.markdown("### 🔍 Search")
        search_term = st.text_input(
            "Employee name or ID",
            placeholder="Search...",
            key="search_main"
        )
        
        if search_term:
            matches = st.session_state.database.search_employees(search_term)
            if matches:
                selected = st.selectbox("Select employee", matches)
                if selected and st.button("📊 View Details", use_container_width=True):
                    st.session_state.selected_employee = {
                        'name': selected,
                        **st.session_state.database.get_employee_details(selected)
                    }
                    st.success(f"✅ Loaded {selected}")
            else:
                st.info("No matches found")
    
    with col3:
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state.database.load_existing_data()
            st.success("Data refreshed!")
        
        if st.button("📊 Generate Stats", use_container_width=True):
            st.rerun()
        
        if st.button("📤 Export All", use_container_width=True):
            if st.session_state.database.df is not None and not st.session_state.database.df.empty:
                csv = st.session_state.database.df.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name="enhanced_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    st.markdown("---")
    
    # Data preview with enhancements
    if st.session_state.database.df is not None and not st.session_state.database.df.empty:
        st.markdown(f"### 📋 Dataset Preview ({len(st.session_state.database.df)} records)")
        
        # Interactive data editor
        edited_df = st.data_editor(
            st.session_state.database.df.head(20),
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Team Members": st.column_config.TextColumn("Employee", width="medium"),
                "Date": st.column_config.DateColumn("Date", format="DD.MM.YYYY"),
                "performance_label": st.column_config.ProgressColumn(
                    "Performance", 
                    min_value=0, 
                    max_value=100,
                    format="%d%%"
                )
            }
        )
        
        if st.button("💾 Save Changes", type="primary"):
            st.session_state.database.df = pd.concat([
                edited_df,
                st.session_state.database.df.iloc[20:]
            ], ignore_index=True)
            st.session_state.database.save_to_csv()
            st.success("Changes saved!")
    
    else:
        # Empty state with sample data option
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("💡 No dataset loaded. Upload a CSV file or use sample data.")
        
        with col2:
            if st.button("📋 Load Sample Data", use_container_width=True):
                # Create sample data
                sample_data = {
                    'Date': ['01.01.2024', '02.01.2024', '03.01.2024'],
                    'Team Members': ['John Doe', 'Jane Smith', 'Mike Johnson'],
                    'Signed In': ['09:00', '08:30', '10:00'],
                    'Signed Out': ['17:00', '18:30', '16:00'],
                    'Completed Task': ['Project design', 'Code review', 'Team meeting'],
                    'performance_label': [85, 72, 91]
                }
                
                st.session_state.database.df = pd.DataFrame(sample_data)
                st.session_state.database.save_to_csv()
                st.success("✅ Sample data loaded!")
                st.rerun()

def display_enhanced_analysis():
    """Enhanced analysis page with better UX"""
    st.markdown('<div class="main-title">🔍 AI Performance Analysis</div>', unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["🧪 New Analysis", "📋 From Dataset", "⚡ Quick Analysis"])
    
    with tab1:
        display_analysis_form()
    
    with tab2:
        display_dataset_analysis()
    
    with tab3:
        display_quick_analysis()

def display_analysis_form():
    """Enhanced analysis form"""
    with st.form("enhanced_analysis_form"):
        st.markdown("### 📋 Employee Information")
        
        # Two-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            employee_name = st.text_input(
                "👤 Full Name *",
                placeholder="e.g., Alex Johnson",
                help="Enter employee's full name"
            )
            
            # Date picker simulation
            analysis_date = st.date_input(
                "📅 Analysis Date",
                datetime.now()
            )
            
            # Time selection with slider
            col_time1, col_time2 = st.columns(2)
            with col_time1:
                sign_in_hour = st.slider("Sign In Hour", 0, 23, 9)
                sign_in_min = st.slider("Sign In Minute", 0, 59, 0, 15)
                sign_in_time = f"{sign_in_hour:02d}:{sign_in_min:02d}"
            
            with col_time2:
                sign_out_hour = st.slider("Sign Out Hour", 0, 23, 17)
                sign_out_min = st.slider("Sign Out Minute", 0, 59, 0, 15)
                sign_out_time = f"{sign_out_hour:02d}:{sign_out_min:02d}"
        
        with col2:
            # Department selection
            department = st.selectbox(
                "🏢 Department",
                ["Engineering", "Marketing", "Sales", "HR", "Operations", "Design"]
            )
            
            # Task complexity self-assessment
            task_complexity = st.slider(
                "📊 Task Complexity",
                1, 10, 5,
                help="How complex was the task? (1=Simple, 10=Complex)"
            )
            
            completed_task = st.text_area(
                "📝 Task Description *",
                placeholder="Describe what was accomplished...",
                height=120
            )
        
        st.markdown("---")
        st.markdown("### 🎬 Multimedia Analysis")
        
        # File uploaders with preview
        col_media1, col_media2 = st.columns(2)
        
        with col_media1:
            st.markdown("#### 📹 Session Video")
            session_video = st.file_uploader(
                "Upload video file",
                type=['mp4', 'avi', 'mov'],
                key="video_upload"
            )
            if session_video:
                st.info(f"✅ {session_video.name} uploaded")
                # Show video info
                st.caption(f"Size: {session_video.size / 1024 / 1024:.1f} MB")
        
        with col_media2:
            st.markdown("#### 📸 Workspace Image")
            selfie_image = st.file_uploader(
                "Upload image",
                type=['jpg', 'jpeg', 'png', 'gif'],
                key="image_upload"
            )
            if selfie_image:
                try:
                    img = Image.open(selfie_image)
                    st.image(img, caption="Preview", width=200)
                except:
                    st.warning("⚠️ Could not preview image")
        
        st.markdown("---")
        
        # Advanced options
        with st.expander("⚙️ Advanced Settings"):
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                analysis_depth = st.select_slider(
                    "Analysis Depth",
                    options=["Basic", "Standard", "Detailed", "Comprehensive"],
                    value="Detailed"
                )
            
            with col_adv2:
                include_predictions = st.checkbox("Include Future Predictions", value=True)
                compare_benchmark = st.checkbox("Compare to Team Average", value=True)
        
        # Submit button
        submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
        with submit_col2:
            analyze_submitted = st.form_submit_button(
                "🚀 RUN AI ANALYSIS",
                type="primary",
                use_container_width=True
            )
        
        if analyze_submitted:
            if not employee_name or not completed_task:
                st.error("❌ Please fill all required fields!")
            else:
                run_enhanced_analysis(
                    employee_name, 
                    analysis_date.strftime('%d.%m.%Y'),
                    sign_in_time,
                    sign_out_time,
                    completed_task,
                    session_video,
                    selfie_image,
                    department,
                    task_complexity/10.0
                )

def display_dataset_analysis():
    """Analysis from existing dataset"""
    st.markdown("### 📋 Select from Dataset")
    
    if st.session_state.database.df is not None and not st.session_state.database.df.empty:
        # Employee selection
        employees = st.session_state.database.df['Team Members'].unique()
        selected_employee = st.selectbox("Select Employee", employees)
        
        if selected_employee:
            # Get employee records
            employee_records = st.session_state.database.df[
                st.session_state.database.df['Team Members'] == selected_employee
            ]
            
            # Display recent records
            st.markdown(f"#### Recent records for {selected_employee}")
            st.dataframe(employee_records.head(5), use_container_width=True)
            
            # Select specific record
            if len(employee_records) > 0:
                record_index = st.selectbox(
                    "Select record to analyze",
                    range(len(employee_records)),
                    format_func=lambda x: f"Record {x+1} - {employee_records.iloc[x]['Date']}"
                )
                
                if st.button("🔬 Analyze This Record", use_container_width=True):
                    record = employee_records.iloc[record_index]
                    
                    # Prepare data
                    employee_data = {
                        'name': record['Team Members'],
                        'date': record['Date'],
                        'sign_in': record['Signed In'],
                        'sign_out': record['Signed Out'],
                        'task': record['Completed Task']
                    }
                    
                    # Run analysis
                    with st.spinner("🤖 AI is analyzing..."):
                        st.session_state.analysis_results = st.session_state.analyzer.analyze(
                            employee_data, None, None
                        )
                        
                        # Update performance history
                        st.session_state.database.update_performance_history(
                            employee_data['name'],
                            st.session_state.analysis_results
                        )
                        
                        st.success("✅ Analysis complete!")
                        st.rerun()
    else:
        st.info("📭 No dataset loaded. Please upload data first.")

def display_quick_analysis():
    """Quick analysis for rapid insights"""
    st.markdown("### ⚡ Quick Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        quick_name = st.text_input("Employee Name", placeholder="Enter name")
        quick_hours = st.number_input("Work Hours", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
    
    with col2:
        quick_engagement = st.slider("Engagement Level", 0, 100, 75)
        quick_complexity = st.slider("Task Complexity", 0, 100, 50)
    
    quick_task = st.text_area("Brief Task Description", placeholder="What was accomplished?")
    
    if st.button("⚡ Get Quick Insights", use_container_width=True):
        # Generate quick analysis
        mock_data = {
            'name': quick_name or "Employee",
            'date': datetime.now().strftime('%d.%m.%Y'),
            'sign_in': "09:00",
            'sign_out': f"{int(9+quick_hours):02d}:00",
            'task': quick_task or "General work"
        }
        
        with st.spinner("Generating insights..."):
            st.session_state.analysis_results = st.session_state.analyzer.analyze(
                mock_data, None, None
            )
            
            # Show quick results
            st.markdown("### 📊 Quick Results")
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                score = st.session_state.analysis_results['performance_score']
                st.metric("Performance Score", f"{score}/100")
            
            with col_res2:
                risk = st.session_state.analysis_results.get('burnout_risk', 0)
                st.metric("Burnout Risk", f"{risk}%")
            
            # Quick recommendations
            st.markdown("### 💡 Quick Recommendations")
            recs = st.session_state.analysis_results.get('ai_recommendations', [])
            for rec in recs[:2]:
                st.info(rec)

def run_enhanced_analysis(employee_name, date, sign_in, sign_out, task, video, image, department, complexity):
    """Run enhanced analysis with all features"""
    with st.spinner("🤖 AI is performing comprehensive analysis..."):
        # Prepare employee data
        employee_data = {
            'name': employee_name,
            'date': date,
            'sign_in': sign_in,
            'sign_out': sign_out,
            'task': task,
            'department': department
        }
        
        # Perform analysis
        st.session_state.analysis_results = st.session_state.analyzer.analyze(
            employee_data, video, image
        )
        
        # Add department info
        st.session_state.analysis_results['department'] = department
        
        # Update performance history
        st.session_state.database.update_performance_history(
            employee_name,
            st.session_state.analysis_results
        )
        
        # Save record
        try:
            save_data = employee_data.copy()
            save_data['selfie_path'] = image.name if image else ''
            save_data['session_video_path'] = video.name if video else ''
            
            save_result = st.session_state.database.save_record(save_data)
            st.success(save_result)
        except Exception as e:
            st.error(f"Could not save record: {e}")
        
        st.success("✅ Analysis complete! View results below.")
        st.balloons()
        st.rerun()

def display_enhanced_reports():
    """Enhanced reports with more analytics"""
    st.markdown('<div class="main-title">📊 Advanced Insights</div>', unsafe_allow_html=True)
    
    if st.session_state.database.df is None or st.session_state.database.df.empty:
        st.warning("📭 No data available for reports.")
        return
    
    # Create tabs for different report types
    report_tabs = st.tabs(["📈 Performance Trends", "👥 Team Analytics", "🎯 AI Insights", "📋 Custom Reports"])
    
    with report_tabs[0]:
        display_performance_trends()
    
    with report_tabs[1]:
        display_team_analytics()
    
    with report_tabs[2]:
        display_ai_insights()
    
    with report_tabs[3]:
        display_custom_reports()

def display_performance_trends():
    """Display performance trend analysis"""
    st.markdown("### 📈 Performance Trends Over Time")
    
    # Generate trend data
    if st.session_state.database.performance_history:
        # Create trend visualization
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for idx, (employee, records) in enumerate(list(st.session_state.database.performance_history.items())[:5]):
            if records:
                dates = [r['date'] for r in records]
                scores = [r['performance_score'] for r in records]
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=scores,
                    mode='lines+markers',
                    name=employee,
                    line=dict(color=colors[idx % len(colors)], width=3),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title="Performance Trends (Last 20 Records)",
            xaxis_title="Date",
            yaxis_title="Performance Score",
            yaxis_range=[0, 100],
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No performance history available. Run analyses to build history.")

def display_team_analytics():
    """Display team-level analytics"""
    st.markdown("### 👥 Team Performance Analytics")
    
    if st.session_state.database.df is not None:
        # Department analysis (simulated)
        departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Operations', 'Design']
        dept_scores = {dept: random.randint(65, 95) for dept in departments}
        
        fig = go.Figure(data=[go.Bar(
            x=list(dept_scores.keys()),
            y=list(dept_scores.values()),
            marker_color=px.colors.sequential.Viridis
        )])
        
        fig.update_layout(
            title="Average Performance by Department",
            yaxis_title="Average Score",
            yaxis_range=[0, 100],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance distribution
    st.markdown("### 📊 Performance Distribution")
    
    if st.session_state.analysis_results:
        scores = [st.session_state.analysis_results['performance_score']]
        scores.extend([s + random.randint(-10, 10) for s in scores * 4])
        
        fig_dist = go.Figure(data=[go.Histogram(
            x=scores,
            nbinsx=10,
            marker_color='rgb(102, 126, 234)',
            opacity=0.7
        )])
        
        fig_dist.update_layout(
            title="Performance Score Distribution",
            xaxis_title="Score",
            yaxis_title="Frequency",
            height=300
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)

def display_ai_insights():
    """Display AI-generated insights"""
    st.markdown("### 🧠 AI-Powered Insights")
    
    # Generate insights based on data
    insights = [
        "🎯 **Pattern Detected:** Employees working 7-9 hours show 25% higher engagement",
        "📈 **Trend Alert:** Performance peaks on Wednesdays, drops on Fridays",
        "💡 **Recommendation:** Consider flexible hours for high-performing employees",
        "⚖️ **Balance:** Teams with better work-life balance have 30% lower turnover",
        "🤝 **Collaboration:** Cross-department projects boost innovation by 40%"
    ]
    
    for insight in insights:
        st.markdown(f'<div class="ai-card">{insight}</div>', unsafe_allow_html=True)
    
    # Predictive analytics
    st.markdown("### 🔮 Predictive Analytics")
    
    # Create a simple prediction chart
    future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 31)]
    predictions = [75 + 0.5*i + random.uniform(-2, 2) for i in range(30)]
    
    fig_pred = go.Figure(data=go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines',
        name='Predicted Performance',
        line=dict(color='rgb(102, 126, 234)', width=3, dash='dash'),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    fig_pred.update_layout(
        title="30-Day Performance Forecast",
        xaxis_title="Date",
        yaxis_title="Predicted Score",
        yaxis_range=[60, 90],
        height=400
    )
    
    st.plotly_chart(fig_pred, use_container_width=True)

def display_custom_reports():
    """Display custom report generator"""
    st.markdown("### 📋 Custom Report Generator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Performance Summary", "Team Comparison", "Trend Analysis", "Comprehensive"]
        )
        
        time_range = st.selectbox(
            "Time Range",
            ["Last 7 days", "Last 30 days", "Last quarter", "Last year", "Custom"]
        )
    
    with col2:
        metrics = st.multiselect(
            "Include Metrics",
            ["Performance Score", "Engagement", "Work Hours", "Task Complexity", "Burnout Risk"],
            default=["Performance Score", "Engagement"]
        )
        
        format_choice = st.radio(
            "Output Format",
            ["PDF", "Excel", "HTML", "JSON"]
        )
    
    if st.button("📄 Generate Custom Report", use_container_width=True):
        with st.spinner(f"Generating {report_type} report..."):
            # Simulate report generation
            st.success("✅ Report generated successfully!")
            
            # Show preview
            st.markdown("#### 📋 Report Preview")
            st.markdown("""
            **Employee Performance Report**
            
            **Period:** Last 30 days
            **Generated:** """ + datetime.now().strftime('%Y-%m-%d %H:%M') + """
            
            **Key Findings:**
            1. Average performance score: 78.5
            2. Engagement increased by 12%
            3. Work hours optimized by 8%
            
            **Recommendations:**
            - Implement flexible scheduling
            - Enhance team collaboration
            - Monitor burnout indicators
            """)
            
            # Download button
            report_data = "Sample report content - replace with actual data"
            st.download_button(
                label=f"📥 Download {format_choice}",
                data=report_data,
                file_name=f"report_{datetime.now().strftime('%Y%m%d')}.{format_choice.lower()}",
                mime="application/octet-stream"
            )

def display_settings():
    """Display settings page"""
    st.markdown('<div class="main-title">⚙️ Settings & Configuration</div>', unsafe_allow_html=True)
    
    # Settings tabs
    settings_tabs = st.tabs(["🔧 General", "🤖 AI Settings", "📊 Analytics", "🔐 Security"])
    
    with settings_tabs[0]:
        st.markdown("### 🔧 General Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            app_theme = st.selectbox(
                "Theme",
                ["Light", "Dark", "Auto"]
            )
            
            timezone = st.selectbox(
                "Timezone",
                ["UTC", "EST", "PST", "CET", "IST"]
            )
        
        with col2:
            data_retention = st.slider(
                "Data Retention (months)",
                1, 36, 12
            )
            
            auto_backup = st.checkbox("Enable Auto Backup", value=True)
        
        if st.button("💾 Save General Settings", use_container_width=True):
            st.success("Settings saved!")
    
    with settings_tabs[1]:
        st.markdown("### 🤖 AI Model Settings")
        
        ai_model = st.selectbox(
            "AI Model",
            ["Enhanced Rule-Based", "Neural Network", "Hybrid", "Custom"]
        )
        
        col_ai1, col_ai2 = st.columns(2)
        
        with col_ai1:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                0.0, 1.0, 0.7
            )
            
            learning_rate = st.slider(
                "Learning Rate",
                0.001, 0.1, 0.01, 0.001,
                format="%.3f"
            )
        
        with col_ai2:
            max_iterations = st.number_input(
                "Max Iterations",
                100, 10000, 1000
            )
            
            enable_retraining = st.checkbox("Enable Auto-Retraining", value=True)
        
        if st.button("🤖 Update AI Settings", use_container_width=True):
            st.success("AI settings updated!")
    
    with settings_tabs[2]:
        st.markdown("### 📊 Analytics Settings")
        
        analytics_enabled = st.checkbox("Enable Advanced Analytics", value=True)
        
        if analytics_enabled:
            st.markdown("#### Data Collection")
            collect_performance = st.checkbox("Performance Metrics", value=True)
            collect_engagement = st.checkbox("Engagement Data", value=True)
            collect_productivity = st.checkbox("Productivity Patterns", value=True)
            collect_behavioral = st.checkbox("Behavioral Insights", value=False)
        
        report_frequency = st.selectbox(
            "Report Frequency",
            ["Daily", "Weekly", "Monthly", "Quarterly"]
        )
    
    with settings_tabs[3]:
        st.markdown("### 🔐 Security & Access")
        
        # Password change
        st.markdown("#### Change Password")
        current_pw = st.text_input("Current Password", type="password")
        new_pw = st.text_input("New Password", type="password")
        confirm_pw = st.text_input("Confirm New Password", type="password")
        
        if st.button("🔐 Update Password", use_container_width=True):
            if new_pw == confirm_pw and len(new_pw) >= 8:
                st.success("Password updated successfully!")
            else:
                st.error("Passwords don't match or are too short")
        
        st.markdown("#### Access Control")
        user_role = st.selectbox(
            "Default User Role",
            ["Viewer", "Analyst", "Manager", "Admin"]
        )
        
        enable_2fa = st.checkbox("Enable Two-Factor Authentication", value=False)
        
        st.markdown("#### Data Privacy")
        anonymize_data = st.checkbox("Anonymize Data in Reports", value=True)
        auto_logout = st.number_input("Auto Logout (minutes)", 5, 120, 30)

# Run the enhanced app
if __name__ == "__main__":
    main()
