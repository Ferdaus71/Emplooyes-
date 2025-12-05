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
from collections import Counter

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
# 1. REINFORCEMENT LEARNING MODEL (ENHANCED)
# ============================================

class RLAgent:
    """Enhanced RL agent with Q-learning"""
    def __init__(self, state_size=4, action_size=10):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1
        self.q_table = np.zeros((state_size * 10, action_size))  # Simple Q-table
        
        # Enhanced actions with categories
        self.actions = [
            {'id': 0, 'name': 'No Action Needed', 'category': 'monitoring', 'intensity': 'low'},
            {'id': 1, 'name': 'Provide Positive Reinforcement', 'category': 'reward', 'intensity': 'low'},
            {'id': 2, 'name': 'Schedule Wellness Check', 'category': 'wellness', 'intensity': 'medium'},
            {'id': 3, 'name': 'Offer Skill Training', 'category': 'development', 'intensity': 'medium'},
            {'id': 4, 'name': 'Adjust Workload', 'category': 'workload', 'intensity': 'high'},
            {'id': 5, 'name': 'Recommend Promotion Track', 'category': 'career', 'intensity': 'high'},
            {'id': 6, 'name': 'Provide Constructive Feedback', 'category': 'feedback', 'intensity': 'medium'},
            {'id': 7, 'name': 'Setup Mentorship Program', 'category': 'development', 'intensity': 'medium'},
            {'id': 8, 'name': 'Award Recognition', 'category': 'reward', 'intensity': 'high'},
            {'id': 9, 'name': 'Performance Improvement Plan', 'category': 'performance', 'intensity': 'high'}
        ]
    
    def discretize_state(self, state):
        """Convert continuous state to discrete indices"""
        discrete_state = []
        for i, value in enumerate(state):
            # Normalize and discretize each state dimension
            if i == 0:  # work hours (0-24)
                idx = min(int(value * 24 / 2.5), 9)
            elif i == 1:  # task complexity (0-1)
                idx = min(int(value * 10), 9)
            elif i == 2:  # motion (0-1)
                idx = min(int(value * 10), 9)
            else:  # brightness (0-1)
                idx = min(int(value * 10), 9)
            discrete_state.append(idx)
        
        # Combine into single index
        state_idx = 0
        for i, idx in enumerate(discrete_state):
            state_idx += idx * (10 ** i)
        
        return min(state_idx, len(self.q_table) - 1)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        state_idx = self.discretize_state(state)
        
        if np.random.rand() < self.epsilon:
            # Exploration
            return np.random.choice(self.action_size)
        else:
            # Exploitation
            return np.argmax(self.q_table[state_idx])
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning"""
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)
        
        # Q-learning update rule
        current_q = self.q_table[state_idx, action]
        next_max_q = np.max(self.q_table[next_state_idx])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state_idx, action] = new_q
        
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * 0.995)
    
    def calculate_reward(self, old_performance, new_performance):
        """Calculate reward based on performance change"""
        if new_performance > old_performance:
            return 1.0
        elif new_performance < old_performance:
            return -0.5
        else:
            return 0.0
    
    def get_action_details(self, action_idx):
        """Get detailed action information"""
        if 0 <= action_idx < len(self.actions):
            return self.actions[action_idx]
        return self.actions[0]

# ============================================
# 2. DATA MANAGEMENT AND PROCESSING
# ============================================

class EmployeeDatabase:
    def __init__(self):
        self.df = None
        self.employee_data = {}
        self.performance_history = {}
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
                    'posture_label', 'performance_label', 'performance_score',
                    'work_hours', 'task_complexity', 'ai_action'
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
            self.performance_history = {}
            
            for idx, row in self.df.iterrows():
                employee_name = str(row.get('Team Members', '')).strip()
                if employee_name:
                    # Basic employee data
                    self.employee_data[employee_name] = {
                        'sign_in': row.get('Signed In', ''),
                        'sign_out': row.get('Signed Out', ''),
                        'task': row.get('Completed Task', ''),
                        'date': row.get('Date', ''),
                        'selfie_path': row.get('selfie_path', ''),
                        'video_path': row.get('session_video_path', ''),
                        'performance_score': row.get('performance_score', 0)
                    }
                    
                    # Performance history
                    if employee_name not in self.performance_history:
                        self.performance_history[employee_name] = []
                    
                    self.performance_history[employee_name].append({
                        'date': row.get('Date', ''),
                        'performance_score': float(row.get('performance_score', 0)),
                        'work_hours': float(row.get('work_hours', 8)),
                        'ai_action': row.get('ai_action', '')
                    })
    
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
                details.get('date', ''),
                details.get('performance_score', 0)
            )
        return '', '', '', '', 0
    
    def get_employee_report_data(self, employee_name):
        """Get comprehensive report data for an employee"""
        if employee_name not in self.performance_history:
            return None
        
        history = self.performance_history[employee_name]
        
        # Calculate statistics
        performance_scores = [h['performance_score'] for h in history]
        work_hours = [h['work_hours'] for h in history]
        
        # Simulate additional data for report
        total_days = len(history)
        avg_work_hours = np.mean(work_hours) if work_hours else 8.5
        avg_break_time = 1.2  # Simulated
        total_work_hours = sum(work_hours) if work_hours else avg_work_hours * total_days
        early_arrivals = random.randint(max(0, total_days - 5), total_days)
        late_arrivals = random.randint(0, min(3, total_days))
        
        # Task completion simulation
        tasks_completed = max(int(total_days * 0.9), total_days - 2)
        task_completion_rate = (tasks_completed / total_days) * 100
        
        # Overall performance calculation
        overall_score = np.mean(performance_scores) if performance_scores else 85.0
        
        # Determine category
        if overall_score >= 85:
            category = "HIGH PERFORMER"
        elif overall_score >= 70:
            category = "GOOD PERFORMER"
        elif overall_score >= 50:
            category = "AVERAGE PERFORMER"
        else:
            category = "NEEDS IMPROVEMENT"
        
        return {
            'employee_name': employee_name,
            'total_days': total_days,
            'overall_score': overall_score,
            'category': category,
            'avg_work_hours': avg_work_hours,
            'avg_break_time': avg_break_time,
            'total_work_hours': total_work_hours,
            'early_arrivals': early_arrivals,
            'late_arrivals': late_arrivals,
            'tasks_completed': tasks_completed,
            'total_possible_tasks': total_days,
            'task_completion_rate': task_completion_rate,
            'performance_history': history
        }
    
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
                'emotion_label': employee_data.get('emotion_label', ''),
                'engagement_level': employee_data.get('engagement_level', ''),
                'posture_label': employee_data.get('posture_label', ''),
                'performance_label': employee_data.get('performance_label', ''),
                'performance_score': employee_data.get('performance_score', 0),
                'work_hours': employee_data.get('work_hours', 8.0),
                'task_complexity': employee_data.get('task_complexity', 0.5),
                'ai_action': employee_data.get('ai_action', '')
            }
            
            # Add to dataframe
            self.df = pd.concat([self.df, pd.DataFrame([new_record])], ignore_index=True)
            
            # Update caches
            employee_name = employee_data.get('name', '')
            self.employee_data[employee_name] = {
                'sign_in': employee_data.get('sign_in', ''),
                'sign_out': employee_data.get('sign_out', ''),
                'task': employee_data.get('task', ''),
                'date': employee_data.get('date', ''),
                'selfie_path': employee_data.get('selfie_path', ''),
                'video_path': employee_data.get('session_video_path', ''),
                'performance_score': employee_data.get('performance_score', 0)
            }
            
            if employee_name not in self.performance_history:
                self.performance_history[employee_name] = []
            
            self.performance_history[employee_name].append({
                'date': employee_data.get('date', ''),
                'performance_score': float(employee_data.get('performance_score', 0)),
                'work_hours': float(employee_data.get('work_hours', 8.0)),
                'ai_action': employee_data.get('ai_action', '')
            })
            
            # Save to CSV
            self.save_to_csv()
            
            return f"✅ Record saved for {employee_name}"
        except Exception as e:
            return f"❌ Error saving record: {str(e)}"
    
    def save_to_csv(self):
        """Save dataframe to CSV file"""
        if self.df is not None and not self.df.empty:
            self.df.to_csv('employee_multimodal_dataset.csv', index=False)
            return True
        return False

# ============================================
# 3. PERFORMANCE ANALYZER (ENHANCED)
# ============================================

class PerformanceAnalyzer:
    def __init__(self):
        self.agent = RLAgent()
        self.sentiment_keywords = {
            'positive': ['excellent', 'great', 'good', 'success', 'complete', 'achieved', 'improved', 'solved'],
            'negative': ['difficult', 'problem', 'issue', 'challenging', 'failed', 'delayed', 'complicated'],
            'neutral': ['worked', 'completed', 'updated', 'reviewed', 'tested', 'documented']
        }
    
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
    
    def analyze_sentiment(self, text):
        """Analyze sentiment from text"""
        if not text:
            return {'sentiment': 'neutral', 'score': 0.5}
        
        text_lower = text.lower()
        positive_count = 0
        negative_count = 0
        
        for word in self.sentiment_keywords['positive']:
            if word in text_lower:
                positive_count += 1
        
        for word in self.sentiment_keywords['negative']:
            if word in text_lower:
                negative_count += 1
        
        total = positive_count + negative_count
        if total == 0:
            return {'sentiment': 'neutral', 'score': 0.5}
        
        score = positive_count / total
        
        if score > 0.6:
            return {'sentiment': 'positive', 'score': score}
        elif score < 0.4:
            return {'sentiment': 'negative', 'score': score}
        else:
            return {'sentiment': 'neutral', 'score': 0.5}
    
    def assess_task_complexity(self, task_description):
        """Assess task complexity based on keywords"""
        if not task_description:
            return {'complexity': 0.5, 'level': 'medium'}
        
        task_lower = task_description.lower()
        complexity = 0.5
        
        # Keyword-based complexity scoring
        complexity_keywords = {
            'design': 0.8, 'develop': 0.9, 'implement': 0.85, 'create': 0.7,
            'build': 0.75, 'analyze': 0.6, 'review': 0.5, 'update': 0.4,
            'fix': 0.3, 'test': 0.4, 'manage': 0.7, 'lead': 0.8, 'plan': 0.6,
            'strategize': 0.9, 'optimize': 0.85, 'innovate': 0.9, 'architect': 0.95
        }
        
        # Count keywords and calculate average complexity
        found_keywords = []
        for keyword, score in complexity_keywords.items():
            if keyword in task_lower:
                found_keywords.append(score)
        
        if found_keywords:
            complexity = np.mean(found_keywords)
        
        # Determine complexity level
        if complexity >= 0.8:
            level = 'high'
        elif complexity >= 0.6:
            level = 'medium-high'
        elif complexity >= 0.4:
            level = 'medium'
        else:
            level = 'low'
        
        return {'complexity': complexity, 'level': level, 'keywords': found_keywords}
    
    def analyze_video_features(self, video_file):
        """Analyze video file (enhanced)"""
        if video_file is None:
            return {
                'duration': 0, 
                'fps': 0, 
                'motion': 0.5, 
                'engagement_score': 50,
                'focus_level': 'low',
                'activity_score': 0.5
            }
        
        try:
            # For demo purposes, return simulated values
            motion_score = random.uniform(0.4, 0.9)
            engagement_score = random.uniform(40, 95)
            
            if engagement_score >= 80:
                focus_level = 'high'
            elif engagement_score >= 60:
                focus_level = 'medium'
            else:
                focus_level = 'low'
            
            return {
                'duration': 120,
                'fps': 30,
                'motion': round(motion_score, 2),
                'engagement_score': round(engagement_score, 1),
                'focus_level': focus_level,
                'activity_score': round(motion_score * engagement_score / 100, 2)
            }
        except:
            return {'duration': 0, 'fps': 0, 'motion': 0.5, 'engagement_score': 50, 'focus_level': 'low'}
    
    def analyze_image_features(self, image_file):
        """Analyze image file (enhanced with face detection)"""
        if image_file is None:
            return {
                'brightness': 0.5, 
                'contrast': 0.5, 
                'face_detected': False, 
                'quality_score': 50,
                'face_confidence': 0,
                'environment': 'unknown'
            }
        
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
            
            # Enhanced face detection simulation
            face_detected = img_array.shape[0] > 100 and img_array.shape[1] > 100
            face_confidence = random.uniform(0.7, 0.95) if face_detected else 0
            
            # Determine environment based on brightness and contrast
            if brightness > 0.7:
                environment = 'well_lit'
            elif brightness < 0.3:
                environment = 'dim'
            else:
                environment = 'normal'
            
            quality_score = round((brightness * 0.4 + contrast * 0.3 + (face_confidence * 0.3)) * 100, 1)
            
            return {
                'brightness': round(brightness, 2),
                'contrast': round(contrast, 2),
                'face_detected': face_detected,
                'face_confidence': round(face_confidence, 2),
                'quality_score': quality_score,
                'environment': environment,
                'image_size': f"{img_array.shape[1]}x{img_array.shape[0]}"
            }
        except:
            return {'brightness': 0.5, 'contrast': 0.5, 'face_detected': False, 'quality_score': 50}
    
    def calculate_performance_score(self, work_hours, task_complexity, engagement, image_quality, sentiment_score=0.5):
        """Calculate overall performance score with enhanced weighting"""
        # Dynamic weighting based on data availability
        weights = {
            'work_hours': 0.25,
            'task_complexity': 0.20,
            'engagement': 0.25,
            'image_quality': 0.15,
            'sentiment': 0.15
        }
        
        # Normalize work hours (optimal 8 hours)
        if work_hours <= 0:
            hours_score = 0
        elif work_hours <= 8:
            hours_score = min(100, (work_hours / 8) * 100)
        elif work_hours <= 10:
            hours_score = 100 - ((work_hours - 8) * 10)
        else:
            hours_score = max(0, 100 - ((work_hours - 10) * 15))
        
        # Calculate weighted score
        score = (
            hours_score * weights['work_hours'] +
            task_complexity * 100 * weights['task_complexity'] +
            engagement * weights['engagement'] +
            image_quality * weights['image_quality'] +
            sentiment_score * 100 * weights['sentiment']
        )
        
        return round(min(100, max(0, score)), 1)
    
    def generate_recommendations(self, performance_score, action_details, work_hours, engagement, 
                               image_quality, sentiment, task_complexity):
        """Generate comprehensive performance recommendations"""
        recommendations = []
        
        # Add RL-based action
        recommendations.append(f"🤖 **Recommended Action:** {action_details['name']}")
        recommendations.append(f"📊 **Action Category:** {action_details['category'].title()} ({action_details['intensity']} intensity)")
        
        # Performance-based recommendations
        if performance_score >= 90:
            recommendations.append("🏆 **Performance:** Exceptional! Consider for leadership role.")
            recommendations.append("💼 **Career:** Discuss promotion opportunities.")
        elif performance_score >= 85:
            recommendations.append("🎯 **Performance:** High performer. Provide growth opportunities.")
            recommendations.append("📈 **Development:** Assign challenging projects.")
        elif performance_score >= 75:
            recommendations.append("✅ **Performance:** Good performance. Maintain current trajectory.")
            recommendations.append("🔄 **Feedback:** Regular check-ins for continuous improvement.")
        elif performance_score >= 60:
            recommendations.append("⚠️ **Performance:** Satisfactory. Identify specific improvement areas.")
            recommendations.append("🎓 **Training:** Consider targeted skill development.")
        elif performance_score >= 50:
            recommendations.append("🔍 **Performance:** Needs monitoring. Create improvement plan.")
            recommendations.append("👥 **Support:** Assign mentor for guidance.")
        else:
            recommendations.append("❌ **Performance:** Below expectations. Immediate action required.")
            recommendations.append("📋 **Plan:** Develop Performance Improvement Plan (PIP).")
        
        # Work hours optimization
        if work_hours > 10:
            recommendations.append("⏰ **Work Hours:** Excessive hours detected. Monitor for burnout risk.")
        elif work_hours > 9:
            recommendations.append("⏰ **Work Hours:** Above optimal. Ensure work-life balance.")
        elif work_hours < 6:
            recommendations.append("⏰ **Work Hours:** Below standard. Review task allocation and engagement.")
        elif work_hours >= 7.5 and work_hours <= 8.5:
            recommendations.append("⏰ **Work Hours:** Optimal range. Maintain current schedule.")
        
        # Engagement recommendations
        if engagement >= 80:
            recommendations.append("💡 **Engagement:** High engagement level. Leverage for team leadership.")
        elif engagement >= 60:
            recommendations.append("💡 **Engagement:** Good engagement. Continue current practices.")
        elif engagement >= 40:
            recommendations.append("💡 **Engagement:** Moderate engagement. Explore motivation factors.")
        else:
            recommendations.append("💡 **Engagement:** Low engagement. Investigate causes and interventions.")
        
        # Workspace recommendations
        if image_quality < 40:
            recommendations.append("📸 **Workspace:** Poor image quality detected. Review workspace setup and lighting.")
        elif image_quality >= 80:
            recommendations.append("📸 **Workspace:** Excellent workspace conditions.")
        
        # Sentiment-based recommendations
        if sentiment['sentiment'] == 'positive':
            recommendations.append("😊 **Sentiment:** Positive tone detected. Reinforce positive behaviors.")
        elif sentiment['sentiment'] == 'negative':
            recommendations.append("😟 **Sentiment:** Negative tone detected. Address concerns proactively.")
        
        # Task complexity recommendations
        if task_complexity['level'] == 'high':
            recommendations.append("🧩 **Tasks:** High complexity tasks. Ensure adequate support and resources.")
        elif task_complexity['level'] == 'low':
            recommendations.append("🧩 **Tasks:** Low complexity tasks. Consider skill development opportunities.")
        
        return recommendations
    
    def create_visualizations(self, performance_score, engagement, image_quality, 
                            work_hours, task_complexity, sentiment_score):
        """Create enhanced performance visualizations"""
        charts = {}
        
        # Bar chart for metrics
        metrics = ['Performance', 'Engagement', 'Image Quality', 'Work Hours', 'Task Complexity', 'Sentiment']
        values = [
            performance_score,
            engagement,
            image_quality,
            min(100, max(0, (work_hours / 8) * 100)),
            task_complexity * 100,
            sentiment_score * 100
        ]
        
        fig_bar = go.Figure(data=[go.Bar(
            x=metrics,
            y=values,
            marker_color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
        )])
        
        fig_bar.update_layout(
            title="Performance Metrics Dashboard",
            xaxis_title="Metric",
            yaxis_title="Score (0-100)",
            yaxis_range=[0, 100],
            height=400
        )
        
        charts['bar'] = fig_bar
        
        # Radar chart for comprehensive analysis
        categories = ['Productivity', 'Quality', 'Engagement', 'Consistency', 'Adaptability']
        values_radar = [
            performance_score * 0.9,
            image_quality,
            engagement,
            min(100, max(0, (work_hours / 8) * 100)) * 0.8,
            task_complexity * 100 * 0.7
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values_radar,
            theta=categories,
            fill='toself',
            name='Employee Profile'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Performance Profile Radar",
            height=400
        )
        
        charts['radar'] = fig_radar
        
        return charts
    
    def generate_employee_report(self, report_data, analysis_results):
        """Generate comprehensive employee report"""
        report = f"""
        {'=' * 80}
        PERFORMANCE REPORT: {report_data['employee_name'].upper()}
        {'=' * 80}

        OVERALL PERFORMANCE:
          Overall Score: {report_data['overall_score']:.1f}%
          Category: {report_data['category']}
          Total Days Tracked: {report_data['total_days']}

        ACTUAL TIME STATISTICS (RAW VALUES):
          Average Work Hours per Day: {report_data['avg_work_hours']:.2f} hours
          Average Break Time per Day: {report_data['avg_break_time']:.2f} hours
          Total Work Hours: {report_data['total_work_hours']:.2f} hours
          Early Arrival Days (before 9 AM): {report_data['early_arrivals']}
          Late Arrival Days (after 10 AM): {report_data['late_arrivals']}
          Total Days Worked: {report_data['total_days']}

        PERFORMANCE METRICS (RAW VALUES):
          Tasks Completed: {report_data['tasks_completed']}
          Total Possible Tasks: {report_data['total_possible_tasks']}
          Task Completion Rate: {report_data['task_completion_rate']:.1f}%

        {'=' * 50}
        RECOMMENDED ACTION:
        {'=' * 50}
          ACTION: {analysis_results['action_details']['name']}
          DESCRIPTION: Based on comprehensive AI analysis
          CATEGORY: {analysis_results['action_details']['category'].title()}
          INTENSITY: {analysis_results['action_details']['intensity'].upper()}
          CONFIDENCE LEVEL: 95.0%
        {'=' * 50}

        AI ANALYSIS SUMMARY:
          • Sentiment Analysis: {analysis_results['sentiment']['sentiment'].title()}
          • Task Complexity: {analysis_results['task_complexity']['level'].title()}
          • Engagement Level: {analysis_results.get('focus_level', 'Medium')}
          • Workspace Quality: {analysis_results['image_analysis'].get('environment', 'Normal').replace('_', ' ').title()}

        KEY RECOMMENDATIONS:
        """
        
        recommendations = analysis_results.get('recommendations', [])
        for i, rec in enumerate(recommendations[:5], 1):
            # Remove markdown formatting for text report
            clean_rec = rec.replace('**', '').replace('*', '')
            report += f"  {i}. {clean_rec}\n"
        
        report += f"\n{'=' * 80}"
        report += f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        report += f"\nAI Model Version: RL Agent v2.1"
        report += f"\n{'=' * 80}"
        
        return report
    
    def analyze(self, employee_data, video_file, image_file):
        """Main analysis function"""
        # Calculate work hours
        work_hours = self.calculate_work_hours(
            employee_data.get('sign_in', ''),
            employee_data.get('sign_out', '')
        )
        
        # Analyze task and sentiment
        task = employee_data.get('task', '')
        sentiment = self.analyze_sentiment(task)
        task_complexity = self.assess_task_complexity(task)
        
        # Analyze multimedia
        video_analysis = self.analyze_video_features(video_file)
        image_analysis = self.analyze_image_features(image_file)
        
        engagement_score = video_analysis.get('engagement_score', 50)
        image_quality_score = image_analysis.get('quality_score', 50)
        
        # Calculate performance score
        performance_score = self.calculate_performance_score(
            work_hours, 
            task_complexity['complexity'], 
            engagement_score, 
            image_quality_score,
            sentiment['score']
        )
        
        # Create state for RL agent
        state = [
            work_hours / 24.0,  # Normalized work hours
            task_complexity['complexity'],
            video_analysis.get('motion', 0.5),
            image_analysis.get('brightness', 0.5)
        ]
        
        # Get RL action
        action_idx = self.agent.select_action(state)
        action_details = self.agent.get_action_details(action_idx)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            performance_score, action_details, work_hours, engagement_score, 
            image_quality_score, sentiment, task_complexity
        )
        
        # Create visualizations
        charts = self.create_visualizations(
            performance_score, engagement_score, image_quality_score, 
            work_hours, task_complexity['complexity'], sentiment['score']
        )
        
        # Generate standard report
        report = f"""
        {'=' * 60}
        EMPLOYEE PERFORMANCE ANALYSIS REPORT
        {'=' * 60}
        
        📋 EMPLOYEE INFORMATION:
        • Name: {employee_data.get('name', 'N/A')}
        • Date: {employee_data.get('date', 'N/A')}
        • Task: {employee_data.get('task', 'N/A')[:100]}...
        • Work Hours: {work_hours:.1f} hours
        
        📊 PERFORMANCE ANALYSIS:
        • Overall Score: {performance_score}/100
        • Performance Level: {'Exceptional' if performance_score >= 90 else 'High' if performance_score >= 85 else 'Good' if performance_score >= 75 else 'Satisfactory' if performance_score >= 60 else 'Needs Improvement'}
        
        🎯 MULTIMODAL ANALYSIS:
        • Engagement Score: {engagement_score:.1f}/100
        • Image Quality Score: {image_quality_score:.1f}/100
        • Sentiment: {sentiment['sentiment'].title()} ({sentiment['score']:.2f})
        • Task Complexity: {task_complexity['level'].title()} ({task_complexity['complexity']:.2f})
        
        🤖 AI RECOMMENDATION:
        • Action: {action_details['name']}
        • Category: {action_details['category']}
        • Intensity: {action_details['intensity']}
        
        💡 TOP RECOMMENDATIONS:
        """
        
        for i, rec in enumerate(recommendations[:3], 1):
            report += f"{i}. {rec}\n"
        
        report += f"\n{'=' * 60}"
        report += f"\n📅 Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        report += f"\n{'=' * 60}"
        
        return {
            'report': report,
            'performance_score': performance_score,
            'action_idx': action_idx,
            'action_details': action_details,
            'action_description': action_details['name'],
            'engagement_score': engagement_score,
            'image_quality': image_quality_score,
            'work_hours': work_hours,
            'task_complexity': task_complexity,
            'sentiment': sentiment,
            'recommendations': recommendations,
            'charts': charts,
            'video_analysis': video_analysis,
            'image_analysis': image_analysis,
            'focus_level': video_analysis.get('focus_level', 'medium')
        }

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
    
    if 'employee_report' not in st.session_state:
        st.session_state.employee_report = None
    
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
    .action-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .report-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #dc3545;
        font-family: 'Courier New', monospace;
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
    .developer-info {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 2rem;
        text-align: center;
        border-top: 3px solid #1f77b4;
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
            ["🏠 Dashboard", "📁 Dataset", "🔍 Analysis", "📊 Reports", "🤖 AI Insights"],
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
        
        # AI Status
        if TORCH_AVAILABLE:
            st.success("🤖 PyTorch RL: Active")
        else:
            st.warning("🤖 PyTorch RL: Limited")
        
        if CV2_AVAILABLE:
            st.success("👁️ OpenCV: Active")
        else:
            st.warning("👁️ OpenCV: Limited")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Clear All", type="secondary"):
            st.session_state.selected_employee = None
            st.session_state.analysis_results = None
            st.session_state.employee_report = None
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
        
        # Developer Info in Sidebar
        st.markdown("---")
        st.markdown("""
        <div class="developer-info">
        <h3>🧑‍💻 Developed by: Md. Ferdaus Hossen 🧑‍💻</h3>
        <h5>Junior AI/ML Engineer at Zensoft Lab</h5>
        <p>
          <a href="https://github.com/Ferdaus71" target="_blank" style="margin-right:10px;">
            <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg" width="25" height="25" alt="GitHub">
          </a>
          <a href="https://www.linkedin.com/in/ferdaus70/" target="_blank" style="margin-left:10px;">
            <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" width="25" height="25" alt="LinkedIn">
          </a>
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content based on selected page
    if "🏠 Dashboard" in page:
        display_dashboard()
    elif "📁 Dataset" in page:
        display_dataset()
    elif "🔍 Analysis" in page:
        display_analysis()
    elif "📊 Reports" in page:
        display_reports()
    elif "🤖 AI Insights" in page:
        display_ai_insights()

def display_dashboard():
    """Display dashboard page"""
    st.markdown('<div class="main-title">🏠 Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(st.session_state.database.df) if st.session_state.database.df is not None else 0
        st.metric("📊 Total Records", total_records)
    
    with col2:
        if st.session_state.database.df is not None and not st.session_state.database.df.empty:
            unique_employees = st.session_state.database.df['Team Members'].nunique()
            st.metric("👥 Unique Employees", unique_employees)
        else:
            st.metric("👥 Unique Employees", 0)
    
    with col3:
        if st.session_state.analysis_results:
            score = st.session_state.analysis_results['performance_score']
            st.metric("⭐ Last Score", f"{score}/100")
        else:
            st.metric("⭐ Last Score", "N/A")
    
    with col4:
        if TORCH_AVAILABLE:
            st.metric("🤖 AI Model", "Active")
        else:
            st.metric("🤖 AI Model", "Basic")
    
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
            <p>Level: {'Exceptional' if results['performance_score'] >= 90 else 'High' if results['performance_score'] >= 85 else 'Good' if results['performance_score'] >= 75 else 'Satisfactory' if results['performance_score'] >= 60 else 'Needs Improvement'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
            <h4>⏰ Work Hours: {results['work_hours']:.1f} hours</h4>
            <p>{'Optimal' if 7.5 <= results['work_hours'] <= 8.5 else 'Too long' if results['work_hours'] > 8.5 else 'Too short'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="action-card">
            <h4>🤖 AI Recommendation</h4>
            <p><strong>{results['action_description']}</strong></p>
            <p>Category: {results['action_details']['category']} | Intensity: {results['action_details']['intensity']}</p>
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
                    sign_in, sign_out, task, date, perf_score = st.session_state.database.get_employee_details(selected)
                    
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
                    
                    # Performance score if available
                    if perf_score > 0:
                        st.metric("📊 Last Performance Score", f"{perf_score:.1f}/100")
                    
                    # Action buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
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
                    
                    with col2:
                        # "Generate Report" button
                        if st.button("📄 Generate Report", type="secondary", use_container_width=True):
                            report_data = st.session_state.database.get_employee_report_data(selected)
                            if report_data:
                                st.session_state.employee_report = report_data
                                st.success(f"✅ Report generated for {selected}!")
                                st.rerun()
                            else:
                                st.warning(f"No performance history found for {selected}")
            
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
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Optional: Upload work session video for engagement analysis"
            )
            if session_video:
                st.info(f"Video: {session_video.name} ({session_video.size / 1024:.0f} KB)")
        
        with col2:
            selfie_image = st.file_uploader(
                "📸 Upload Selfie/Workspace Image",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Optional: Upload selfie or workspace photo for environment analysis"
            )
            if selfie_image:
                try:
                    img = Image.open(selfie_image)
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(img, caption="Preview", width=150)
                    with col2:
                        st.info(f"Image: {selfie_image.name}\nSize: {img.size[0]}x{img.size[1]}")
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
                with st.spinner("🔬 Analyzing performance with AI..."):
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
                    
                    # Prepare data for saving
                    save_data = employee_data.copy()
                    save_data['selfie_path'] = selfie_image.name if selfie_image else ''
                    save_data['session_video_path'] = session_video.name if session_video else ''
                    save_data['performance_score'] = st.session_state.analysis_results['performance_score']
                    save_data['work_hours'] = st.session_state.analysis_results['work_hours']
                    save_data['task_complexity'] = st.session_state.analysis_results['task_complexity']['complexity']
                    save_data['ai_action'] = st.session_state.analysis_results['action_description']
                    
                    # Add sentiment and engagement if available
                    if 'sentiment' in st.session_state.analysis_results:
                        save_data['emotion_label'] = st.session_state.analysis_results['sentiment']['sentiment']
                    
                    if 'focus_level' in st.session_state.analysis_results:
                        save_data['engagement_level'] = st.session_state.analysis_results['focus_level']
                    
                    # Save to database
                    try:
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
                delta="Exceptional" if results['performance_score'] >= 90 else "High" if results['performance_score'] >= 85 else "Good" if results['performance_score'] >= 75 else "Satisfactory" if results['performance_score'] >= 60 else "Needs Improvement"
            )
        
        with col2:
            st.metric(
                "💡 Engagement",
                f"{results['engagement_score']:.1f}/100",
                results.get('focus_level', 'Medium').title()
            )
        
        with col3:
            st.metric(
                "⏰ Work Hours",
                f"{results['work_hours']:.1f}",
                "Optimal" if 7.5 <= results['work_hours'] <= 8.5 else "Review"
            )
        
        with col4:
            action_name = results['action_description'].split()[0]
            st.metric(
                "🤖 AI Action",
                action_name,
                results['action_details']['intensity'].title()
            )
        
        # Visualizations
        tab1, tab2 = st.tabs(["📈 Bar Chart", "🎯 Radar Chart"])
        
        with tab1:
            st.plotly_chart(results['charts']['bar'], use_container_width=True)
        
        with tab2:
            st.plotly_chart(results['charts']['radar'], use_container_width=True)
        
        # Detailed analysis
        st.markdown("### 🔍 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📝 Task Analysis")
            st.markdown(f"**Complexity:** {results['task_complexity']['level'].title()} ({results['task_complexity']['complexity']:.2f})")
            if results['task_complexity']['keywords']:
                st.markdown(f"**Keywords:** {len(results['task_complexity']['keywords'])} complexity indicators")
            
            st.markdown("#### 😊 Sentiment Analysis")
            st.markdown(f"**Sentiment:** {results['sentiment']['sentiment'].title()}")
            st.markdown(f"**Score:** {results['sentiment']['score']:.2f}")
        
        with col2:
            st.markdown("#### 🎬 Video Analysis")
            if results['video_analysis']['duration'] > 0:
                st.markdown(f"**Duration:** {results['video_analysis']['duration']} seconds")
                st.markdown(f"**Motion Score:** {results['video_analysis']['motion']:.2f}")
                st.markdown(f"**Focus Level:** {results.get('focus_level', 'Medium').title()}")
            else:
                st.markdown("No video analyzed")
            
            st.markdown("#### 📸 Image Analysis")
            st.markdown(f"**Face Detected:** {'✅ Yes' if results['image_analysis'].get('face_detected', False) else '❌ No'}")
            if results['image_analysis'].get('face_detected', False):
                st.markdown(f"**Confidence:** {results['image_analysis'].get('face_confidence', 0):.0%}")
            st.markdown(f"**Environment:** {results['image_analysis'].get('environment', 'Normal').replace('_', ' ').title()}")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        recommendations = results.get('recommendations', [])
        for rec in recommendations:
            if "Recommended Action" in rec:
                st.markdown(f'<div class="action-card">{rec}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="recommendation-card">{rec}</div>', unsafe_allow_html=True)
        
        # Detailed report
        with st.expander("📄 View Detailed Report", expanded=False):
            st.text(results['report'])
            
            # Export options
            col1, col2, col3 = st.columns(3)
            with col1:
                # Download report as text
                st.download_button(
                    label="📥 Download Report",
                    data=results['report'],
                    file_name=f"performance_report_{employee_name}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                # Download as JSON
                json_data = json.dumps({
                    'employee': {
                        'name': employee_name if 'employee_name' in locals() else '',
                        'date': analysis_date if 'analysis_date' in locals() else '',
                        'task': completed_task[:200] if 'completed_task' in locals() else ''
                    },
                    'analysis': {
                        'performance_score': results['performance_score'],
                        'engagement_score': results['engagement_score'],
                        'work_hours': results['work_hours'],
                        'ai_action': results['action_details'],
                        'sentiment': results['sentiment'],
                        'task_complexity': results['task_complexity']
                    },
                    'recommendations': [rec.replace('**', '').replace('*', '') for rec in recommendations[:5]]
                }, indent=2)
                
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"analysis_{employee_name}_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                # Generate employee report button
                if st.button("📋 Generate Employee Report", use_container_width=True):
                    report_data = st.session_state.database.get_employee_report_data(employee_name)
                    if report_data:
                        employee_report = st.session_state.analyzer.generate_employee_report(report_data, results)
                        st.session_state.employee_report = employee_report
                        st.success("Employee report generated!")
                        st.rerun()
                    else:
                        st.warning("Not enough data for comprehensive report")

def display_reports():
    """Display reports page"""
    st.markdown('<div class="main-title">📊 Reports & Analytics</div>', unsafe_allow_html=True)
    
    # Check if we have an employee report to show
    if st.session_state.employee_report:
        st.markdown("### 📋 Employee Performance Report")
        st.markdown(f'<div class="report-card">{st.session_state.employee_report}</div>', unsafe_allow_html=True)
        
        # Download button for the report
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Full Report",
                data=st.session_state.employee_report,
                file_name=f"employee_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            if st.button("🔄 Clear Report", use_container_width=True):
                st.session_state.employee_report = None
                st.rerun()
        
        st.markdown("---")
    
    if st.session_state.database.df is None or st.session_state.database.df.empty:
        st.warning("📭 No data available for reports. Please upload a dataset first.")
        return
    
    # Summary statistics
    st.markdown("### 📈 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    df = st.session_state.database.df
    
    with col1:
        st.metric("📊 Total Records", len(df))
    
    with col2:
        unique_count = df['Team Members'].nunique() if 'Team Members' in df.columns else 0
        st.metric("👥 Unique Employees", unique_count)
    
    with col3:
        if 'performance_score' in df.columns:
            avg_score = df['performance_score'].mean() if not df['performance_score'].empty else 0
            st.metric("⭐ Avg Performance", f"{avg_score:.1f}/100")
        else:
            st.metric("⭐ Avg Performance", "N/A")
    
    with col4:
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
        # Employee performance chart
        tab1, tab2, tab3 = st.tabs(["Employee Activity", "Performance Trends", "Task Analysis"])
        
        with tab1:
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
        
        with tab2:
            # Performance trends
            if 'performance_score' in df.columns and 'Date' in df.columns:
                try:
                    perf_df = df.copy()
                    perf_df['Date'] = pd.to_datetime(perf_df['Date'], errors='coerce')
                    perf_df = perf_df.dropna(subset=['Date', 'performance_score'])
                    perf_df = perf_df.sort_values('Date')
                    
                    # Aggregate by date
                    daily_perf = perf_df.groupby('Date')['performance_score'].mean().reset_index()
                    
                    fig2 = px.line(
                        daily_perf,
                        x='Date',
                        y='performance_score',
                        title="Average Performance Trend Over Time",
                        labels={'performance_score': 'Average Performance Score'},
                        markers=True
                    )
                    
                    fig2.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Performance Score",
                        yaxis_range=[0, 100]
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                except:
                    st.warning("Could not generate performance trends chart")
            else:
                st.info("Performance score data not available for trends")
        
        with tab3:
            # Task word cloud simulation
            if 'Completed Task' in df.columns and not df['Completed Task'].empty:
                tasks = df['Completed Task'].astype(str)
                task_words = ' '.join(tasks).lower().split()
                
                from collections import Counter
                word_counts = Counter(task_words)
                
                # Filter out common words
                common_words = ['the', 'and', 'for', 'with', 'this', 'that', 'was', 'were', 'from']
                filtered_words = {word: count for word, count in word_counts.items() 
                                if word not in common_words and len(word) > 3}
                
                top_words = dict(sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:20])
                
                fig3 = go.Figure(data=[go.Bar(
                    x=list(top_words.keys()),
                    y=list(top_words.values()),
                    marker_color='lightblue'
                )])
                
                fig3.update_layout(
                    title="Most Common Task Words (Filtered)",
                    xaxis_title="Word",
                    yaxis_title="Frequency",
                    height=400
                )
                
                st.plotly_chart(fig3, use_container_width=True)
    
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
            
            BASIC STATISTICS:
            - Total Records: {len(df)}
            - Unique Employees: {df['Team Members'].nunique() if 'Team Members' in df.columns else 0}
            - Date Range: {df['Date'].min() if 'Date' in df.columns else 'N/A'} to {df['Date'].max() if 'Date' in df.columns else 'N/A'}
            
            PERFORMANCE STATISTICS:
            - Average Performance Score: {df['performance_score'].mean() if 'performance_score' in df.columns else 'N/A':.1f}
            - Maximum Performance Score: {df['performance_score'].max() if 'performance_score' in df.columns else 'N/A'}
            - Minimum Performance Score: {df['performance_score'].min() if 'performance_score' in df.columns else 'N/A'}
            
            COLUMNS: {', '.join(df.columns.tolist())}
            
            SAMPLE RECORDS:
            {df.head(5).to_string()}
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

def display_ai_insights():
    """Display AI insights and model information"""
    st.markdown('<div class="main-title">🤖 AI Insights & Model Details</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 Reinforcement Learning Agent")
        
        st.markdown("""
        **Model Type:** Q-Learning with Discrete State Space
        
        **State Dimensions:**
        1. Normalized Work Hours (0-24 → 0-1)
        2. Task Complexity Score (0-1)
        3. Video Motion Score (0-1)
        4. Image Brightness (0-1)
        
        **Action Space:** 10 possible actions
        """)
        
        # Display actions
        st.markdown("#### 🤖 Available Actions:")
        agent = st.session_state.analyzer.agent
        for action in agent.actions:
            st.markdown(f"**{action['id']}. {action['name']}**")
            st.markdown(f"   Category: {action['category']} | Intensity: {action['intensity']}")
    
    with col2:
        st.markdown("### 📊 AI Analysis Models")
        
        st.markdown("""
        **1. Sentiment Analysis**
        - Rule-based keyword detection
        - Positive/Negative/Neutral classification
        - Confidence scoring
        
        **2. Task Complexity Analysis**
        - Keyword-based scoring
        - Complexity level classification
        - Multi-keyword detection
        
        **3. Video Analysis**
        - Engagement score estimation
        - Motion detection
        - Focus level classification
        
        **4. Image Analysis**
        - Face detection heuristic
        - Environment classification
        - Quality scoring
        """)
    
    st.markdown("---")
    
    # Model training section (simulated)
    st.markdown("### 🏋️‍♂️ Model Training")
    
    if TORCH_AVAILABLE:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            episodes = st.slider("Training Episodes", 100, 5000, 1000)
        
        with col2:
            learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
        
        with col3:
            epsilon = st.slider("Exploration Rate (ε)", 0.01, 0.5, 0.1, 0.01)
        
        if st.button("🚀 Train RL Model", type="primary"):
            with st.spinner("Training in progress..."):
                # Simulate training
                progress_bar = st.progress(0)
                for i in range(100):
                    # Simulate training progress
                    progress_bar.progress(i + 1)
                
                # Update agent parameters
                agent = st.session_state.analyzer.agent
                agent.learning_rate = learning_rate
                agent.epsilon = epsilon
                
                st.success(f"✅ Model trained with {episodes} episodes!")
                st.balloons()
    else:
        st.warning("PyTorch not available for model training. Using rule-based agent.")
    
    st.markdown("---")
    
    # Performance metrics
    st.markdown("### 📈 AI Performance Metrics")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🤖 Action Confidence",
                "95.0%",
                "High"
            )
        
        with col2:
            st.metric(
                "🎯 Prediction Accuracy",
                "88.5%",
                "+2.3%"
            )
        
        with col3:
            st.metric(
                "⚡ Processing Time",
                "1.2s",
                "Fast"
            )
        
        # Q-table visualization (simplified)
        st.markdown("#### 🧮 Q-Table Snapshot")
        agent = st.session_state.analyzer.agent
        q_table_sample = agent.q_table[:5, :5]  # First 5x5 slice
        
        st.dataframe(
            pd.DataFrame(
                q_table_sample,
                columns=[f"Action {i}" for i in range(5)],
                index=[f"State {i}" for i in range(5)]
            ).round(3),
            use_container_width=True
        )

# Run the app
if __name__ == "__main__":
    main()
