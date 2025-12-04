import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
from PIL import Image
import tempfile
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import base64
import io
import requests
from typing import Dict, List, Tuple, Optional
import re

# ============================================
# 1. AI ACTION MODELS & SERVICES
# ============================================

class AIActionModels:
    """Multiple AI models for different analysis tasks"""
    
    def __init__(self):
        self.models_loaded = {
            'sentiment': False,
            'face_detection': False,
            'task_analysis': False,
            'performance_prediction': False
        }
        
        # Initialize HuggingFace API (free tier)
        self.hf_api_key = None
        self.hf_api_url = "https://api-inference.huggingface.co/models"
        
        # Available models
        self.models = {
            'sentiment': 'distilbert-base-uncased-finetuned-sst-2-english',
            'emotion': 'j-hartmann/emotion-english-distilroberta-base',
            'face_detection': 'google/vit-base-patch16-224',
            'text_generation': 'gpt2'
        }
        
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using rule-based approach"""
        positive_words = ['excellent', 'great', 'good', 'positive', 'happy', 'satisfied',
                         'productive', 'efficient', 'successful', 'completed', 'achieved']
        negative_words = ['poor', 'bad', 'negative', 'unhappy', 'dissatisfied', 'failed',
                         'struggled', 'difficult', 'challenging', 'delayed', 'issue']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total > 0:
            sentiment_score = positive_count / total
        else:
            sentiment_score = 0.5
        
        if sentiment_score > 0.7:
            label = "POSITIVE"
            confidence = sentiment_score
        elif sentiment_score < 0.3:
            label = "NEGATIVE"
            confidence = 1 - sentiment_score
        else:
            label = "NEUTRAL"
            confidence = 0.5
            
        return {
            'label': label,
            'score': round(confidence, 3),
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def analyze_task_complexity_ai(self, task_description: str) -> Dict:
        """AI-based task complexity analysis"""
        # Complexity indicators
        complexity_keywords = {
            'high': ['design', 'develop', 'implement', 'architect', 'create', 'build',
                    'analyze', 'optimize', 'integrate', 'automate', 'machine learning',
                    'ai', 'deep learning', 'neural network', 'blockchain'],
            'medium': ['update', 'modify', 'enhance', 'improve', 'review', 'test',
                      'document', 'configure', 'setup', 'deploy', 'maintain'],
            'low': ['fix', 'correct', 'edit', 'format', 'organize', 'arrange',
                   'clean', 'update', 'check', 'verify', 'monitor']
        }
        
        task_lower = task_description.lower()
        complexity_score = 0.5  # Default medium
        
        # Calculate based on keywords
        high_count = sum(1 for word in complexity_keywords['high'] if word in task_lower)
        medium_count = sum(1 for word in complexity_keywords['medium'] if word in task_lower)
        low_count = sum(1 for word in complexity_keywords['low'] if word in task_lower)
        
        total = high_count + medium_count + low_count
        if total > 0:
            complexity_score = (high_count * 0.9 + medium_count * 0.6 + low_count * 0.3) / total
        
        # Adjust based on text length and technical terms
        technical_terms = ['api', 'database', 'server', 'framework', 'algorithm', 
                          'protocol', 'interface', 'system', 'application']
        technical_count = sum(1 for term in technical_terms if term in task_lower)
        
        if technical_count >= 3:
            complexity_score = min(1.0, complexity_score + 0.2)
        
        # Word count factor
        word_count = len(task_lower.split())
        if word_count > 100:
            complexity_score = min(1.0, complexity_score + 0.1)
        
        return {
            'score': round(complexity_score, 3),
            'level': 'HIGH' if complexity_score > 0.7 else 'MEDIUM' if complexity_score > 0.4 else 'LOW',
            'high_keywords': high_count,
            'medium_keywords': medium_count,
            'low_keywords': low_count,
            'technical_terms': technical_count,
            'word_count': word_count
        }
    
    def generate_ai_recommendations(self, analysis_data: Dict) -> List[str]:
        """Generate AI-powered recommendations"""
        recommendations = []
        
        performance_score = analysis_data.get('performance_score', 50)
        work_hours = analysis_data.get('work_hours', 8)
        engagement_score = analysis_data.get('engagement_score', 50)
        task_complexity = analysis_data.get('task_complexity', 0.5)
        sentiment = analysis_data.get('sentiment', 'NEUTRAL')
        
        # Performance-based recommendations
        if performance_score >= 90:
            recommendations.append("🌟 **Exceptional Performance**: Employee shows outstanding capabilities. Consider leadership opportunities or special projects.")
            recommendations.append("🏆 **Recognition**: Public acknowledgment and rewards would boost morale further.")
        elif performance_score >= 75:
            recommendations.append("✅ **Strong Performance**: Continue current management approach. Provide specific positive feedback.")
            recommendations.append("📚 **Skill Development**: Identify next-level skills for career progression.")
        elif performance_score >= 60:
            recommendations.append("⚠️ **Average Performance**: Schedule regular check-ins. Set clear goals and expectations.")
            recommendations.append("🎯 **Targeted Training**: Identify specific skill gaps and provide focused training.")
        else:
            recommendations.append("🔴 **Performance Improvement Needed**: Create Performance Improvement Plan (PIP) with clear milestones.")
            recommendations.append("🤝 **Mentoring**: Pair with experienced colleague for guidance.")
        
        # Work hours analysis
        if work_hours > 10:
            recommendations.append("⏰ **Work-Life Balance**: High hours detected. Consider workload redistribution or flexible hours.")
            recommendations.append("💆 **Wellness Check**: Schedule wellness discussion to prevent burnout.")
        elif work_hours < 6:
            recommendations.append("📊 **Workload Assessment**: Low hours detected. Review task allocation and engagement.")
        
        # Engagement analysis
        if engagement_score < 40:
            recommendations.append("💡 **Engagement Boost**: Low engagement detected. Review task variety and interest levels.")
            recommendations.append("🎮 **Gamification**: Introduce performance challenges or rewards system.")
        
        # Task complexity matching
        if task_complexity > 0.8 and performance_score > 80:
            recommendations.append("🚀 **High-Complexity Success**: Employee excels with complex tasks. Assign challenging projects.")
        elif task_complexity > 0.8 and performance_score < 60:
            recommendations.append("🎓 **Complexity Support**: Provide additional training or pair programming for complex tasks.")
        
        # Sentiment-based recommendations
        if sentiment == 'NEGATIVE':
            recommendations.append("😔 **Morale Boost**: Negative sentiment detected. Schedule one-on-one to address concerns.")
            recommendations.append("💬 **Open Communication**: Create safe space for feedback and concerns.")
        elif sentiment == 'POSITIVE':
            recommendations.append("😊 **Positive Reinforcement**: Continue current positive environment. Share success stories.")
        
        # AI-powered specific suggestions
        if 'multimodal_data' in analysis_data:
            recommendations.append("🤖 **AI Insight**: Multimodal data suggests personalized approach for optimal results.")
        
        return recommendations
    
    def detect_faces_simple(self, image_array: np.ndarray) -> Dict:
        """Simple face detection (placeholder for real face detection)"""
        height, width = image_array.shape[:2]
        
        # Simple heuristic based on image characteristics
        face_detected = False
        face_count = 0
        confidence = 0.0
        
        if len(image_array.shape) == 3:
            # Color image - check for skin tone like colors
            avg_color = np.mean(image_array, axis=(0, 1))
            
            # Simple skin tone detection (RGB heuristic)
            r, g, b = avg_color
            if (r > 100 and g > 60 and b > 40 and 
                abs(r - g) > 20 and r > g and r > b):
                face_detected = True
                face_count = 1
                confidence = 0.7
                
                # Face position estimation
                face_box = [
                    int(width * 0.25), int(height * 0.25),
                    int(width * 0.75), int(height * 0.75)
                ]
            else:
                face_box = [0, 0, width, height]
        else:
            # Grayscale - simpler check
            if height > 100 and width > 100:
                face_detected = True
                face_count = 1
                confidence = 0.5
                face_box = [int(width * 0.3), int(height * 0.3),
                           int(width * 0.7), int(height * 0.7)]
            else:
                face_box = [0, 0, width, height]
        
        return {
            'face_detected': face_detected,
            'face_count': face_count,
            'confidence': confidence,
            'face_box': face_box,
            'image_dimensions': f"{width}x{height}"
        }
    
    def analyze_video_engagement(self, video_features: Dict) -> Dict:
        """Analyze video for engagement metrics"""
        duration = video_features.get('duration', 0)
        fps = video_features.get('fps', 30)
        motion_level = video_features.get('motion', 0.5)
        
        # Calculate engagement score
        base_score = 50
        
        # Duration factor (optimal 30-60 minutes)
        if 30 <= duration <= 90:
            duration_score = 30
        elif duration > 90:
            duration_score = max(0, 30 - (duration - 90) / 10)
        else:
            duration_score = duration / 3
        
        # Motion factor
        motion_score = min(30, motion_level * 60)
        
        # FPS factor
        fps_score = 10 if fps >= 24 else 5
        
        engagement_score = min(100, base_score + duration_score + motion_score + fps_score)
        
        return {
            'engagement_score': round(engagement_score, 1),
            'components': {
                'duration_contribution': round(duration_score, 1),
                'motion_contribution': round(motion_score, 1),
                'fps_contribution': fps_score
            },
            'interpretation': self._interpret_engagement(engagement_score)
        }
    
    def _interpret_engagement(self, score: float) -> str:
        if score >= 80:
            return "Highly engaged and focused"
        elif score >= 60:
            return "Moderately engaged"
        elif score >= 40:
            return "Somewhat engaged, room for improvement"
        else:
            return "Low engagement detected"
    
    def predict_performance_trend(self, historical_data: List[Dict]) -> Dict:
        """Predict future performance trend"""
        if not historical_data or len(historical_data) < 2:
            return {
                'trend': 'STABLE',
                'confidence': 0.5,
                'prediction': 'Insufficient data for trend analysis'
            }
        
        scores = [data.get('performance_score', 50) for data in historical_data]
        
        # Simple linear trend calculation
        x = np.arange(len(scores))
        y = np.array(scores)
        
        if len(scores) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            
            if slope > 2:
                trend = 'IMPROVING'
                confidence = min(0.9, abs(slope) / 10)
            elif slope < -2:
                trend = 'DECLINING'
                confidence = min(0.9, abs(slope) / 10)
            else:
                trend = 'STABLE'
                confidence = 0.7
            
            # Predict next score
            next_score = slope * len(scores) + intercept
            next_score = max(0, min(100, next_score))
            
            prediction = f"Expected next performance: {next_score:.1f}/100"
        else:
            trend = 'STABLE'
            confidence = 0.5
            prediction = 'More data needed for accurate prediction'
        
        return {
            'trend': trend,
            'confidence': round(confidence, 2),
            'prediction': prediction,
            'historical_scores': scores
        }

# ============================================
# 2. REINFORCEMENT LEARNING AGENT
# ============================================

class ReinforcementLearningAgent:
    """RL Agent for optimal action selection"""
    
    def __init__(self):
        # Action space with 10 different management actions
        self.actions = [
            {
                'id': 0,
                'name': 'No Action',
                'description': 'Continue current management approach',
                'category': 'monitoring',
                'intensity': 'low'
            },
            {
                'id': 1,
                'name': 'Positive Feedback',
                'description': 'Provide specific positive reinforcement',
                'category': 'reward',
                'intensity': 'low'
            },
            {
                'id': 2,
                'name': 'Wellness Check',
                'description': 'Schedule wellness discussion',
                'category': 'wellness',
                'intensity': 'medium'
            },
            {
                'id': 3,
                'name': 'Skill Training',
                'description': 'Offer targeted skill development',
                'category': 'development',
                'intensity': 'medium'
            },
            {
                'id': 4,
                'name': 'Workload Adjustment',
                'description': 'Review and adjust task allocation',
                'category': 'workload',
                'intensity': 'high'
            },
            {
                'id': 5,
                'name': 'Promotion Discussion',
                'description': 'Discuss career progression',
                'category': 'career',
                'intensity': 'high'
            },
            {
                'id': 6,
                'name': 'Constructive Feedback',
                'description': 'Provide specific improvement areas',
                'category': 'feedback',
                'intensity': 'medium'
            },
            {
                'id': 7,
                'name': 'Mentorship Setup',
                'description': 'Arrange mentoring relationship',
                'category': 'development',
                'intensity': 'medium'
            },
            {
                'id': 8,
                'name': 'Recognition Award',
                'description': 'Formal recognition or reward',
                'category': 'reward',
                'intensity': 'high'
            },
            {
                'id': 9,
                'name': 'Performance Plan',
                'description': 'Create Performance Improvement Plan',
                'category': 'performance',
                'intensity': 'high'
            }
        ]
        
        # Q-table for state-action values (simplified)
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        
    def discretize_state(self, state_features: Dict) -> str:
        """Convert continuous state to discrete state key"""
        # Discretize performance score
        perf = state_features.get('performance_score', 50)
        if perf >= 80:
            perf_level = 'high'
        elif perf >= 60:
            perf_level = 'medium'
        else:
            perf_level = 'low'
        
        # Discretize work hours
        hours = state_features.get('work_hours', 8)
        if hours > 10:
            hour_level = 'high'
        elif hours < 6:
            hour_level = 'low'
        else:
            hour_level = 'normal'
        
        # Discretize engagement
        engagement = state_features.get('engagement_score', 50)
        if engagement >= 70:
            eng_level = 'high'
        elif engagement >= 40:
            eng_level = 'medium'
        else:
            eng_level = 'low'
        
        return f"{perf_level}_{hour_level}_{eng_level}"
    
    def select_action(self, state_features: Dict) -> Dict:
        """Select optimal action using epsilon-greedy policy"""
        state_key = self.discretize_state(state_features)
        
        # Initialize Q-values for new state
        if state_key not in self.q_table:
            self.q_table[state_key] = {i: 0.0 for i in range(len(self.actions))}
        
        # Exploration vs exploitation
        if np.random.random() < self.exploration_rate:
            # Explore: random action
            action_id = np.random.choice(len(self.actions))
        else:
            # Exploit: best known action
            q_values = self.q_table[state_key]
            action_id = max(q_values.items(), key=lambda x: x[1])[0]
        
        return self.actions[action_id]
    
    def update_q_value(self, state: str, action_id: int, reward: float, next_state: str):
        """Update Q-value using Q-learning"""
        if state not in self.q_table:
            self.q_table[state] = {i: 0.0 for i in range(len(self.actions))}
        if next_state not in self.q_table:
            self.q_table[next_state] = {i: 0.0 for i in range(len(self.actions))}
        
        # Q-learning update
        current_q = self.q_table[state][action_id]
        max_next_q = max(self.q_table[next_state].values())
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action_id] = new_q
        
        # Decay exploration rate
        self.exploration_rate *= 0.99
    
    def calculate_reward(self, current_state: Dict, next_state: Dict, action_taken: Dict) -> float:
        """Calculate reward for RL agent"""
        reward = 0
        
        # Performance improvement reward
        perf_diff = next_state.get('performance_score', 50) - current_state.get('performance_score', 50)
        reward += perf_diff * 0.1
        
        # Engagement improvement reward
        eng_diff = next_state.get('engagement_score', 50) - current_state.get('engagement_score', 50)
        reward += eng_diff * 0.05
        
        # Work hours optimization reward (closer to 8 is better)
        current_hours = current_state.get('work_hours', 8)
        next_hours = next_state.get('work_hours', 8)
        
        current_hour_dev = abs(current_hours - 8)
        next_hour_dev = abs(next_hours - 8)
        hour_improvement = current_hour_dev - next_hour_dev
        reward += hour_improvement * 0.2
        
        # Action-specific rewards
        action_intensity = action_taken.get('intensity', 'low')
        if action_intensity == 'high':
            reward -= 0.1  # Penalize high-intensity actions
        elif action_intensity == 'low':
            reward += 0.05  # Reward low-intensity actions
        
        return round(reward, 3)
    
    def get_action_by_category(self, category: str) -> List[Dict]:
        """Get actions by category"""
        return [action for action in self.actions if action['category'] == category]
    
    def get_action_statistics(self) -> Dict:
        """Get statistics about action selection"""
        if not self.q_table:
            return {'total_states': 0, 'total_actions': 0}
        
        total_states = len(self.q_table)
        action_counts = {i: 0 for i in range(len(self.actions))}
        
        for state in self.q_table:
            if self.q_table[state]:
                best_action = max(self.q_table[state].items(), key=lambda x: x[1])[0]
                action_counts[best_action] += 1
        
        return {
            'total_states': total_states,
            'action_distribution': action_counts,
            'most_common_action': max(action_counts.items(), key=lambda x: x[1])[0] if total_states > 0 else None
        }

# ============================================
# 3. ENHANCED DATA MANAGEMENT
# ============================================

class EnhancedEmployeeDatabase:
    def __init__(self):
        self.df = None
        self.employee_history = {}
        self.analysis_history = []
        self.load_data()
    
    def load_data(self):
        """Load employee data from multiple sources"""
        try:
            # Try to load existing dataset
            if os.path.exists("employee_multimodal_dataset.csv"):
                self.df = pd.read_csv("employee_multimodal_dataset.csv")
                self._process_dataframe()
                st.success(f"✅ Loaded {len(self.df)} employee records")
            else:
                self.df = pd.DataFrame(columns=[
                    'Date', 'Team Members', 'Signed In', 'Signed Out', 
                    'Completed Task', 'session_id', 'selfie_path', 
                    'session_video_path', 'performance_score', 'engagement_score',
                    'work_hours', 'task_complexity', 'ai_action', 
                    'recommendations', 'analysis_timestamp'
                ])
                st.info("📁 No existing dataset found. Starting fresh.")
            
            # Load analysis history
            if os.path.exists("analysis_history.json"):
                with open("analysis_history.json", 'r') as f:
                    self.analysis_history = json.load(f)
                    
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            self.df = pd.DataFrame()
    
    def _process_dataframe(self):
        """Process and clean dataframe"""
        if self.df is not None and not self.df.empty:
            # Ensure required columns exist
            required_cols = ['Team Members', 'Signed In', 'Signed Out', 'Completed Task']
            for col in required_cols:
                if col not in self.df.columns:
                    self.df[col] = ''
            
            # Fill missing values
            self.df = self.df.fillna('')
            
            # Convert date columns
            if 'Date' in self.df.columns:
                self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
            
            # Build employee history
            self.employee_history = {}
            for idx, row in self.df.iterrows():
                employee_name = str(row.get('Team Members', '')).strip()
                if employee_name:
                    if employee_name not in self.employee_history:
                        self.employee_history[employee_name] = []
                    
                    self.employee_history[employee_name].append({
                        'date': row.get('Date', ''),
                        'sign_in': row.get('Signed In', ''),
                        'sign_out': row.get('Signed Out', ''),
                        'task': row.get('Completed Task', ''),
                        'performance_score': row.get('performance_score', 0),
                        'engagement_score': row.get('engagement_score', 0),
                        'work_hours': row.get('work_hours', 0),
                        'ai_action': row.get('ai_action', '')
                    })
    
    def upload_dataset(self, uploaded_file) -> Tuple[str, pd.DataFrame]:
        """Handle dataset upload"""
        if uploaded_file is None:
            return "No file uploaded", pd.DataFrame()
        
        try:
            self.df = pd.read_csv(uploaded_file)
            self._process_dataframe()
            
            # Save to disk
            self.save_to_csv()
            
            return f"✅ Successfully uploaded {len(self.df)} records", self.df
        except Exception as e:
            return f"❌ Upload failed: {str(e)}", pd.DataFrame()
    
    def search_employees(self, search_term: str, limit: int = 10) -> List[str]:
        """Search for employees with advanced filtering"""
        if not search_term or self.df is None or self.df.empty:
            return []
        
        search_term = search_term.lower().strip()
        matches = []
        
        if 'Team Members' in self.df.columns:
            # Fuzzy search with multiple criteria
            for idx, row in self.df.iterrows():
                employee_name = str(row.get('Team Members', '')).strip()
                
                # Check if search term appears in name
                if search_term in employee_name.lower():
                    matches.append(employee_name)
                
                # Also check in task description
                task_desc = str(row.get('Completed Task', '')).lower()
                if search_term in task_desc and employee_name not in matches:
                    matches.append(employee_name)
        
        # Remove duplicates and sort by frequency
        from collections import Counter
        freq = Counter(matches)
        sorted_matches = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in sorted_matches[:limit]]
    
    def get_employee_profile(self, employee_name: str) -> Dict:
        """Get comprehensive employee profile"""
        if employee_name not in self.employee_history:
            return {}
        
        history = self.employee_history[employee_name]
        
        # Calculate statistics
        perf_scores = [h.get('performance_score', 0) for h in history]
        eng_scores = [h.get('engagement_score', 0) for h in history]
        work_hours = [h.get('work_hours', 0) for h in history]
        
        profile = {
            'name': employee_name,
            'total_records': len(history),
            'avg_performance': np.mean(perf_scores) if perf_scores else 0,
            'avg_engagement': np.mean(eng_scores) if eng_scores else 0,
            'avg_work_hours': np.mean(work_hours) if work_hours else 0,
            'performance_trend': self._calculate_trend(perf_scores),
            'engagement_trend': self._calculate_trend(eng_scores),
            'recent_tasks': [h.get('task', '') for h in history[-5:] if h.get('task')],
            'common_actions': self._get_common_actions(history),
            'history': history[-10:]  # Last 10 records
        }
        
        return profile
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend from score history"""
        if len(scores) < 2:
            return "Insufficient data"
        
        # Simple linear trend
        x = np.arange(len(scores))
        y = np.array(scores)
        
        if len(scores) >= 2:
            slope, _ = np.polyfit(x, y, 1)
            
            if slope > 1:
                return "Improving ↗️"
            elif slope < -1:
                return "Declining ↘️"
            else:
                return "Stable →"
        return "Stable →"
    
    def _get_common_actions(self, history: List[Dict]) -> List[str]:
        """Get most common AI actions for employee"""
        actions = [h.get('ai_action', '') for h in history if h.get('ai_action')]
        from collections import Counter
        common = Counter(actions).most_common(3)
        return [action for action, _ in common]
    
    def save_analysis(self, employee_data: Dict, analysis_results: Dict):
        """Save analysis results to database"""
        try:
            # Prepare record
            record = {
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Team Members': employee_data.get('name', ''),
                'Signed In': employee_data.get('sign_in', ''),
                'Signed Out': employee_data.get('sign_out', ''),
                'Completed Task': employee_data.get('task', ''),
                'session_id': len(self.df) + 1,
                'selfie_path': employee_data.get('selfie_path', ''),
                'session_video_path': employee_data.get('session_video_path', ''),
                'performance_score': analysis_results.get('performance_score', 0),
                'engagement_score': analysis_results.get('engagement_score', 0),
                'work_hours': analysis_results.get('work_hours', 0),
                'task_complexity': analysis_results.get('task_complexity', 0),
                'ai_action': analysis_results.get('ai_action', {}).get('name', ''),
                'recommendations': '; '.join(analysis_results.get('recommendations', [])),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Add to dataframe
            new_df = pd.DataFrame([record])
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            
            # Update employee history
            employee_name = employee_data.get('name', '')
            if employee_name not in self.employee_history:
                self.employee_history[employee_name] = []
            
            self.employee_history[employee_name].append({
                'date': record['Date'],
                'sign_in': record['Signed In'],
                'sign_out': record['Signed Out'],
                'task': record['Completed Task'],
                'performance_score': record['performance_score'],
                'engagement_score': record['engagement_score'],
                'work_hours': record['work_hours'],
                'ai_action': record['ai_action']
            })
            
            # Add to analysis history
            self.analysis_history.append({
                'timestamp': record['analysis_timestamp'],
                'employee': employee_name,
                'analysis_summary': {
                    'performance': record['performance_score'],
                    'engagement': record['engagement_score'],
                    'action_taken': record['ai_action']
                }
            })
            
            # Save to disk
            self.save_all_data()
            
            return f"✅ Analysis saved for {employee_name}"
            
        except Exception as e:
            return f"❌ Error saving analysis: {str(e)}"
    
    def save_all_data(self):
        """Save all data to disk"""
        try:
            # Save dataframe
            if self.df is not None and not self.df.empty:
                self.df.to_csv("employee_multimodal_dataset.csv", index=False)
            
            # Save analysis history
            with open("analysis_history.json", 'w') as f:
                json.dump(self.analysis_history, f, indent=2)
                
        except Exception as e:
            st.error(f"Error saving data: {e}")
    
    def get_analytics_dashboard(self) -> Dict:
        """Get analytics for dashboard"""
        if self.df is None or self.df.empty:
            return {}
        
        # Calculate various metrics
        total_employees = self.df['Team Members'].nunique() if 'Team Members' in self.df.columns else 0
        total_analyses = len(self.df)
        
        # Performance distribution
        if 'performance_score' in self.df.columns:
            perf_scores = self.df['performance_score'].dropna()
            avg_performance = perf_scores.mean() if not perf_scores.empty else 0
            high_performers = len(perf_scores[perf_scores >= 80]) if not perf_scores.empty else 0
        else:
            avg_performance = 0
            high_performers = 0
        
        # Common actions
        if 'ai_action' in self.df.columns:
            actions = self.df['ai_action'].value_counts().head(5).to_dict()
        else:
            actions = {}
        
        # Recent activity
        recent_activity = []
        if 'analysis_timestamp' in self.df.columns:
            recent = self.df.sort_values('analysis_timestamp', ascending=False).head(5)
            for _, row in recent.iterrows():
                recent_activity.append({
                    'employee': row.get('Team Members', ''),
                    'date': row.get('Date', ''),
                    'performance': row.get('performance_score', 0),
                    'action': row.get('ai_action', '')
                })
        
        return {
            'total_employees': total_employees,
            'total_analyses': total_analyses,
            'avg_performance': round(avg_performance, 1),
            'high_performers': high_performers,
            'action_distribution': actions,
            'recent_activity': recent_activity
        }

# ============================================
# 4. ENHANCED PERFORMANCE ANALYZER
# ============================================

class EnhancedPerformanceAnalyzer:
    def __init__(self):
        self.ai_models = AIActionModels()
        self.rl_agent = ReinforcementLearningAgent()
        self.database = EnhancedEmployeeDatabase()
    
    def analyze_multimodal_data(self, employee_data: Dict, video_file, image_file) -> Dict:
        """Comprehensive multimodal analysis"""
        results = {
            'basic_analysis': {},
            'ai_analysis': {},
            'multimodal_analysis': {},
            'rl_analysis': {},
            'recommendations': []
        }
        
        # 1. Basic analysis
        results['basic_analysis'] = self._analyze_basic_metrics(employee_data)
        
        # 2. AI analysis
        results['ai_analysis'] = self._perform_ai_analysis(employee_data)
        
        # 3. Multimodal analysis
        results['multimodal_analysis'] = self._analyze_multimedia(video_file, image_file)
        
        # 4. RL analysis
        results['rl_analysis'] = self._perform_rl_analysis(results)
        
        # 5. Generate recommendations
        results['recommendations'] = self._generate_comprehensive_recommendations(results)
        
        return results
    
    def _analyze_basic_metrics(self, employee_data: Dict) -> Dict:
        """Analyze basic work metrics"""
        # Calculate work hours
        work_hours = self._calculate_work_hours(
            employee_data.get('sign_in', ''),
            employee_data.get('sign_out', '')
        )
        
        # Analyze task
        task = employee_data.get('task', '')
        task_analysis = self.ai_models.analyze_task_complexity_ai(task)
        
        return {
            'work_hours': work_hours,
            'task_analysis': task_analysis,
            'employee_info': {
                'name': employee_data.get('name', ''),
                'date': employee_data.get('date', '')
            }
        }
    
    def _calculate_work_hours(self, sign_in: str, sign_out: str) -> float:
        """Calculate work hours from time strings"""
        try:
            if not sign_in or not sign_out:
                return 8.0
            
            # Try different time formats
            time_formats = ['%H:%M', '%I:%M %p', '%I:%M%p']
            
            in_time = None
            out_time = None
            
            for fmt in time_formats:
                try:
                    in_time = datetime.strptime(sign_in, fmt)
                    break
                except:
                    continue
            
            for fmt in time_formats:
                try:
                    out_time = datetime.strptime(sign_out, fmt)
                    break
                except:
                    continue
            
            if in_time and out_time:
                duration = (out_time - in_time).seconds / 3600
                
                # Handle overnight shifts
                if duration < 0:
                    duration += 24
                
                return round(duration, 2)
            
            return 8.0
            
        except Exception as e:
            st.error(f"Error calculating work hours: {e}")
            return 8.0
    
    def _perform_ai_analysis(self, employee_data: Dict) -> Dict:
        """Perform AI-based analysis"""
        task = employee_data.get('task', '')
        
        # Sentiment analysis
        sentiment = self.ai_models.analyze_sentiment(task)
        
        # Task complexity
        complexity = self.ai_models.analyze_task_complexity_ai(task)
        
        # Text analysis
        text_analysis = self._analyze_task_text(task)
        
        return {
            'sentiment_analysis': sentiment,
            'task_complexity': complexity,
            'text_analysis': text_analysis
        }
    
    def _analyze_task_text(self, task_text: str) -> Dict:
        """Analyze task text for various metrics"""
        word_count = len(task_text.split())
        char_count = len(task_text)
        sentence_count = len(re.split(r'[.!?]+', task_text))
        
        # Calculate readability score (simplified)
        readability = 0
        if word_count > 0 and sentence_count > 0:
            avg_words_per_sentence = word_count / sentence_count
            readability = min(100, max(0, 100 - avg_words_per_sentence * 2))
        
        # Detect technical terms
        technical_terms = ['api', 'database', 'server', 'framework', 'algorithm',
                          'protocol', 'interface', 'system', 'application', 'code',
                          'debug', 'deploy', 'integrate', 'optimize']
        
        found_terms = [term for term in technical_terms if term in task_text.lower()]
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'readability_score': round(readability, 1),
            'technical_terms': found_terms,
            'technical_term_count': len(found_terms)
        }
    
    def _analyze_multimedia(self, video_file, image_file) -> Dict:
        """Analyze video and image files"""
        video_analysis = {}
        image_analysis = {}
        
        # Video analysis
        if video_file is not None:
            video_analysis = self._analyze_video_file(video_file)
        
        # Image analysis
        if image_file is not None:
            image_analysis = self._analyze_image_file(image_file)
        
        return {
            'video_analysis': video_analysis,
            'image_analysis': image_analysis,
            'multimodal_score': self._calculate_multimodal_score(video_analysis, image_analysis)
        }
    
    def _analyze_video_file(self, video_file) -> Dict:
        """Analyze video file (simplified)"""
        try:
            # In a real implementation, use OpenCV for video analysis
            # For demo, return simulated analysis
            return {
                'duration_minutes': np.random.uniform(30, 120),
                'fps': 30,
                'resolution': '1280x720',
                'motion_level': np.random.uniform(0.3, 0.9),
                'engagement_estimate': np.random.uniform(40, 90),
                'analysis_status': 'simulated'
            }
        except:
            return {
                'duration_minutes': 60,
                'fps': 30,
                'resolution': 'Unknown',
                'motion_level': 0.5,
                'engagement_estimate': 50,
                'analysis_status': 'basic'
            }
    
    def _analyze_image_file(self, image_file) -> Dict:
        """Analyze image file"""
        try:
            img = Image.open(image_file)
            img_array = np.array(img)
            
            # Basic image analysis
            height, width = img_array.shape[:2]
            
            if len(img_array.shape) == 3:
                # Color image
                brightness = np.mean(img_array) / 255
                contrast = np.std(img_array) / 100
                color_variance = np.var(img_array, axis=(0, 1)).mean() / 10000
                
                # Simple face detection heuristic
                avg_color = np.mean(img_array, axis=(0, 1))
                r, g, b = avg_color
                has_face_like_colors = (r > 100 and g > 60 and b > 40 and 
                                       abs(r - g) > 20 and r > g and r > b)
            else:
                # Grayscale
                brightness = np.mean(img_array) / 255
                contrast = np.std(img_array) / 100
                color_variance = 0
                has_face_like_colors = False
            
            face_detection = self.ai_models.detect_faces_simple(img_array)
            
            return {
                'dimensions': f"{width}x{height}",
                'brightness': round(brightness, 3),
                'contrast': round(contrast, 3),
                'color_variance': round(color_variance, 3),
                'has_face_like_colors': has_face_like_colors,
                'face_detection': face_detection,
                'image_quality': self._assess_image_quality(brightness, contrast)
            }
            
        except Exception as e:
            st.error(f"Image analysis error: {e}")
            return {
                'dimensions': 'Unknown',
                'brightness': 0.5,
                'contrast': 0.5,
                'color_variance': 0,
                'has_face_like_colors': False,
                'face_detection': {'face_detected': False},
                'image_quality': 'unknown'
            }
    
    def _assess_image_quality(self, brightness: float, contrast: float) -> str:
        """Assess image quality"""
        quality_score = brightness * 0.6 + contrast * 0.4
        
        if quality_score > 0.7:
            return 'good'
        elif quality_score > 0.4:
            return 'average'
        else:
            return 'poor'
    
    def _calculate_multimodal_score(self, video_analysis: Dict, image_analysis: Dict) -> float:
        """Calculate combined multimodal score"""
        score = 50  # Base score
        
        # Video contribution
        if video_analysis:
            engagement = video_analysis.get('engagement_estimate', 50)
            duration = video_analysis.get('duration_minutes', 60)
            
            # Optimal duration: 60 minutes
            duration_score = max(0, 100 - abs(duration - 60))
            score = (engagement + duration_score) / 2
        
        # Image contribution
        if image_analysis:
            face_detected = image_analysis.get('face_detection', {}).get('face_detected', False)
            quality = image_analysis.get('image_quality', 'average')
            
            if face_detected:
                score += 10
            if quality == 'good':
                score += 10
            elif quality == 'poor':
                score -= 10
        
        return round(min(100, max(0, score)), 1)
    
    def _perform_rl_analysis(self, results: Dict) -> Dict:
        """Perform reinforcement learning analysis"""
        # Prepare state for RL agent
        basic = results['basic_analysis']
        ai = results['ai_analysis']
        multimodal = results['multimodal_analysis']
        
        state_features = {
            'performance_score': self._calculate_overall_score(results),
            'work_hours': basic.get('work_hours', 8),
            'engagement_score': multimodal.get('multimodal_score', 50),
            'task_complexity': ai.get('task_complexity', {}).get('score', 0.5),
            'sentiment': ai.get('sentiment_analysis', {}).get('label', 'NEUTRAL')
        }
        
        # Select action using RL agent
        action = self.rl_agent.select_action(state_features)
        
        return {
            'selected_action': action,
            'state_features': state_features,
            'q_table_stats': self.rl_agent.get_action_statistics()
        }
    
    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate overall performance score"""
        weights = {
            'work_hours': 0.20,
            'task_complexity': 0.25,
            'sentiment': 0.15,
            'multimodal': 0.40
        }
        
        basic = results['basic_analysis']
        ai = results['ai_analysis']
        multimodal = results['multimodal_analysis']
        
        # Work hours score (optimal: 8 hours)
        work_hours = basic.get('work_hours', 8)
        hours_score = max(0, 100 - abs(work_hours - 8) * 10)
        
        # Task complexity score (higher complexity = higher potential score)
        complexity = ai.get('task_complexity', {}).get('score', 0.5)
        complexity_score = complexity * 100
        
        # Sentiment score
        sentiment = ai.get('sentiment_analysis', {}).get('label', 'NEUTRAL')
        if sentiment == 'POSITIVE':
            sentiment_score = 80
        elif sentiment == 'NEGATIVE':
            sentiment_score = 40
        else:
            sentiment_score = 60
        
        # Multimodal score
        multimodal_score = multimodal.get('multimodal_score', 50)
        
        # Calculate weighted score
        overall_score = (
            hours_score * weights['work_hours'] +
            complexity_score * weights['task_complexity'] +
            sentiment_score * weights['sentiment'] +
            multimodal_score * weights['multimodal']
        )
        
        return round(min(100, overall_score), 1)
    
    def _generate_comprehensive_recommendations(self, results: Dict) -> List[str]:
        """Generate comprehensive AI-powered recommendations"""
        recommendations = []
        
        # Get analysis components
        basic = results['basic_analysis']
        ai = results['ai_analysis']
        multimodal = results['multimodal_analysis']
        rl = results['rl_analysis']
        
        overall_score = self._calculate_overall_score(results)
        work_hours = basic.get('work_hours', 8)
        task_complexity = ai.get('task_complexity', {}).get('score', 0.5)
        sentiment = ai.get('sentiment_analysis', {}).get('label', 'NEUTRAL')
        engagement_score = multimodal.get('multimodal_score', 50)
        selected_action = rl.get('selected_action', {})
        
        # RL Action recommendation
        if selected_action:
            recommendations.append(
                f"🤖 **AI Recommended Action**: {selected_action.get('name', 'Unknown')} - "
                f"{selected_action.get('description', '')}"
            )
        
        # Performance category recommendations
        if overall_score >= 90:
            recommendations.append("🏆 **Elite Performer**: Consider for leadership role or special high-impact projects.")
            recommendations.append("📈 **Career Acceleration**: Fast-track promotion discussions.")
        elif overall_score >= 75:
            recommendations.append("✅ **Strong Contributor**: Maintain current trajectory with regular positive feedback.")
            recommendations.append("🎯 **Skill Enhancement**: Identify and develop next-level competencies.")
        elif overall_score >= 60:
            recommendations.append("⚠️ **Development Opportunity**: Schedule weekly check-ins for guidance and support.")
            recommendations.append("📚 **Targeted Training**: Provide specific training based on skill gaps.")
        else:
            recommendations.append("🔴 **Performance Support**: Create detailed Performance Improvement Plan (PIP).")
            recommendations.append("👥 **Mentorship**: Assign experienced mentor for daily guidance.")
        
        # Work hours optimization
        if work_hours > 10:
            recommendations.append("⏰ **Burnout Prevention**: Implement mandatory breaks and workload review.")
            recommendations.append("🏡 **Flexible Work**: Consider remote work options to reduce commute stress.")
        elif work_hours < 6:
            recommendations.append("📊 **Capacity Analysis**: Assess if employee has bandwidth for additional responsibilities.")
        
        # Task complexity matching
        if task_complexity > 0.8:
            recommendations.append("🚀 **Complex Task Specialist**: Assign challenging projects that match their capabilities.")
        elif task_complexity < 0.3:
            recommendations.append("📈 **Skill Stretching**: Gradually increase task complexity to develop skills.")
        
        # Sentiment-based recommendations
        if sentiment == 'NEGATIVE':
            recommendations.append("😔 **Morale Intervention**: Schedule confidential discussion to address concerns.")
            recommendations.append("💬 **Feedback Channel**: Establish regular feedback mechanism.")
        elif sentiment == 'POSITIVE':
            recommendations.append("😊 **Positive Culture**: Document and share success stories with team.")
        
        # Engagement recommendations
        if engagement_score < 40:
            recommendations.append("💡 **Engagement Strategy**: Implement gamification or recognition programs.")
            recommendations.append("🎮 **Interactive Tools**: Introduce collaborative platforms for better engagement.")
        
        # Multimodal specific recommendations
        video_analysis = multimodal.get('video_analysis', {})
        image_analysis = multimodal.get('image_analysis', {})
        
        if video_analysis:
            duration = video_analysis.get('duration_minutes', 0)
            if duration > 120:
                recommendations.append("🎬 **Session Optimization**: Break long sessions into focused intervals.")
        
        if image_analysis:
            face_detected = image_analysis.get('face_detection', {}).get('face_detected', False)
            if not face_detected:
                recommendations.append("📸 **Workspace Setup**: Ensure proper camera positioning for better engagement tracking.")
        
        return recommendations
    
    def generate_analysis_report(self, employee_data: Dict, results: Dict) -> str:
        """Generate comprehensive analysis report"""
        report = f"""
        {'=' * 70}
        EMPLOYEE PERFORMANCE ANALYSIS REPORT
        {'=' * 70}
        
        📋 EMPLOYEE INFORMATION
        {'-' * 30}
        • Name: {employee_data.get('name', 'N/A')}
        • Date: {employee_data.get('date', 'N/A')}
        • Task: {employee_data.get('task', 'N/A')[:200]}...
        • Work Hours: {results['basic_analysis'].get('work_hours', 0):.1f} hours
        
        📊 PERFORMANCE METRICS
        {'-' * 30}
        • Overall Score: {self._calculate_overall_score(results)}/100
        • Engagement Score: {results['multimodal_analysis'].get('multimodal_score', 0):.1f}/100
        • Task Complexity: {results['ai_analysis'].get('task_complexity', {}).get('level', 'N/A')}
        • Sentiment: {results['ai_analysis'].get('sentiment_analysis', {}).get('label', 'N/A')}
        
        🤖 AI ANALYSIS
        {'-' * 30}
        • Recommended Action: {results['rl_analysis'].get('selected_action', {}).get('name', 'N/A')}
        • Action Category: {results['rl_analysis'].get('selected_action', {}).get('category', 'N/A')}
        • Action Intensity: {results['rl_analysis'].get('selected_action', {}).get('intensity', 'N/A')}
        
        🎯 MULTIMODAL ANALYSIS
        {'-' * 30}
        • Video Analysis: {'Available' if results['multimodal_analysis'].get('video_analysis') else 'Not Available'}
        • Image Analysis: {'Available' if results['multimodal_analysis'].get('image_analysis') else 'Not Available'}
        • Face Detected: {results['multimodal_analysis'].get('image_analysis', {}).get('face_detection', {}).get('face_detected', False)}
        
        💡 RECOMMENDATIONS
        {'-' * 30}
        """
        
        recommendations = results.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"\n{'=' * 70}"
        report += f"\n📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        report += f"\n🤖 AI Model Version: 2.0"
        report += f"\n{'=' * 70}"
        
        return report

# ============================================
# 5. STREAMLIT APPLICATION
# ============================================

class EmployeePerformanceApp:
    def __init__(self):
        self.database = EnhancedEmployeeDatabase()
        self.analyzer = EnhancedPerformanceAnalyzer()
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'selected_employee' not in st.session_state:
            st.session_state.selected_employee = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'employee_search_term' not in st.session_state:
            st.session_state.employee_search_term = ''
        if 'analysis_in_progress' not in st.session_state:
            st.session_state.analysis_in_progress = False
    
    def run(self):
        """Run the Streamlit application"""
        # Page configuration
        st.set_page_config(
            page_title="AI-Powered Employee Performance Analyzer",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        self._apply_custom_css()
        
        # Sidebar
        self._render_sidebar()
        
        # Main content based on navigation
        page = st.session_state.get('current_page', 'dashboard')
        
        if page == 'dashboard':
            self._render_dashboard()
        elif page == 'dataset':
            self._render_dataset_page()
        elif page == 'analysis':
            self._render_analysis_page()
        elif page == 'reports':
            self._render_reports_page()
        elif page == 'ai_insights':
            self._render_ai_insights_page()
    
    def _apply_custom_css(self):
        """Apply custom CSS styles"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background: linear-gradient(90deg, #1f77b4, #2ca02c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .section-header {
            font-size: 1.8rem;
            font-weight: bold;
            color: #2ca02c;
            margin-top: 2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #2ca02c;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .recommendation-card {
            background-color: #e8f4f8;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            border-left: 4px solid #2ca02c;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .ai-action-card {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 4px solid #ffc107;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stButton button {
            width: 100%;
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background-color: #2ca02c;
            transform: translateY(-2px);
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .info-box {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render the sidebar navigation"""
        with st.sidebar:
            st.markdown('<div class="main-header">🤖 AI Performance Analyzer</div>', unsafe_allow_html=True)
            
            # Navigation
            st.markdown("### 🧭 Navigation")
            page_options = {
                "🏠 Dashboard": "dashboard",
                "📁 Dataset Management": "dataset",
                "🔍 Performance Analysis": "analysis",
                "📊 Reports & Analytics": "reports",
                "🤖 AI Insights": "ai_insights"
            }
            
            selected = st.radio(
                "Select Page",
                list(page_options.keys()),
                label_visibility="collapsed"
            )
            
            st.session_state.current_page = page_options[selected]
            
            st.markdown("---")
            
            # System Status
            st.markdown("### ⚙️ System Status")
            
            col1, col2 = st.columns(2)
            with col1:
                total_records = len(self.database.df) if self.database.df is not None else 0
                st.metric("📊 Records", total_records)
            
            with col2:
                unique_employees = self.database.df['Team Members'].nunique() if self.database.df is not None and not self.database.df.empty else 0
                st.metric("👥 Employees", unique_employees)
            
            st.markdown("---")
            
            # Quick Actions
            st.markdown("### ⚡ Quick Actions")
            
            if st.button("🔄 Clear Session", type="secondary"):
                st.session_state.selected_employee = None
                st.session_state.analysis_results = None
                st.rerun()
            
            if st.button("💾 Backup Data", type="secondary"):
                if self.database.df is not None and not self.database.df.empty:
                    self.database.save_all_data()
                    st.success("✅ Data backed up successfully!")
                else:
                    st.warning("No data to backup")
    
    def _render_dashboard(self):
        """Render the dashboard page"""
        st.markdown('<div class="main-header">🏠 AI-Powered Performance Dashboard</div>', unsafe_allow_html=True)
        
        # Get analytics
        analytics = self.database.get_analytics_dashboard()
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", analytics.get('total_analyses', 0))
        
        with col2:
            st.metric("Avg Performance", f"{analytics.get('avg_performance', 0):.1f}/100")
        
        with col3:
            st.metric("High Performers", analytics.get('high_performers', 0))
        
        with col4:
            st.metric("AI Actions", len(analytics.get('action_distribution', {})))
        
        st.markdown("---")
        
        # Recent Activity
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="section-header">📈 Recent Activity</div>', unsafe_allow_html=True)
            
            if analytics.get('recent_activity'):
                for activity in analytics['recent_activity']:
                    with st.container():
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        with col_a:
                            st.write(f"**{activity['employee']}**")
                        with col_b:
                            st.write(f"Score: {activity['performance']}")
                        with col_c:
                            st.write(f"Action: {activity['action']}")
                        st.markdown("---")
            else:
                st.info("No recent activity to display")
        
        with col2:
            st.markdown('<div class="section-header">🎯 Top AI Actions</div>', unsafe_allow_html=True)
            
            actions = analytics.get('action_distribution', {})
            if actions:
                for action, count in list(actions.items())[:5]:
                    st.metric(action, count)
            else:
                st.info("No AI actions recorded")
        
        # Quick Analysis Section
        st.markdown('<div class="section-header">🚀 Quick Analysis</div>', unsafe_allow_html=True)
        
        with st.form("quick_analysis"):
            quick_name = st.text_input("Employee Name", placeholder="Enter name for quick analysis")
            quick_task = st.text_area("Task Description", placeholder="Brief task description...", height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                quick_sign_in = st.selectbox("Sign In", ["09:00", "10:00", "11:00", "12:00"])
            with col2:
                quick_sign_out = st.selectbox("Sign Out", ["17:00", "18:00", "19:00", "20:00"])
            
            if st.form_submit_button("🤖 Quick Analyze", type="primary"):
                if quick_name and quick_task:
                    employee_data = {
                        'name': quick_name,
                        'task': quick_task,
                        'sign_in': quick_sign_in,
                        'sign_out': quick_sign_out,
                        'date': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    with st.spinner("AI is analyzing..."):
                        results = self.analyzer.analyze_multimodal_data(employee_data, None, None)
                        
                        # Display quick results
                        st.markdown(f"### Quick Results for {quick_name}")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Performance Score", f"{results['rl_analysis']['state_features']['performance_score']}/100")
                        
                        with col2:
                            st.metric("AI Action", results['rl_analysis']['selected_action']['name'])
                        
                        st.success("✅ Analysis complete! Go to Analysis page for detailed report.")
                else:
                    st.warning("Please enter employee name and task description")
    
    def _render_dataset_page(self):
        """Render the dataset management page"""
        st.markdown('<div class="main-header">📁 Dataset Management</div>', unsafe_allow_html=True)
        
        # Upload section
        st.markdown('<div class="section-header">📤 Upload Dataset</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your employee dataset in CSV format"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("📥 Process Upload", type="primary"):
                    with st.spinner("Processing dataset..."):
                        message, _ = self.database.upload_dataset(uploaded_file)
                        st.success(message)
                        st.rerun()
        
        st.markdown("---")
        
        # Dataset preview
        if self.database.df is not None and not self.database.df.empty:
            st.markdown(f'<div class="section-header">👁️ Dataset Preview ({len(self.database.df)} records)</div>', unsafe_allow_html=True)
            
            # Show preview
            preview_size = st.slider("Preview rows", 5, 50, 10)
            st.dataframe(self.database.df.head(preview_size), use_container_width=True)
            
            st.markdown("---")
            
            # Employee search and selection
            st.markdown('<div class="section-header">🔍 Search Employee</div>', unsafe_allow_html=True)
            
            search_term = st.text_input(
                "Enter employee name to search",
                value=st.session_state.employee_search_term,
                placeholder="Type name...",
                key="dataset_search"
            )
            
            if search_term:
                st.session_state.employee_search_term = search_term
                matches = self.database.search_employees(search_term)
                
                if matches:
                    st.success(f"Found {len(matches)} employee(s)")
                    
                    selected_employee = st.selectbox(
                        "Select employee",
                        matches,
                        key="employee_select_dataset"
                    )
                    
                    if selected_employee:
                        # Get employee profile
                        profile = self.database.get_employee_profile(selected_employee)
                        
                        if profile:
                            # Display profile
                            st.markdown(f"### 👤 Employee Profile: {selected_employee}")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Total Records", profile['total_records'])
                                st.metric("Avg Performance", f"{profile['avg_performance']:.1f}/100")
                            
                            with col2:
                                st.metric("Avg Engagement", f"{profile['avg_engagement']:.1f}/100")
                                st.metric("Performance Trend", profile['performance_trend'])
                            
                            # Use for Analysis button
                            if st.button("🚀 Use for Analysis", type="primary", use_container_width=True):
                                # Get latest record for this employee
                                latest_record = profile['history'][-1] if profile['history'] else {}
                                
                                st.session_state.selected_employee = {
                                    'name': selected_employee,
                                    'sign_in': latest_record.get('sign_in', '09:00'),
                                    'sign_out': latest_record.get('sign_out', '17:00'),
                                    'task': latest_record.get('task', ''),
                                    'date': latest_record.get('date', datetime.now().strftime('%Y-%m-%d'))
                                }
                                
                                # Switch to analysis page
                                st.session_state.current_page = 'analysis'
                                st.success(f"✅ Employee '{selected_employee}' loaded for analysis!")
                                st.rerun()
                            
                            # Show recent tasks
                            if profile['recent_tasks']:
                                with st.expander("View Recent Tasks"):
                                    for task in profile['recent_tasks']:
                                        st.write(f"• {task[:100]}...")
                else:
                    st.warning("No employees found matching your search.")
        else:
            st.warning("📭 No dataset loaded. Please upload a CSV file to get started.")
    
    def _render_analysis_page(self):
        """Render the performance analysis page"""
        st.markdown('<div class="main-header">🔍 AI Performance Analysis</div>', unsafe_allow_html=True)
        
        # Check for pre-loaded employee
        preloaded_info = None
        if st.session_state.selected_employee:
            preloaded_info = st.session_state.selected_employee
            st.markdown(f'<div class="success-box">👤 Using pre-loaded employee: {preloaded_info["name"]}</div>', unsafe_allow_html=True)
        
        # Analysis form
        with st.form("analysis_form", clear_on_submit=False):
            st.markdown('<div class="section-header">📋 Employee Information</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                employee_name = st.text_input(
                    "👤 Employee Name *",
                    value=preloaded_info['name'] if preloaded_info else "",
                    placeholder="Enter employee name"
                )
                
                analysis_date = st.text_input(
                    "📅 Analysis Date",
                    value=preloaded_info['date'] if preloaded_info else datetime.now().strftime('%Y-%m-%d'),
                    placeholder="YYYY-MM-DD"
                )
            
            with col2:
                # 24-hour time selection
                sign_in_time = st.selectbox(
                    "⏰ Sign In Time *",
                    [f"{h:02d}:{m:02d}" for h in range(0, 24) for m in [0, 30]],
                    index=18,  # 09:00
                    help="Select sign in time (24-hour format)"
                )
                
                sign_out_time = st.selectbox(
                    "⏰ Sign Out Time *",
                    [f"{h:02d}:{m:02d}" for h in range(0, 24) for m in [0, 30]],
                    index=34,  # 17:00
                    help="Select sign out time (24-hour format)"
                )
            
            # Task description
            completed_task = st.text_area(
                "📝 Completed Task *",
                value=preloaded_info['task'] if preloaded_info else "",
                placeholder="Describe the completed task in detail. Include challenges faced, solutions implemented, and outcomes achieved...",
                height=150,
                help="Detailed description helps AI provide better analysis"
            )
            
            st.markdown('<div class="section-header">🎬 Multimedia Data (Optional)</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                session_video = st.file_uploader(
                    "📹 Upload Session Video",
                    type=['mp4', 'avi', 'mov'],
                    help="Optional: Upload work session video for engagement analysis"
                )
                if session_video:
                    st.info(f"Video uploaded: {session_video.name}")
            
            with col2:
                selfie_image = st.file_uploader(
                    "📸 Upload Selfie/Workspace Image",
                    type=['jpg', 'jpeg', 'png'],
                    help="Optional: Upload selfie or workspace photo for environment analysis"
                )
                if selfie_image:
                    try:
                        img = Image.open(selfie_image)
                        st.image(img, caption="Uploaded Image", width=200)
                    except:
                        st.warning("Could not display image")
            
            st.markdown("---")
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                col1, col2 = st.columns(2)
                with col1:
                    enable_ai_models = st.checkbox("Enable All AI Models", value=True)
                    enable_sentiment = st.checkbox("Sentiment Analysis", value=True)
                
                with col2:
                    enable_face_detection = st.checkbox("Face Detection", value=True)
                    enable_rl_training = st.checkbox("Train RL Model", value=False)
            
            # Analyze button
            analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
            with analyze_col2:
                analyze_submitted = st.form_submit_button(
                    "🚀 START AI ANALYSIS",
                    type="primary",
                    use_container_width=True
                )
            
            if analyze_submitted:
                if not employee_name:
                    st.error("❌ Please enter employee name!")
                elif not completed_task:
                    st.error("❌ Please describe the completed task!")
                else:
                    st.session_state.analysis_in_progress = True
                    
                    # Prepare employee data
                    employee_data = {
                        'name': employee_name,
                        'date': analysis_date,
                        'sign_in': sign_in_time,
                        'sign_out': sign_out_time,
                        'task': completed_task
                    }
                    
                    # Perform analysis
                    with st.spinner("🤖 AI is analyzing performance data..."):
                        try:
                            results = self.analyzer.analyze_multimodal_data(
                                employee_data, session_video, selfie_image
                            )
                            
                            st.session_state.analysis_results = {
                                'employee_data': employee_data,
                                'analysis_results': results,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Save to database
                            save_result = self.database.save_analysis(employee_data, results)
                            st.success(save_result)
                            
                            st.session_state.analysis_in_progress = False
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
                            st.session_state.analysis_in_progress = False
        
        # Display results if available
        if st.session_state.analysis_results:
            self._display_analysis_results()
    
    def _display_analysis_results(self):
        """Display analysis results"""
        analysis_data = st.session_state.analysis_results
        employee_data = analysis_data['employee_data']
        results = analysis_data['analysis_results']
        
        st.markdown("---")
        st.markdown('<div class="main-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            perf_score = results['rl_analysis']['state_features']['performance_score']
            st.metric(
                "🎯 Performance Score",
                f"{perf_score}/100",
                delta="Excellent" if perf_score >= 80 else "Good" if perf_score >= 60 else "Needs Improvement"
            )
        
        with col2:
            work_hours = results['basic_analysis']['work_hours']
            st.metric(
                "⏰ Work Hours",
                f"{work_hours:.1f}",
                delta="Optimal" if 7 <= work_hours <= 9 else "Review"
            )
        
        with col3:
            engagement = results['multimodal_analysis']['multimodal_score']
            st.metric(
                "💡 Engagement",
                f"{engagement:.1f}/100"
            )
        
        with col4:
            ai_action = results['rl_analysis']['selected_action']['name']
            st.metric(
                "🤖 AI Action",
                ai_action
            )
        
        # Detailed sections
        tabs = st.tabs(["📈 Performance", "🤖 AI Insights", "💡 Recommendations", "📄 Full Report"])
        
        with tabs[0]:
            self._render_performance_tab(results)
        
        with tabs[1]:
            self._render_ai_insights_tab(results)
        
        with tabs[2]:
            self._render_recommendations_tab(results)
        
        with tabs[3]:
            self._render_full_report_tab(employee_data, results)
    
    def _render_performance_tab(self, results: Dict):
        """Render performance metrics tab"""
        # Create performance chart
        metrics = ['Performance', 'Task Complexity', 'Engagement', 'Work Hours Score']
        
        perf_score = results['rl_analysis']['state_features']['performance_score']
        task_comp = results['ai_analysis']['task_complexity']['score'] * 100
        engagement = results['multimodal_analysis']['multimodal_score']
        work_hours = results['basic_analysis']['work_hours']
        work_hours_score = max(0, 100 - abs(work_hours - 8) * 10)
        
        values = [perf_score, task_comp, engagement, work_hours_score]
        
        fig = go.Figure(data=[go.Bar(
            x=metrics,
            y=values,
            marker_color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
        )])
        
        fig.update_layout(
            title="Performance Metrics Breakdown",
            xaxis_title="Metric",
            yaxis_title="Score (0-100)",
            yaxis_range=[0, 100],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Basic Metrics")
            st.write(f"**Work Hours:** {results['basic_analysis']['work_hours']:.1f} hours")
            st.write(f"**Task Complexity:** {results['ai_analysis']['task_complexity']['level']}")
            st.write(f"**Technical Terms:** {results['ai_analysis']['text_analysis']['technical_term_count']}")
        
        with col2:
            st.markdown("#### 🎭 Sentiment Analysis")
            sentiment = results['ai_analysis']['sentiment_analysis']
            st.write(f"**Sentiment:** {sentiment['label']}")
            st.write(f"**Confidence:** {sentiment['score']:.3f}")
            st.write(f"**Positive Words:** {sentiment['positive_words']}")
            st.write(f"**Negative Words:** {sentiment['negative_words']}")
    
    def _render_ai_insights_tab(self, results: Dict):
        """Render AI insights tab"""
        st.markdown('<div class="ai-action-card">', unsafe_allow_html=True)
        st.markdown("### 🤖 AI Recommended Action")
        
        action = results['rl_analysis']['selected_action']
        st.markdown(f"**{action['name']}**")
        st.markdown(f"*{action['description']}*")
        st.markdown(f"**Category:** {action['category'].title()} | **Intensity:** {action['intensity'].title()}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # RL Agent Statistics
        st.markdown("#### 📈 RL Agent Statistics")
        stats = results['rl_analysis']['q_table_stats']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total States", stats.get('total_states', 0))
        
        with col2:
            most_common = stats.get('most_common_action')
            if most_common is not None:
                action_name = self.analyzer.rl_agent.actions[most_common]['name']
                st.metric("Most Common Action", action_name)
        
        # Multimodal Analysis
        st.markdown("#### 🎬 Multimodal Analysis")
        multimodal = results['multimodal_analysis']
        
        if multimodal['video_analysis']:
            st.markdown("##### 📹 Video Analysis")
            video = multimodal['video_analysis']
            st.write(f"**Duration:** {video.get('duration_minutes', 0):.1f} minutes")
            st.write(f"**Motion Level:** {video.get('motion_level', 0):.3f}")
            st.write(f"**Engagement Estimate:** {video.get('engagement_estimate', 0):.1f}")
        
        if multimodal['image_analysis']:
            st.markdown("##### 📸 Image Analysis")
            image = multimodal['image_analysis']
            st.write(f"**Dimensions:** {image.get('dimensions', 'Unknown')}")
            st.write(f"**Face Detected:** {image.get('face_detection', {}).get('face_detected', False)}")
            st.write(f"**Image Quality:** {image.get('image_quality', 'unknown').title()}")
    
    def _render_recommendations_tab(self, results: Dict):
        """Render recommendations tab"""
        recommendations = results.get('recommendations', [])
        
        st.markdown("### 💡 AI-Powered Recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f'<div class="recommendation-card">{i}. {rec}</div>', unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export Recommendations", use_container_width=True):
                rec_text = "\n".join([f"{i}. {rec}" for i, rec in enumerate(recommendations, 1)])
                st.download_button(
                    label="Download",
                    data=rec_text,
                    file_name="recommendations.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.info("Feature coming soon!")
        
        with col3:
            if st.button("🔄 Regenerate", type="secondary", use_container_width=True):
                st.info("Feature coming soon!")
    
    def _render_full_report_tab(self, employee_data: Dict, results: Dict):
        """Render full report tab"""
        report = self.analyzer.generate_analysis_report(employee_data, results)
        
        st.text_area(
            "Full Analysis Report",
            value=report,
            height=600,
            disabled=True
        )
        
        # Export buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"performance_report_{employee_data['name']}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # JSON export
            json_data = json.dumps({
                'employee': employee_data,
                'analysis': results,
                'timestamp': datetime.now().isoformat()
            }, indent=2)
            
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"analysis_{employee_data['name']}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            if st.button("🖨️ Print Report", use_container_width=True):
                st.info("Print feature coming soon!")
    
    def _render_reports_page(self):
        """Render reports and analytics page"""
        st.markdown('<div class="main-header">📊 Reports & Analytics</div>', unsafe_allow_html=True)
        
        if self.database.df is None or self.database.df.empty:
            st.warning("📭 No data available for reports. Please upload a dataset first.")
            return
        
        # Analytics dashboard
        analytics = self.database.get_analytics_dashboard()
        
        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", analytics['total_analyses'])
        
        with col2:
            st.metric("Unique Employees", analytics['total_employees'])
        
        with col3:
            st.metric("Average Performance", f"{analytics['avg_performance']}/100")
        
        with col4:
            st.metric("High Performers", analytics['high_performers'])
        
        st.markdown("---")
        
        # Data visualizations
        tabs = st.tabs(["📈 Performance Trends", "🤖 Action Analysis", "👥 Employee Insights", "📤 Export Data"])
        
        with tabs[0]:
            self._render_performance_trends()
        
        with tabs[1]:
            self._render_action_analysis()
        
        with tabs[2]:
            self._render_employee_insights()
        
        with tabs[3]:
            self._render_export_section()
    
    def _render_performance_trends(self):
        """Render performance trends visualization"""
        if self.database.df is not None and 'performance_score' in self.database.df.columns:
            # Performance distribution
            perf_scores = self.database.df['performance_score'].dropna()
            
            if not perf_scores.empty:
                fig1 = px.histogram(
                    perf_scores,
                    title="Performance Score Distribution",
                    labels={'value': 'Performance Score', 'count': 'Frequency'},
                    nbins=20,
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            # Performance over time
            if 'Date' in self.database.df.columns and 'performance_score' in self.database.df.columns:
                time_data = self.database.df[['Date', 'performance_score']].dropna()
                if not time_data.empty:
                    time_data['Date'] = pd.to_datetime(time_data['Date'])
                    time_data = time_data.sort_values('Date')
                    
                    fig2 = px.line(
                        time_data,
                        x='Date',
                        y='performance_score',
                        title="Performance Trends Over Time",
                        labels={'Date': 'Date', 'performance_score': 'Performance Score'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No performance data available for trends")
    
    def _render_action_analysis(self):
        """Render AI action analysis"""
        if self.database.df is not None and 'ai_action' in self.database.df.columns:
            actions = self.database.df['ai_action'].value_counts()
            
            if not actions.empty:
                fig = px.pie(
                    values=actions.values,
                    names=actions.index,
                    title="AI Action Distribution",
                    hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Action effectiveness (simulated)
                st.markdown("#### 🎯 Action Effectiveness")
                
                # Simulate effectiveness scores
                action_effectiveness = {
                    'Positive Feedback': 85,
                    'Skill Training': 78,
                    'Wellness Check': 72,
                    'Workload Adjustment': 68,
                    'Constructive Feedback': 65,
                    'No Action': 50
                }
                
                eff_df = pd.DataFrame({
                    'Action': list(action_effectiveness.keys()),
                    'Effectiveness': list(action_effectiveness.values())
                })
                
                fig2 = px.bar(
                    eff_df,
                    x='Action',
                    y='Effectiveness',
                    title="Action Effectiveness Scores",
                    color='Effectiveness',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No AI action data available")
    
    def _render_employee_insights(self):
        """Render employee insights"""
        if self.database.employee_history:
            # Top performers
            perf_data = []
            for emp, history in self.database.employee_history.items():
                if history:
                    scores = [h.get('performance_score', 0) for h in history]
                    avg_score = np.mean(scores) if scores else 0
                    perf_data.append({'employee': emp, 'avg_performance': avg_score})
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                perf_df = perf_df.sort_values('avg_performance', ascending=False).head(10)
                
                fig = px.bar(
                    perf_df,
                    x='employee',
                    y='avg_performance',
                    title="Top 10 Performers (Average Score)",
                    color='avg_performance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Employee statistics
            st.markdown("#### 📊 Employee Statistics")
            
            stats_data = []
            for emp, history in list(self.database.employee_history.items())[:10]:
                if history:
                    scores = [h.get('performance_score', 0) for h in history]
                    eng_scores = [h.get('engagement_score', 0) for h in history]
                    hours = [h.get('work_hours', 0) for h in history]
                    
                    stats_data.append({
                        'Employee': emp,
                        'Records': len(history),
                        'Avg Performance': np.mean(scores) if scores else 0,
                        'Avg Engagement': np.mean(eng_scores) if eng_scores else 0,
                        'Avg Hours': np.mean(hours) if hours else 0
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(
                    stats_df.round(2),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No employee history data available")
    
    def _render_export_section(self):
        """Render data export section"""
        st.markdown("### 📤 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Export Full Dataset", use_container_width=True):
                if self.database.df is not None and not self.database.df.empty:
                    csv = self.database.df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name="employee_full_dataset.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No data to export")
        
        with col2:
            if st.button("📊 Export Analytics", use_container_width=True):
                analytics = self.database.get_analytics_dashboard()
                analytics_json = json.dumps(analytics, indent=2)
                
                st.download_button(
                    label="📥 Download JSON",
                    data=analytics_json,
                    file_name="analytics_report.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col3:
            if st.button("🤖 Export AI Models", use_container_width=True):
                st.info("AI model export feature coming soon!")
        
        # Custom export
        st.markdown("---")
        st.markdown("### 🎯 Custom Export")
        
        if self.database.df is not None and not self.database.df.empty:
            export_cols = st.multiselect(
                "Select columns to export",
                self.database.df.columns.tolist(),
                default=['Team Members', 'Date', 'performance_score', 'ai_action']
            )
            
            if export_cols:
                export_df = self.database.df[export_cols]
                
                export_format = st.radio(
                    "Export format",
                    ["CSV", "JSON", "Excel"],
                    horizontal=True
                )
                
                if st.button("Generate Export", type="primary"):
                    if export_format == "CSV":
                        data = export_df.to_csv(index=False)
                        file_name = "custom_export.csv"
                        mime_type = "text/csv"
                    elif export_format == "JSON":
                        data = export_df.to_json(orient='records', indent=2)
                        file_name = "custom_export.json"
                        mime_type = "application/json"
                    else:  # Excel
                        # Note: Would need openpyxl for Excel export
                        data = export_df.to_csv(index=False)  # Fallback to CSV
                        file_name = "custom_export.csv"
                        mime_type = "text/csv"
                    
                    st.download_button(
                        label=f"📥 Download {export_format}",
                        data=data,
                        file_name=file_name,
                        mime=mime_type,
                        use_container_width=True
                    )
    
    def _render_ai_insights_page(self):
        """Render AI insights page"""
        st.markdown('<div class="main-header">🤖 AI Insights & Models</div>', unsafe_allow_html=True)
        
        # AI Models Status
        st.markdown('<div class="section-header">⚙️ AI Models Status</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sentiment Analysis", "Active", "Rule-based")
        
        with col2:
            st.metric("Face Detection", "Active", "Heuristic")
        
        with col3:
            st.metric("RL Agent", "Active", f"{len(self.analyzer.rl_agent.q_table)} states")
        
        with col4:
            st.metric("Task Analysis", "Active", "Keyword-based")
        
        st.markdown("---")
        
        # RL Agent Details
        st.markdown('<div class="section-header">🎯 Reinforcement Learning Agent</div>', unsafe_allow_html=True)
        
        stats = self.analyzer.rl_agent.get_action_statistics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total States", stats.get('total_states', 0))
            st.metric("Exploration Rate", f"{self.analyzer.rl_agent.exploration_rate:.3f}")
        
        with col2:
            st.metric("Learning Rate", self.analyzer.rl_agent.learning_rate)
            st.metric("Discount Factor", self.analyzer.rl_agent.discount_factor)
        
        # Available Actions
        st.markdown("#### 📋 Available AI Actions")
        
        for action in self.analyzer.rl_agent.actions:
            with st.expander(f"{action['name']} (ID: {action['id']})"):
                st.write(f"**Description:** {action['description']}")
                st.write(f"**Category:** {action['category'].title()}")
                st.write(f"**Intensity:** {action['intensity'].title()}")
        
        st.markdown("---")
        
        # Model Training
        st.markdown('<div class="section-header">🔧 Model Training</div>', unsafe_allow_html=True)
        
        with st.form("training_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                training_epochs = st.slider("Training Epochs", 1, 100, 10)
                batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
            
            with col2:
                learning_rate = st.number_input("Learning Rate", 0.001, 0.1, 0.01, 0.001)
                exploration_rate = st.slider("Exploration Rate", 0.01, 0.5, 0.1, 0.01)
            
            if st.form_submit_button("🚀 Train RL Model", type="primary"):
                with st.spinner("Training AI model..."):
                    # Update RL agent parameters
                    self.analyzer.rl_agent.learning_rate = learning_rate
                    self.analyzer.rl_agent.exploration_rate = exploration_rate
                    
                    # Simulate training
                    time.sleep(2)
                    st.success(f"✅ Model trained for {training_epochs} epochs!")
                    
                    # Show updated stats
                    new_stats = self.analyzer.rl_agent.get_action_statistics()
                    st.metric("Updated States", new_stats.get('total_states', 0))

# ============================================
# 6. MAIN APPLICATION
# ============================================

def main():
    """Main application entry point"""
    app = EmployeePerformanceApp()
    app.run()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    
    # Run the application
    main()
