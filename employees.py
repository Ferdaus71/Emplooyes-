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

# Try to import cv2 with fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    st.warning("OpenCV (cv2) not available. Some video features will be limited.")
    CV2_AVAILABLE = False
    # Create a dummy cv2 module
    class DummyCV2:
        CAP_PROP_FPS = 5
        CAP_PROP_FRAME_COUNT = 7
        COLOR_BGR2GRAY = 6
        COLOR_RGB2GRAY = 7
        
        @staticmethod
        def VideoCapture(path):
            return DummyVideoCapture()
        
        @staticmethod
        def cvtColor(img, code):
            return img if len(img.shape) == 2 else np.mean(img, axis=2).astype(np.uint8)
    
    class DummyVideoCapture:
        def __init__(self):
            self.is_opened_val = True
            
        def isOpened(self):
            return self.is_opened_val
            
        def get(self, prop):
            if prop == 5:  # CAP_PROP_FPS
                return 30.0
            elif prop == 7:  # CAP_PROP_FRAME_COUNT
                return 100
            return 0
            
        def read(self):
            return False, None
            
        def release(self):
            pass
    
    cv2 = DummyCV2()

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    st.warning("PyTorch not available. RL features will be limited.")
    TORCH_AVAILABLE = False

from collections import deque
import random

# ============================================
# 1. REINFORCEMENT LEARNING MODEL
# ============================================

if TORCH_AVAILABLE:
    class RLPolicyNetwork(nn.Module):
        def __init__(self, state_size, action_size, hidden_size=64):
            super(RLPolicyNetwork, self).__init__()
            
            self.state_processor = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            self.action_value = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
            )
            
            self.state_value = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        
        def forward(self, state):
            state_features = self.state_processor(state)
            action_values = self.action_value(state_features)
            state_value = self.state_value(state_features)
            return action_values, state_value
    
    class HybridRLAgent:
        def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95):
            self.state_size = state_size
            self.action_size = action_size
            self.gamma = gamma
            self.learning_rate = learning_rate
            
            self.policy_net = RLPolicyNetwork(state_size, action_size)
            self.target_net = RLPolicyNetwork(state_size, action_size)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
            
            self.memory = deque(maxlen=1000)
            
            self.actions = [
                "No action needed",
                "Provide positive reinforcement",
                "Schedule wellness check",
                "Offer skill training",
                "Adjust workload",
                "Recommend promotion track",
                "Provide constructive feedback"
            ]
            
            self.batch_size = 16
            self.update_target_freq = 50
            self.steps_done = 0
        
        def select_action(self, state, epsilon=0.1):
            if random.random() < epsilon:
                return random.randint(0, self.action_size - 1)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_values, _ = self.policy_net(state_tensor)
                return torch.argmax(action_values).item()
        
        def store_experience(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))
        
        def train_step(self):
            if len(self.memory) < self.batch_size:
                return 0
            
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)
            
            current_q_values, _ = self.policy_net(states)
            current_q = current_q_values.gather(1, actions)
            
            with torch.no_grad():
                next_q_values, _ = self.target_net(next_states)
                next_q = next_q_values.max(1)[0].unsqueeze(1)
            
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
            loss = nn.MSELoss()(current_q, target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.steps_done += 1
            if self.steps_done % self.update_target_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            return loss.item()
        
        def get_action_description(self, action_idx):
            if 0 <= action_idx < len(self.actions):
                return self.actions[action_idx]
            return "Unknown action"
        
        def save_model(self, path):
            torch.save({
                'policy_state_dict': self.policy_net.state_dict(),
                'target_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)
        
        def load_model(self, path):
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=torch.device('cpu'))
                self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                return True
            return False
else:
    # Fallback RL agent without PyTorch
    class HybridRLAgent:
        def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95):
            self.state_size = state_size
            self.action_size = action_size
            self.gamma = gamma
            self.learning_rate = learning_rate
            
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
            # Simple rule-based action selection
            work_hours = state[0] * 24 if len(state) > 0 else 8
            task_complexity = state[1] if len(state) > 1 else 0.5
            
            if work_hours > 10:
                return 2  # Wellness check
            elif task_complexity > 0.7:
                return 3  # Skill training
            else:
                return 1  # Positive reinforcement
        
        def store_experience(self, state, action, reward, next_state, done):
            pass
        
        def train_step(self):
            return 0
        
        def get_action_description(self, action_idx):
            if 0 <= action_idx < len(self.actions):
                return self.actions[action_idx]
            return "Unknown action"
        
        def save_model(self, path):
            pass
        
        def load_model(self, path):
            return False

# ============================================
# 2. DATA MANAGEMENT AND PROCESSING
# ============================================

class EmployeeDatabase:
    def __init__(self):
        self.df = None
        self.employee_data = {}
        self.load_existing_data()
    
    def load_existing_data(self):
        default_path = "employee_multimodal_dataset.csv"
        if os.path.exists(default_path):
            try:
                self.df = pd.read_csv(default_path)
                self.process_dataframe()
                st.success(f"Loaded existing dataset with {len(self.df)} records")
            except Exception as e:
                st.error(f"Error loading existing dataset: {e}")
                self.df = pd.DataFrame()
        else:
            self.df = pd.DataFrame()
    
    def process_dataframe(self):
        if self.df is not None and not self.df.empty:
            required_columns = ['Team Members', 'Signed In', 'Signed Out', 'Completed Task']
            for col in required_columns:
                if col not in self.df.columns:
                    self.df[col] = ''
            
            time_columns = ['Signed In', 'Signed Out']
            for col in time_columns:
                if col in self.df.columns:
                    try:
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce').dt.strftime('%I:%M %p')
                    except:
                        pass
            
            self.df = self.df.fillna('')
            
            self.employee_data = {}
            for idx, row in self.df.iterrows():
                employee_name = str(row.get('Team Members', '')).strip()
                if employee_name:
                    self.employee_data[employee_name] = {
                        'sign_in': row.get('Signed In', ''),
                        'sign_out': row.get('Signed Out', ''),
                        'task': row.get('Completed Task', ''),
                        'date': row.get('Date', ''),
                        'session_id': row.get('session_id', ''),
                        'selfie_path': row.get('selfie_path', ''),
                        'session_video_path': row.get('session_video_path', '')
                    }
    
    def upload_dataset(self, uploaded_file):
        if uploaded_file is None:
            return "No file uploaded. Please select a CSV file.", pd.DataFrame()
        
        try:
            self.df = pd.read_csv(uploaded_file)
            self.process_dataframe()
            message = f"Dataset uploaded successfully. Loaded {len(self.df)} employee records."
            return message, self.df
            
        except Exception as e:
            return f"Error uploading dataset: {str(e)}", pd.DataFrame()
    
    def search_employees(self, search_term):
        if not search_term or self.df is None or self.df.empty:
            return []
        
        search_term = search_term.lower().strip()
        matches = []
        
        if 'Team Members' in self.df.columns:
            mask = self.df['Team Members'].astype(str).str.lower().str.contains(search_term, na=False)
            matches_df = self.df[mask]
            
            for idx, row in matches_df.iterrows():
                employee_name = str(row['Team Members']).strip()
                if employee_name:
                    matches.append(employee_name)
        
        matches = list(dict.fromkeys(matches))[:10]
        return matches
    
    def get_employee_details(self, employee_name):
        if not employee_name or employee_name not in self.employee_data:
            return '', '', '', ''
        
        details = self.employee_data.get(employee_name, {})
        return (
            details.get('sign_in', ''),
            details.get('sign_out', ''),
            details.get('task', ''),
            details.get('date', '')
        )
    
    def save_new_record(self, employee_data):
        if self.df is None:
            self.df = pd.DataFrame()
        
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
        
        self.df = pd.concat([self.df, pd.DataFrame([new_record])], ignore_index=True)
        
        self.employee_data[employee_data.get('name', '')] = {
            'sign_in': employee_data.get('sign_in', ''),
            'sign_out': employee_data.get('sign_out', ''),
            'task': employee_data.get('task', ''),
            'date': employee_data.get('date', '')
        }
        
        self.save_to_csv()
        
        return f"Record saved for {employee_data.get('name', '')}"
    
    def save_to_csv(self):
        if self.df is not None and not self.df.empty:
            self.df.to_csv('employee_multimodal_dataset.csv', index=False)

# ============================================
# 3. MULTIMODAL FEATURE EXTRACTOR
# ============================================

class MultimodalFeatureExtractor:
    def __init__(self):
        self.feature_cache = {}
    
    def extract_video_features(self, video_file):
        if video_file is None:
            return np.zeros(5)
        
        try:
            if not CV2_AVAILABLE:
                # Return dummy features if cv2 not available
                return np.array([30.0, 100, 3.33, 500.0, 1000.0], dtype=np.float32)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_file.read())
                video_path = tmp_file.name
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                os.unlink(video_path)
                return np.zeros(5)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            features = [fps, total_frames, duration]
            
            try:
                # Try to read a few frames
                frame_count = 0
                motion_values = []
                
                while frame_count < min(total_frames, 10):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frame_variance = np.var(gray)
                        motion_values.append(frame_variance)
                    
                    frame_count += 1
                
                if motion_values:
                    avg_motion = np.mean(motion_values)
                    max_motion = np.max(motion_values)
                    features.extend([avg_motion, max_motion])
                else:
                    features.extend([0, 0])
            except:
                features.extend([0, 0])
            
            cap.release()
            os.unlink(video_path)
            
            while len(features) < 5:
                features.append(0)
            
            return np.array(features[:5], dtype=np.float32)
            
        except Exception as e:
            st.error(f"Video processing error: {str(e)}")
            return np.zeros(5)
    
    def extract_image_features(self, image_file):
        if image_file is None:
            return np.zeros(5)
        
        try:
            img = Image.open(image_file)
            img_array = np.array(img)
            
            if len(img_array.shape) == 0:
                return np.zeros(5)
            
            height, width = img_array.shape[:2]
            
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            features = [
                width, height,
                brightness, contrast,
                1 if height > 50 and width > 50 else 0
            ]
            
            return np.array(features[:5], dtype=np.float32)
            
        except Exception as e:
            st.error(f"Image processing error: {str(e)}")
            return np.zeros(5)

# ============================================
# 4. HYBRID PERFORMANCE ANALYZER
# ============================================

class HybridPerformanceAnalyzer:
    def __init__(self):
        self.feature_extractor = MultimodalFeatureExtractor()
        
        # Reduced state size for simplicity
        state_size = 2 + 5 + 5  # work_hours + task_complexity + 5 video features + 5 image features
        action_size = 7
        
        self.rl_agent = HybridRLAgent(state_size, action_size)
        
        if os.path.exists('rl_agent_model.pth') and TORCH_AVAILABLE:
            try:
                self.rl_agent.load_model('rl_agent_model.pth')
                st.success("Loaded pre-trained RL model")
            except:
                st.info("Could not load RL model. Using default agent.")
        else:
            st.info("No pre-trained model found. Using rule-based agent.")
    
    def create_state_vector(self, employee_data, video_file, image_file):
        work_hours = self.calculate_work_hours(
            employee_data.get('sign_in', ''),
            employee_data.get('sign_out', '')
        )
        
        task = employee_data.get('task', '')
        task_complexity = self.assess_task_complexity(task)
        
        video_features = self.feature_extractor.extract_video_features(video_file)
        image_features = self.feature_extractor.extract_image_features(image_file)
        
        state_vector = np.concatenate([
            [work_hours / 24.0, task_complexity],
            video_features,
            image_features
        ])
        
        return state_vector.astype(np.float32)
    
    def calculate_reward(self, performance_score, work_hours, action_taken):
        base_reward = performance_score / 100.0
        
        if work_hours > 12:
            hour_penalty = (work_hours - 12) * 0.1
            base_reward -= hour_penalty
        elif work_hours < 4:
            hour_penalty = (4 - work_hours) * 0.05
            base_reward -= hour_penalty
        
        action_bonuses = {
            0: 0.0, 1: 0.2, 2: 0.1, 3: 0.15, 4: -0.1, 5: 0.3, 6: 0.05
        }
        
        bonus = action_bonuses.get(action_taken, 0.0)
        
        reward = base_reward + bonus
        return np.clip(reward, -1.0, 1.0)
    
    def analyze_performance(self, employee_data, video_file, image_file, train_agent=False):
        state = self.create_state_vector(employee_data, video_file, image_file)
        
        epsilon = 0.1 if train_agent else 0.0
        action_idx = self.rl_agent.select_action(state, epsilon)
        
        work_hours = self.calculate_work_hours(
            employee_data.get('sign_in', ''),
            employee_data.get('sign_out', '')
        )
        
        task = employee_data.get('task', '')
        task_complexity = self.assess_task_complexity(task)
        
        video_features = self.feature_extractor.extract_video_features(video_file)
        engagement_score = min(100, (video_features[3] / 1000) * 100) if len(video_features) > 3 else 50
        
        image_features = self.feature_extractor.extract_image_features(image_file)
        image_quality = min(100, (image_features[3] / 100) * 100) if len(image_features) > 3 else 50
        
        performance_score = self.calculate_performance_score(
            work_hours, task_complexity, engagement_score, image_quality
        )
        
        reward = self.calculate_reward(performance_score, work_hours, action_idx)
        
        if train_agent and TORCH_AVAILABLE:
            next_state = state.copy()
            self.rl_agent.store_experience(state, action_idx, reward, next_state, False)
            loss = self.rl_agent.train_step()
            if loss:
                st.info(f"RL Model trained. Loss: {loss:.4f}")
        
        recommendations = self.generate_hybrid_recommendations(
            performance_score, action_idx, work_hours, engagement_score, image_quality
        )
        
        charts = self.create_visualizations(
            performance_score, engagement_score, image_quality, work_hours, task_complexity
        )
        
        report = self.generate_hybrid_report(
            employee_data, performance_score, action_idx,
            engagement_score, image_quality, recommendations
        )
        
        return {
            'report': report,
            'performance_score': performance_score,
            'action_idx': action_idx,
            'action_description': self.rl_agent.get_action_description(action_idx),
            'engagement_score': engagement_score,
            'image_quality': image_quality,
            'work_hours': work_hours,
            'recommendations': recommendations,
            'charts': charts,
            'rl_reward': reward
        }
    
    def calculate_work_hours(self, sign_in, sign_out):
        try:
            if not sign_in or not sign_out:
                return 8.0
            
            # Handle both 12-hour and 24-hour formats
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
            
            if duration < 0:
                duration += 24
            
            return round(duration, 2)
            
        except:
            return 8.0
    
    def assess_task_complexity(self, task_description):
        if not task_description:
            return 0.5
        
        task_lower = task_description.lower()
        complexity = 0.5
        
        complexity_keywords = {
            'design': 0.8, 'develop': 0.9, 'implement': 0.85,
            'create': 0.7, 'build': 0.75, 'analyze': 0.6,
            'review': 0.5, 'update': 0.4, 'fix': 0.3, 'test': 0.4
        }
        
        for keyword, score in complexity_keywords.items():
            if keyword in task_lower:
                complexity = max(complexity, score)
        
        return complexity
    
    def calculate_performance_score(self, work_hours, task_complexity, engagement, image_quality):
        weights = {
            'work_hours': 0.25,
            'task_complexity': 0.25,
            'engagement': 0.30,
            'image_quality': 0.20
        }
        
        hours_score = min(100, (work_hours / 8) * 100) if work_hours <= 12 else 100
        
        score = (
            hours_score * weights['work_hours'] +
            task_complexity * 100 * weights['task_complexity'] +
            engagement * weights['engagement'] +
            image_quality * weights['image_quality']
        )
        
        return round(min(100, score), 1)
    
    def generate_hybrid_recommendations(self, performance_score, action_idx, work_hours, engagement, image_quality):
        recommendations = []
        
        action_descriptions = {
            0: "Maintain current management approach.",
            1: "Provide specific positive feedback on recent achievements.",
            2: "Schedule a wellness discussion and consider work-life balance adjustments.",
            3: "Identify skill gaps and provide targeted training opportunities.",
            4: "Review current workload and consider redistribution if necessary.",
            5: "Discuss career progression and development opportunities.",
            6: "Provide constructive feedback with specific improvement areas."
        }
        
        recommendations.append(f"Recommended Action: {action_descriptions.get(action_idx, 'Monitor performance')}")
        
        if performance_score >= 85:
            recommendations.append("Excellent overall performance. Consider leadership opportunities.")
        elif performance_score >= 70:
            recommendations.append("Good performance. Focus on skill development.")
        elif performance_score >= 50:
            recommendations.append("Average performance. Identify improvement areas.")
        else:
            recommendations.append("Performance needs immediate attention.")
        
        if work_hours > 10:
            recommendations.append("High work hours detected. Monitor for burnout risk.")
        elif work_hours < 6:
            recommendations.append("Low work hours. Assess task allocation and engagement.")
        
        if engagement < 40:
            recommendations.append("Low engagement detected. Review task interest and motivation.")
        
        if image_quality < 40:
            recommendations.append("Poor image quality may indicate workspace issues.")
        
        return recommendations
    
    def create_visualizations(self, performance_score, engagement, image_quality, work_hours, task_complexity):
        charts = {}
        
        # Simple bar chart instead of radar for compatibility
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
    
    def generate_hybrid_report(self, employee_data, performance_score, action_idx,
                               engagement_score, image_quality, recommendations):
        report = f"""
        EMPLOYEE PERFORMANCE ANALYSIS REPORT
        {'=' * 50}
        
        Employee Information:
        - Name: {employee_data.get('name', 'N/A')}
        - Date: {employee_data.get('date', 'N/A')}
        - Task: {employee_data.get('task', 'N/A')}
        - Work Hours: {self.calculate_work_hours(employee_data.get('sign_in', ''), employee_data.get('sign_out', ''))}
        
        Performance Analysis:
        - Overall Score: {performance_score}/100
        - Performance Level: {'Excellent' if performance_score >= 85 else 'Good' if performance_score >= 70 else 'Average' if performance_score >= 50 else 'Needs Improvement'}
        
        Multimodal Analysis:
        - Engagement Score: {engagement_score:.1f}/100
        - Image Quality Score: {image_quality:.1f}/100
        
        Recommended Action:
        - {self.rl_agent.get_action_description(action_idx)}
        
        Recommendations:
        """
        
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        report += f"\n{'=' * 50}"
        report += f"\nReport Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return report

# ============================================
# 5. STREAMLIT APPLICATION
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
        st.session_state.analyzer = HybridPerformanceAnalyzer()
    
    if 'current_employee' not in st.session_state:
        st.session_state.current_employee = None
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='main-header'>📊 Employee Performance Analyzer</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            ["Dashboard", "Dataset Management", "Performance Analysis", "Reports"]
        )
        
        st.markdown("---")
        st.markdown("### System Status")
        
        # System checks
        if not CV2_AVAILABLE:
            st.warning("⚠️ OpenCV not available")
        if not TORCH_AVAILABLE:
            st.warning("⚠️ PyTorch not available")
        
        if st.session_state.database.df is not None:
            st.success(f"✅ Dataset: {len(st.session_state.database.df)} records")
        else:
            st.warning("📊 No dataset loaded")
        
        st.markdown("---")
        
        if st.button("Clear All Data"):
            st.session_state.analysis_results = None
            st.session_state.current_employee = None
            st.rerun()
    
    # Main content
    if page == "Dashboard":
        display_dashboard()
    elif page == "Dataset Management":
        display_dataset_management()
    elif page == "Performance Analysis":
        display_performance_analysis()
    elif page == "Reports":
        display_reports()

def display_dashboard():
    st.markdown("<div class='main-header'>Dashboard Overview</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Employees", 
                 len(st.session_state.database.df) if st.session_state.database.df is not None else 0)
    
    with col2:
        avg_score = st.session_state.analysis_results.get('performance_score', 0) if st.session_state.analysis_results else 0
        st.metric("Performance Score", f"{avg_score:.1f}/100")
    
    with col3:
        model_status = "Available" if TORCH_AVAILABLE else "Limited"
        st.metric("RL Model", model_status)
    
    st.markdown("---")
    
    # Recent Analysis
    if st.session_state.analysis_results:
        st.markdown("### Recent Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(st.session_state.analysis_results['charts']['bar'], use_container_width=True)
        
        with col2:
            st.markdown("#### Key Metrics")
            st.markdown(f"**Performance:** {st.session_state.analysis_results['performance_score']:.1f}/100")
            st.markdown(f"**Engagement:** {st.session_state.analysis_results['engagement_score']:.1f}/100")
            st.markdown(f"**Work Hours:** {st.session_state.analysis_results['work_hours']:.1f}")
            st.markdown(f"**Recommended Action:** {st.session_state.analysis_results['action_description']}")
        
        # Recommendations
        st.markdown("#### Recommendations")
        recommendations = st.session_state.analysis_results.get('recommendations', [])
        for rec in recommendations[:3]:
            st.markdown(f"• {rec}")
    
    else:
        st.info("No analysis results available. Go to 'Performance Analysis' page to analyze an employee.")

def display_dataset_management():
    st.markdown("<div class='main-header'>Dataset Management</div>", unsafe_allow_html=True)
    
    # Upload section
    st.markdown("### Upload Dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        if st.button("Process Dataset"):
            with st.spinner("Processing..."):
                message, df = st.session_state.database.upload_dataset(uploaded_file)
                st.success(message)
                st.rerun()
    
    # Dataset preview
    if st.session_state.database.df is not None and not st.session_state.database.df.empty:
        st.markdown("### Dataset Preview")
        st.dataframe(st.session_state.database.df.head(10), use_container_width=True)
        
        # Employee search
        st.markdown("### Employee Search")
        search_term = st.text_input("Search by name")
        
        if search_term:
            matches = st.session_state.database.search_employees(search_term)
            
            if matches:
                selected_employee = st.selectbox("Select employee", matches)
                
                if selected_employee:
                    sign_in, sign_out, task, date = st.session_state.database.get_employee_details(selected_employee)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_input("Sign In", value=sign_in, disabled=True)
                        st.text_input("Task", value=task, disabled=True)
                    with col2:
                        st.text_input("Sign Out", value=sign_out, disabled=True)
                        st.text_input("Date", value=date, disabled=True)
                    
                    if st.button("Use for Analysis"):
                        st.session_state.current_employee = selected_employee
                        st.success(f"Employee {selected_employee} selected")
                        st.rerun()
            else:
                st.warning("No matches found")
    else:
        st.warning("No dataset loaded. Upload a CSV file to get started.")

def display_performance_analysis():
    st.markdown("<div class='main-header'>Performance Analysis</div>", unsafe_allow_html=True)
    
    # System warnings
    if not CV2_AVAILABLE:
        st.warning("Video analysis features are limited. OpenCV is not available.")
    if not TORCH_AVAILABLE:
        st.warning("RL features are limited. PyTorch is not available.")
    
    # Employee information form
    with st.form("analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            employee_name = st.text_input("Employee Name", 
                                        value=st.session_state.current_employee or "")
            analysis_date = st.text_input("Analysis Date", 
                                        value=datetime.now().strftime("%d.%m.%Y"))
        
        with col2:
            sign_in_time = st.selectbox("Sign In Time", 
                                       [f"{h:02d}:{m:02d}" for h in range(0, 24) for m in [0, 30]],
                                       index=18)
            sign_out_time = st.selectbox("Sign Out Time", 
                                        [f"{h:02d}:{m:02d}" for h in range(0, 24) for m in [0, 30]],
                                        index=34)
        
        completed_task = st.text_area("Completed Task", 
                                    placeholder="Describe the task...",
                                    height=100)
        
        # Multimedia upload
        st.markdown("### Multimedia Data")
        col1, col2 = st.columns(2)
        
        with col1:
            session_video = st.file_uploader("Session Video", type=['mp4', 'avi', 'mov'])
        
        with col2:
            selfie_image = st.file_uploader("Selfie/Workspace Image", type=['jpg', 'jpeg', 'png'])
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.form_submit_button("🚀 Analyze Performance", type="primary", use_container_width=True)
        
        if analyze_button:
            if not employee_name:
                st.error("Please enter employee name")
            else:
                with st.spinner("Analyzing..."):
                    employee_data = {
                        'name': employee_name,
                        'date': analysis_date,
                        'sign_in': sign_in_time,
                        'sign_out': sign_out_time,
                        'task': completed_task
                    }
                    
                    st.session_state.analysis_results = st.session_state.analyzer.analyze_performance(
                        employee_data, session_video, selfie_image, False
                    )
                    
                    # Save record
                    try:
                        employee_data['selfie_path'] = selfie_image.name if selfie_image else ''
                        employee_data['session_video_path'] = session_video.name if session_video else ''
                        result = st.session_state.database.save_new_record(employee_data)
                        st.success(result)
                    except Exception as e:
                        st.error(f"Error saving record: {e}")
                    
                    st.rerun()
    
    # Display results
    if st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("### Analysis Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Performance", f"{st.session_state.analysis_results['performance_score']:.1f}")
        
        with col2:
            st.metric("Engagement", f"{st.session_state.analysis_results['engagement_score']:.1f}")
        
        with col3:
            st.metric("Work Hours", f"{st.session_state.analysis_results['work_hours']:.1f}")
        
        with col4:
            st.metric("Action", st.session_state.analysis_results['action_description'].split()[0])
        
        # Chart
        st.plotly_chart(st.session_state.analysis_results['charts']['bar'], use_container_width=True)
        
        # Recommendations
        st.markdown("### Recommendations")
        recommendations = st.session_state.analysis_results.get('recommendations', [])
        for rec in recommendations:
            st.markdown(f"• {rec}")
        
        # Report
        with st.expander("View Detailed Report"):
            st.text(st.session_state.analysis_results['report'])

def display_reports():
    st.markdown("<div class='main-header'>Reports</div>", unsafe_allow_html=True)
    
    if st.session_state.database.df is None or st.session_state.database.df.empty:
        st.warning("No data available")
        return
    
    # Summary
    st.markdown("### Dataset Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(st.session_state.database.df))
    
    with col2:
        unique_employees = st.session_state.database.df['Team Members'].nunique()
        st.metric("Unique Employees", unique_employees)
    
    with col3:
        st.metric("Data Columns", len(st.session_state.database.df.columns))
    
    # Export options
    st.markdown("### Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export CSV"):
            csv = st.session_state.database.df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="employee_data.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Generate Summary"):
            summary = f"""
            Dataset Summary
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Total Records: {len(st.session_state.database.df)}
            Columns: {', '.join(st.session_state.database.df.columns.tolist())}
            """
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
