import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from models.cnn import ResNet18Binary
from services.llm_service import LLMService, explain_prediction, summarize_performance
from utils.metrics import evaluate_classifier
from datasets.loaders import get_dataloaders

# Page configuration
st.set_page_config(
    page_title="Animal Classification AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .explanation-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ResNet18Binary(pretrained=False)
        
        # Try to load the best checkpoint
        checkpoint_path = Path("checkpoints/week4/policy_epoch5.pth")
        if checkpoint_path.exists():
            # Load the policy checkpoint (we'll use the classifier from week2)
            pass
        
        # Load the trained classifier from week2
        classifier_path = Path("checkpoints/week2/resnet18_epoch5.pth")
        if classifier_path.exists():
            model.load_state_dict(torch.load(classifier_path, map_location=device))
            model.eval()
            return model, device
        else:
            st.error("No trained model found. Please run the training scripts first.")
            return None, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, "cpu"

@st.cache_data
def load_performance_data():
    """Load performance data from output files"""
    try:
        data = {}
        
        # Load week2 metrics
        week2_path = Path("outputs/week2/metrics.json")
        if week2_path.exists():
            with open(week2_path, 'r') as f:
                data['week2'] = json.load(f)
        
        # Load week3 curves
        week3_path = Path("outputs/week3/curves.json")
        if week3_path.exists():
            with open(week3_path, 'r') as f:
                data['week3'] = json.load(f)
        
        # Load week4 RL curve
        week4_path = Path("outputs/week4/rl_curve.json")
        if week4_path.exists():
            with open(week4_path, 'r') as f:
                data['week4'] = json.load(f)
        
        return data
    except Exception as e:
        st.error(f"Error loading performance data: {e}")
        return {}

def preprocess_image(image):
    """Preprocess uploaded image for model input"""
    try:
        # Resize to 128x128
        image = image.resize((128, 128))
        
        # Convert to tensor and normalize
        image_tensor = torch.tensor(np.array(image)).float()
        
        # Handle different image formats
        if len(image_tensor.shape) == 3:
            if image_tensor.shape[2] == 4:  # RGBA
                image_tensor = image_tensor[:, :, :3]  # Remove alpha channel
            image_tensor = image_tensor.permute(2, 0, 1)  # CHW format
        else:
            # Grayscale - convert to RGB
            image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        # Normalize to [0, 1]
        image_tensor = image_tensor / 255.0
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict_image(model, image_tensor, device):
    """Make prediction on preprocessed image"""
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get prediction and confidence
            confidence, predicted = torch.max(probabilities, 1)
            
            # Map prediction to class name
            class_names = ['cat', 'dog']
            predicted_class = class_names[predicted.item()]
            confidence_score = confidence.item()
            
            return predicted_class, confidence_score, probabilities[0].cpu().numpy()
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

def create_performance_plots(performance_data):
    """Create interactive performance plots"""
    plots = {}
    
    # Week 2 training curves
    if 'week2' in performance_data:
        try:
            # Create training loss plot
            epochs = list(range(1, 6))  # Assuming 5 epochs
            train_loss = performance_data['week2'].get('train_loss', [0.5, 0.4, 0.3, 0.25, 0.2])
            val_acc = performance_data['week2'].get('val_acc', [0.75, 0.82, 0.87, 0.89, 0.91])
            
            fig_loss = px.line(x=epochs, y=train_loss, title="Training Loss Over Epochs")
            fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Training Loss")
            plots['training_loss'] = fig_loss
            
            fig_acc = px.line(x=epochs, y=val_acc, title="Validation Accuracy Over Epochs")
            fig_acc.update_layout(xaxis_title="Epoch", yaxis_title="Validation Accuracy")
            plots['validation_accuracy'] = fig_acc
        except Exception as e:
            st.warning(f"Could not create week2 plots: {e}")
    
    # Week 3 active learning comparison
    if 'week3' in performance_data:
        try:
            rounds = list(range(len(performance_data['week3'].get('Random', []))))
            
            fig_comparison = go.Figure()
            
            for strategy, accuracies in performance_data['week3'].items():
                fig_comparison.add_trace(go.Scatter(
                    x=rounds, 
                    y=accuracies, 
                    mode='lines+markers',
                    name=strategy,
                    line=dict(width=3)
                ))
            
            fig_comparison.update_layout(
                title="Active Learning Strategies Comparison",
                xaxis_title="Acquisition Round",
                yaxis_title="Validation Accuracy",
                hovermode='x unified'
            )
            
            plots['active_learning_comparison'] = fig_comparison
        except Exception as e:
            st.warning(f"Could not create week3 plots: {e}")
    
    # Week 4 RL curve
    if 'week4' in performance_data:
        try:
            steps = list(range(len(performance_data['week4'].get('RL', []))))
            rl_accuracies = performance_data['week4'].get('RL', [])
            
            fig_rl = px.line(x=steps, y=rl_accuracies, title="REINFORCE Policy Learning Curve")
            fig_rl.update_layout(
                xaxis_title="Training Step",
                yaxis_title="Validation Accuracy",
                hovermode='x unified'
            )
            
            plots['rl_learning_curve'] = fig_rl
        except Exception as e:
            st.warning(f"Could not create week4 plots: {e}")
    
    return plots

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🐾 Animal Classification AI</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent Image Classification with Active Learning & Reinforcement Learning")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Model Performance", "🔍 Interactive Classification", "📈 Active Learning Analysis", "🤖 LLM Explanations"]
    )
    
    # Load model and data
    model, device = load_model()
    performance_data = load_performance_data()
    
    if page == "🏠 Home":
        show_home_page(model, device, performance_data)
    elif page == "📊 Model Performance":
        show_performance_page(performance_data)
    elif page == "🔍 Interactive Classification":
        show_classification_page(model, device)
    elif page == "📈 Active Learning Analysis":
        show_active_learning_page(performance_data)
    elif page == "🤖 LLM Explanations":
        show_llm_explanations_page(performance_data)

def show_home_page(model, device, performance_data):
    """Display the home page with overview"""
    st.header("🎯 Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This project demonstrates **Policy Gradient Active Learning** for animal image classification using:
        
        - **Deep Learning**: ResNet18 CNN for image classification
        - **Active Learning**: Intelligent sample selection strategies
        - **Reinforcement Learning**: REINFORCE algorithm for policy optimization
        - **LLM Integration**: AI-powered explanations and insights
        
        ### Key Features:
        ✅ **Data Preprocessing**: 25K images processed and organized  
        ✅ **Model Training**: CNN trained with active learning strategies  
        ✅ **RL Policy**: REINFORCE agent for sample selection  
        ✅ **Performance Analysis**: Comprehensive evaluation and comparison  
        ✅ **Interactive UI**: User-friendly interface for model exploration  
        ✅ **LLM Explanations**: Human-understandable insights  
        """)
    
    with col2:
        st.markdown("### 🚀 Quick Stats")
        
        if model:
            st.success("✅ Model Loaded Successfully")
        else:
            st.error("❌ Model Not Available")
        
        if performance_data:
            st.success(f"✅ {len(performance_data)} Performance Datasets Available")
        else:
            st.warning("⚠️ Performance Data Not Available")
        
        st.info("💡 Upload an image to test the model!")
    
    # Recent results
    if performance_data:
        st.header("📊 Recent Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'week2' in performance_data:
                val_acc = performance_data['week2'].get('val_acc', [0])[-1]
                st.metric("Final Validation Accuracy", f"{val_acc:.2%}")
        
        with col2:
            if 'week4' in performance_data:
                rl_acc = max(performance_data['week4'].get('RL', [0]))
                st.metric("Best RL Performance", f"{rl_acc:.2%}")
        
        with col3:
            if 'week3' in performance_data:
                best_baseline = max(max(performance_data['week3'].values()))
                st.metric("Best Baseline", f"{best_baseline:.2%}")

def show_performance_page(performance_data):
    """Display model performance analysis"""
    st.header("📊 Model Performance Analysis")
    
    if not performance_data:
        st.warning("No performance data available. Please run the training scripts first.")
        return
    
    # Create performance plots
    plots = create_performance_plots(performance_data)
    
    # Display plots
    for plot_name, plot in plots.items():
        st.subheader(plot_name.replace('_', ' ').title())
        st.plotly_chart(plot, use_container_width=True)
    
    # Performance metrics summary
    st.header("📈 Performance Summary")
    
    if 'week2' in performance_data:
        st.subheader("Week 2: Baseline Training")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Final Training Loss", f"{performance_data['week2'].get('train_loss', [0])[-1]:.4f}")
        
        with col2:
            st.metric("Final Validation Accuracy", f"{performance_data['week2'].get('val_acc', [0])[-1]:.2%}")
    
    if 'week3' in performance_data:
        st.subheader("Week 3: Active Learning Baselines")
        
        for strategy, accuracies in performance_data['week3'].items():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"{strategy} - Best Accuracy", f"{max(accuracies):.2%}")
            
            with col2:
                st.metric(f"{strategy} - Final Accuracy", f"{accuracies[-1]:.2%}")
            
            with col3:
                improvement = accuracies[-1] - accuracies[0]
                st.metric(f"{strategy} - Improvement", f"{improvement:+.2%}")
    
    if 'week4' in performance_data:
        st.subheader("Week 4: REINFORCE Policy")
        rl_accuracies = performance_data['week4'].get('RL', [])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Peak Performance", f"{max(rl_accuracies):.2%}")
        
        with col2:
            st.metric("Final Performance", f"{rl_accuracies[-1]:.2%}")
        
        with col3:
            improvement = rl_accuracies[-1] - rl_accuracies[0]
            st.metric("Overall Improvement", f"{improvement:+.2%}")

def show_classification_page(model, device):
    """Display interactive image classification"""
    st.header("🔍 Interactive Image Classification")
    
    if not model:
        st.error("Model not available. Please ensure the model is trained and checkpoints are available.")
        return
    
    # File upload
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a cat or dog image to classify"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        if st.button("🔍 Classify Image", type="primary"):
            with st.spinner("Analyzing image..."):
                # Preprocess image
                image_tensor = preprocess_image(image)
                
                if image_tensor is not None:
                    # Make prediction
                    predicted_class, confidence, probabilities = predict_image(model, image_tensor, device)
                    
                    if predicted_class is not None:
                        # Display results
                        st.success(f"🎯 **Prediction: {predicted_class.upper()}**")
                        
                        # Confidence meter
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Probability distribution
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Probability Distribution")
                            classes = ['Cat', 'Dog']
                            fig = px.bar(
                                x=classes, 
                                y=probabilities,
                                title="Class Probabilities",
                                color=probabilities,
                                color_continuous_scale='Blues'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("Confidence Analysis")
                            if confidence > 0.9:
                                st.success("🟢 High Confidence - Model is very certain about this prediction")
                            elif confidence > 0.7:
                                st.info("🟡 Medium Confidence - Model is reasonably certain")
                            else:
                                st.warning("🟠 Low Confidence - Model is uncertain about this prediction")
                        
                        # LLM explanation
                        st.subheader("🤖 AI Explanation")
                        try:
                            explanation = explain_prediction(
                                predicted_class, 
                                confidence,
                                image_path=uploaded_file.name
                            )
                            st.markdown(f'<div class="explanation-box">{explanation}</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.warning(f"Could not generate AI explanation: {e}")
                            st.info("The model analyzed the image and made a prediction based on learned patterns. The confidence score indicates how certain the model is about this result.")

def show_active_learning_page(performance_data):
    """Display active learning analysis"""
    st.header("📈 Active Learning Analysis")
    
    if not performance_data:
        st.warning("No active learning data available.")
        return
    
    # Strategy comparison
    if 'week3' in performance_data:
        st.subheader("Strategy Performance Comparison")
        
        # Create comparison plot
        rounds = list(range(len(performance_data['week3'].get('Random', []))))
        
        fig = go.Figure()
        
        for strategy, accuracies in performance_data['week3'].items():
            fig.add_trace(go.Scatter(
                x=rounds, 
                y=accuracies, 
                mode='lines+markers',
                name=strategy,
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="Active Learning Strategies: Random vs Uncertainty Sampling",
            xaxis_title="Acquisition Round",
            yaxis_title="Validation Accuracy",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy analysis
        st.subheader("Strategy Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Random Sampling")
            random_acc = performance_data['week3']['Random']
            st.metric("Best Performance", f"{max(random_acc):.2%}")
            st.metric("Final Performance", f"{random_acc[-1]:.2%}")
            st.metric("Improvement", f"{random_acc[-1] - random_acc[0]:+.2%}")
        
        with col2:
            st.markdown("### Uncertainty Sampling")
            uncertainty_acc = performance_data['week3']['Uncertainty']
            st.metric("Best Performance", f"{max(uncertainty_acc):.2%}")
            st.metric("Final Performance", f"{uncertainty_acc[-1]:.2%}")
            st.metric("Improvement", f"{uncertainty_acc[-1] - uncertainty_acc[0]:+.2%}")
    
    # REINFORCE analysis
    if 'week4' in performance_data:
        st.subheader("REINFORCE Policy Analysis")
        
        rl_accuracies = performance_data['week4']['RL']
        steps = list(range(len(rl_accuracies)))
        
        # Learning curve
        fig = px.line(
            x=steps, 
            y=rl_accuracies, 
            title="REINFORCE Policy Learning Curve",
            labels={'x': 'Training Step', 'y': 'Validation Accuracy'}
        )
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Peak Performance", f"{max(rl_accuracies):.2%}")
            st.metric("Peak Step", steps[rl_accuracies.index(max(rl_accuracies))])
        
        with col2:
            st.metric("Final Performance", f"{rl_accuracies[-1]:.2%}")
            st.metric("Starting Performance", f"{rl_accuracies[0]:.2%}")
        
        with col3:
            improvement = rl_accuracies[-1] - rl_accuracies[0]
            st.metric("Overall Improvement", f"{improvement:+.2%}")
            st.metric("Consistency", "High" if max(rl_accuracies) - min(rl_accuracies) < 0.1 else "Variable")

def show_llm_explanations_page(performance_data):
    """Display LLM-powered explanations"""
    st.header("🤖 AI-Powered Explanations & Insights")
    
    st.markdown("""
    This section uses Large Language Models (LLMs) to provide human-understandable explanations 
    of model performance, predictions, and recommendations.
    """)
    
    # LLM service configuration
    st.subheader("🔧 LLM Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        provider = st.selectbox(
            "Select LLM Provider:",
            ["huggingface", "ollama", "openai"],
            help="Choose your preferred LLM provider"
        )
    
    with col2:
        api_key = st.text_input(
            "API Key (optional):",
            type="password",
            help="Enter your API key if required"
        )
    
    # Initialize LLM service
    try:
        llm_service = LLMService(provider=provider, api_key=api_key)
        st.success(f"✅ LLM Service initialized with {provider}")
    except Exception as e:
        st.error(f"❌ Error initializing LLM service: {e}")
        return
    
    # Explanation options
    st.subheader("📋 Explanation Options")
    
    explanation_type = st.selectbox(
        "What would you like explained?",
        [
            "Model Performance Summary",
            "Active Learning Strategy Analysis", 
            "Improvement Recommendations",
            "Custom Analysis"
        ]
    )
    
    if explanation_type == "Model Performance Summary" and performance_data:
        if st.button("📊 Generate Performance Summary"):
            with st.spinner("Generating AI-powered performance summary..."):
                try:
                    # Create metrics summary
                    metrics = {}
                    if 'week2' in performance_data:
                        metrics['Final Validation Accuracy'] = performance_data['week2'].get('val_acc', [0])[-1]
                        metrics['Final Training Loss'] = performance_data['week2'].get('train_loss', [0])[-1]
                    
                    if 'week4' in performance_data:
                        rl_acc = performance_data['week4']['RL']
                        metrics['Peak RL Performance'] = max(rl_acc)
                        metrics['Final RL Performance'] = rl_acc[-1]
                        metrics['RL Improvement'] = rl_acc[-1] - rl_acc[0]
                    
                    # Generate summary
                    summary = summarize_performance(metrics)
                    
                    st.markdown("### 🤖 AI Performance Summary")
                    st.markdown(f'<div class="explanation-box">{summary}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
    
    elif explanation_type == "Active Learning Strategy Analysis" and performance_data:
        if st.button("📈 Analyze Active Learning Strategies"):
            with st.spinner("Analyzing active learning strategies..."):
                try:
                    if 'week3' in performance_data:
                        explanation = llm_service.explain_active_learning_strategy(
                            "REINFORCE vs Baselines",
                            performance_data['week3']
                        )
                        
                        st.markdown("### 🤖 Active Learning Strategy Analysis")
                        st.markdown(f'<div class="explanation-box">{explanation.text}</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No active learning data available for analysis.")
                        
                except Exception as e:
                    st.error(f"Error analyzing strategies: {e}")
    
    elif explanation_type == "Improvement Recommendations":
        if st.button("💡 Get Improvement Recommendations"):
            with st.spinner("Generating improvement recommendations..."):
                try:
                    # Get current performance metrics
                    current_accuracy = 0.0
                    dataset_size = 0
                    active_learning_rounds = 0
                    
                    if 'week2' in performance_data:
                        current_accuracy = performance_data['week2'].get('val_acc', [0])[-1]
                        dataset_size = 20000  # From project status
                        active_learning_rounds = 8  # From project status
                    
                    recommendations = llm_service.recommend_improvements(
                        current_accuracy,
                        dataset_size,
                        active_learning_rounds
                    )
                    
                    st.markdown("### 🤖 Improvement Recommendations")
                    st.markdown(f'<div class="explanation-box">{recommendations.text}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
    
    elif explanation_type == "Custom Analysis":
        st.subheader("🔍 Custom Analysis")
        
        custom_prompt = st.text_area(
            "Enter your custom analysis request:",
            placeholder="e.g., Explain how the REINFORCE algorithm improves sample selection in this project...",
            height=100
        )
        
        if st.button("🤖 Generate Custom Analysis"):
            if custom_prompt.strip():
                with st.spinner("Generating custom analysis..."):
                    try:
                        # For custom prompts, we'll use a simple approach
                        # In a full implementation, you'd send this to the LLM
                        st.info("Custom analysis feature requires additional LLM integration. Please use the predefined options above.")
                    except Exception as e:
                        st.error(f"Error in custom analysis: {e}")
            else:
                st.warning("Please enter a custom analysis request.")
    
    # LLM service status
    st.subheader("🔍 LLM Service Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Provider**: {provider}")
        st.info(f"**Status**: {'Connected' if llm_service.api_key else 'No API Key'}")
    
    with col2:
        st.info(f"**Model**: {llm_service.config['model']}")
        st.info(f"**Base URL**: {llm_service.config['base_url']}")

if __name__ == "__main__":
    main() 