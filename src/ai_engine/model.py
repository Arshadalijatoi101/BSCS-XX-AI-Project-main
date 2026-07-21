import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
class AITutorModel:
    model = None
    feature_cols = []
    label_encoder = None
    
    @staticmethod
    def train(data, model_type="Random Forest", test_size=0.2):
        """Train the AI model"""
        
        # Prepare features and target
        feature_cols = ['attendance', 'quiz_scores', 'assignment_scores', 'exam_scores', 'hours_studied']
        target_col = 'performance_level'
        
        # Check which columns exist
        available_cols = [col for col in feature_cols if col in data.columns]
        if not available_cols:
            available_cols = [col for col in data.columns if col != target_col and data[col].dtype in ['int64', 'float64']]
            available_cols = available_cols[:5]
        
        X = data[available_cols]
        y = data[target_col] if target_col in data.columns else data.iloc[:, -1]
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        AITutorModel.label_encoder = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
        
        # Select model
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=10, random_state=42)
        elif model_type == "SVM":
            model = SVC(kernel='rbf', probability=True, random_state=42)
        elif model_type == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model_type': model_type,
            'test_size': test_size,
            'feature_columns': available_cols
        }
        
        AITutorModel.model = model
        AITutorModel.feature_cols = available_cols
        
        return model, metrics, X_test, y_test
    
    @staticmethod
    def predict(model, student_data):
        """Predict performance level for a student"""
        
        if AITutorModel.feature_cols is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare input
        features = [student_data.get(col, 0) for col in AITutorModel.feature_cols]
        X_input = np.array([features]).reshape(1, -1)
        
        # Predict
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]
        
        # Decode prediction
        if AITutorModel.label_encoder:
            prediction_label = AITutorModel.label_encoder.inverse_transform([prediction])[0]
        else:
            prediction_label = f"Level {prediction + 1}"
        
        # Generate recommendation
        recommendation = AITutorModel.generate_recommendation(prediction_label, student_data)
        # Create probabilities dict
        prob_dict = {}
        if AITutorModel.label_encoder:
            for i, cls in enumerate(AITutorModel.label_encoder.classes_):
                prob_dict[cls] = probabilities[i]
        else:
            prob_dict = {f"Level {i+1}": p for i, p in enumerate(probabilities)}
        
        return prediction_label, recommendation, prob_dict
    
    @staticmethod
    def generate_recommendation(performance_level, student_data):
        """Generate personalized recommendation"""
        
        recommendations = {
            'Excellent': {
                'message': "🎉 Outstanding performance! You're excelling in your studies.",
                'suggestions': [
                    "Continue your current study habits",
                    "Help classmates who are struggling",
                    "Explore advanced topics in your field"
                ]
            },
            'Good': {
                'message': "👍 Good performance! You're on the right track.",
                'suggestions': [
                    "Focus on areas where you scored lower",
                    "Increase study time slightly",
                    "Practice more problem-solving"
                ]
            },
            'Average': {
                'message': "📊 Average performance. There's room for improvement.",
                'suggestions': [
                    f"Focus on attendance ({student_data.get('attendance', 0)}%)",
                    f"Improve quiz scores ({student_data.get('quiz_scores', 0)}%)",
                    "Create a consistent study schedule"
                ]
            },
            'Needs Improvement': {
                'message': "⚠️ Needs improvement. Don't worry, let's work on it together!",
                'suggestions': [
                    "Create a daily study plan",
                    "Focus on understanding concepts first",
                    "Take regular breaks to avoid burnout"
                ]
            }
        }
        
        return recommendations.get(performance_level, recommendations['Average'])
    
    @staticmethod
    def display_metrics(metrics):
        """Format metrics for display"""
        return f"""
╔══════════════════════════════════════════╗
║          MODEL PERFORMANCE METRICS       ║
╠══════════════════════════════════════════╣
║ Model:           {metrics['model_type']:<20} ║
║ Test Size:       {metrics['test_size']:.0%} ║
╠══════════════════════════════════════════╣
║ Accuracy:        {metrics['accuracy']:.2%} ║
║ Precision:       {metrics['precision']:.2%} ║
║ Recall:          {metrics['recall']:.2%} ║
║ F1 Score:        {metrics['f1_score']:.2%} ║
╠══════════════════════════════════════════╣
║ Features Used: {len(metrics['feature_columns'])} columns  ║
╚══════════════════════════════════════════╝
"""
    
    @staticmethod
    def display_prediction(student_data, prediction, recommendation, probabilities):
        """Format prediction for display"""
        
        result = f"""
╔══════════════════════════════════════════════════╗
║              STUDENT PERFORMANCE REPORT          ║
╠══════════════════════════════════════════════════╣
║ 📊 Input Data:                                    ║
"""
        for key, value in student_data.items():
            result += f"║   {key.replace('_', ' ').title():20}: {value:>6.1f} ║\n"
        
        result += f"""╠══════════════════════════════════════════════════╣
║ 🎯 Prediction: {prediction:<30} ║
╠══════════════════════════════════════════════════╣
║ 📈 Confidence:                                    ║
"""
        for level, prob in probabilities.items():
            result += f"║   {level:20}: {prob:>6.1%} ║\n"
        
        result += f"""╠══════════════════════════════════════════════════╣
║ 💡 Recommendation:                                 ║
║   {recommendation['message']} ║
║                                                    ║
"""
        for suggestion in recommendation['suggestions'][:3]:
            result += f"║   • {suggestion:<47} ║\n"
        
        result += """╚══════════════════════════════════════════════════╝
"""
        return result
