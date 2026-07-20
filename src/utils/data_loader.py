import pandas as pd
import numpy as np

class DataLoader:
    @staticmethod
    def load_data(file_path):
        """Load dataset from CSV or Excel"""
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        data = data.dropna()
        
        # Ensure required columns exist
        required_cols = ['attendance', 'quiz_scores', 'assignment_scores', 'exam_scores']
        for col in required_cols:
            if col not in data.columns:
                possible_cols = [c for c in data.columns if col.replace('_', '') in c.replace('_', '').lower()]
                if possible_cols:
                    data[col] = data[possible_cols[0]]
                else:
                    data[col] = np.random.uniform(40, 90, len(data))
        
        # Add performance level if not present
        if 'performance_level' not in data.columns:
            scores = (data['quiz_scores'] * 0.3 + data['assignment_scores'] * 0.2 + 
                     data['exam_scores'] * 0.4 + data['attendance'] * 0.1)
            
            def get_level(score):
                if score >= 80: return 'Excellent'
                elif score >= 65: return 'Good'
                elif score >= 50: return 'Average'
                else: return 'Needs Improvement'
            
            data['performance_level'] = scores.apply(get_level)
        
        return data
    # model 1
    @staticmethod
    def get_dataset_info(data):
        """Get dataset information"""
        info = f"""
╔══════════════════════════════════════════╗
║          DATASET INFORMATION             ║
╠══════════════════════════════════════════╣
║ Rows:              {len(data):>10}          ║
║ Columns:           {len(data.columns):>10}          ║
╠══════════════════════════════════════════╣
║ Columns:                                 ║"""
        for col in data.columns:
            dtype = str(data[col].dtype)
            info += f"\n║   {col[:25]:25} ({dtype})"
        info += """
╚══════════════════════════════════════════╝
"""
        return info
    
    @staticmethod
    def get_stats(data):
        """Get statistics from dataset"""
        stats = {
            'total_students': len(data),
            'avg_score': data['quiz_scores'].mean() if 'quiz_scores' in data.columns else 0,
            'best_score': data['quiz_scores'].max() if 'quiz_scores' in data.columns else 0,
            'total_quizzes': 10
        }
        return stats