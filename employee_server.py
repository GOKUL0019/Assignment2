from flask import Flask, render_template, request, redirect, url_for
import os
import pickle
import numpy as np
from werkzeug.utils import secure_filename
from transformers import pipeline
from datetime import datetime  
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import ast
import radon
from radon.complexity import cc_visit

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize AI models (load only once)
try:
    nlp_summarizer = pipeline("summarization")
    sentiment_analyzer = pipeline("sentiment-analysis")
except:
    print("NLP models could not be loaded. Text analysis will be limited.")
    nlp_summarizer = None
    sentiment_analyzer = None

try:
    cv_model = ResNet50(weights='imagenet', include_top=False)
except:
    print("CV model could not be loaded. Image analysis will be limited.")
    cv_model = None

def analyze_text_content(folder_path):
    """Analyze text documents using NLP"""
    text_content = ""
    text_files = []
    
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.txt', '.md', '.docx', '.pdf')):
                try:
                    with open(os.path.join(root, filename), 'r', encoding='utf-8') as f:
                        content = f.read()
                        text_content += content + " "
                        text_files.append({
                            'filename': filename,
                            'length': len(content.split()),
                            'content_sample': content[:100] + '...' if len(content) > 100 else content
                        })
                except:
                    continue
    
    analysis_result = {
        'text_files': text_files,
        'total_files': len(text_files),
        'total_words': len(text_content.split()),
        'unique_words': 0,
        'sentiment': 'neutral',
        'sentiment_score': 0,
        'summary_quality': 0
    }
    
    if text_content and nlp_summarizer and sentiment_analyzer:
        try:
            # Sentiment analysis (first 512 chars)
            sentiment = sentiment_analyzer(text_content[:512])[0]
            analysis_result['sentiment'] = sentiment['label']
            analysis_result['sentiment_score'] = sentiment['score']
            
            # Summary quality (first 1024 chars)
            summary = nlp_summarizer(text_content[:1024], max_length=130, min_length=30, do_sample=False)
            analysis_result['summary'] = summary[0]['summary_text']
            analysis_result['summary_quality'] = min(1.0, len(summary[0]['summary_text'])/len(text_content))
        except Exception as e:
            print(f"Text analysis error: {e}")
    
    if text_content:
        vectorizer = TfidfVectorizer()
        try:
            X = vectorizer.fit_transform([text_content])
            analysis_result['unique_words'] = len(vectorizer.vocabulary_)
        except:
            pass
    
    return analysis_result

def analyze_designs(folder_path):
    """Analyze design files using computer vision"""
    design_files = []
    features = []
    
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                design_files.append({
                    'filename': filename,
                    'path': os.path.join(root, filename)
                })
    
    analysis_result = {
        'design_files': design_files,
        'total_designs': len(design_files),
        'feature_diversity': 0,
        'avg_activation': 0
    }
    
    if cv_model and design_files:
        try:
            for design in design_files[:10]:  # Limit to 10 files for performance
                img = image.load_img(design['path'], target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features.append(cv_model.predict(x))
            
            if features:
                features_array = np.concatenate(features)
                analysis_result['feature_diversity'] = float(np.mean(np.std(features_array, axis=0)))
                analysis_result['avg_activation'] = float(np.mean(features_array))
        except Exception as e:
            print(f"Image analysis error: {e}")
    
    return analysis_result

def analyze_code_quality(folder_path):
    """Analyze code quality and complexity"""
    code_files = []
    total_lines = 0
    total_functions = 0
    total_complexity = 0
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        
                        # Basic metrics
                        file_metrics = {
                            'filename': file,
                            'lines': len(lines),
                            'functions': 0,
                            'avg_complexity': 0
                        }
                        
                        # Complexity analysis
                        try:
                            blocks = cc_visit(content)
                            file_metrics['functions'] = len(blocks)
                            if blocks:
                                file_metrics['avg_complexity'] = sum(b.complexity for b in blocks) / len(blocks)
                        except:
                            pass
                        
                        code_files.append(file_metrics)
                        total_lines += file_metrics['lines']
                        total_functions += file_metrics['functions']
                        total_complexity += file_metrics['avg_complexity'] * file_metrics['functions']
                except:
                    continue
    
    analysis_result = {
        'code_files': code_files,
        'total_files': len(code_files),
        'total_lines': total_lines,
        'total_functions': total_functions,
        'avg_complexity': total_complexity / total_functions if total_functions > 0 else 0
    }
    
    return analysis_result

def calculate_progress(analysis_result, project_type):
    """Calculate progress based on analysis results and project type"""
    base_progress = 0
    
    if project_type in ['documentation', 'report', 'writing']:
        # Text-based projects
        base_progress = min(50, analysis_result.get('total_files', 0) * 5)
        base_progress += min(30, analysis_result.get('total_words', 0) / 100)
        base_progress += min(20, analysis_result.get('unique_words', 0) / 50)
        
        # Boost for positive sentiment
        if analysis_result.get('sentiment') == 'POSITIVE':
            base_progress = min(100, base_progress * 1.1)
    
    elif project_type in ['design', 'artwork', 'graphics']:
        # Design projects
        base_progress = min(70, analysis_result.get('total_designs', 0) * 7)
        base_progress = min(100, base_progress + (analysis_result.get('feature_diversity', 0) * 10))
    
    elif project_type in ['software', 'coding', 'development']:
        # Code projects
        base_progress = min(60, analysis_result.get('total_files', 0) * 4)
        base_progress += min(20, analysis_result.get('total_lines', 0) / 100)
        base_progress += min(20, analysis_result.get('total_functions', 0) * 2)
        
        # Penalize for high complexity
        complexity = analysis_result.get('avg_complexity', 0)
        if complexity > 5:
            base_progress = max(0, base_progress - (complexity - 5) * 3)
    
    else:
        # Generic project - just count files
        base_progress = min(100, analysis_result.get('total_files', 0) * 3)
    
    return min(100, max(0, round(base_progress, 2)))

def analyze_project(folder_path, project_type):
    """Main analysis function that routes to specific analyzers"""
    analysis_result = {
        'project_type': project_type,
        'total_files': len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    }
    
    # Run type-specific analysis
    if project_type in ['documentation', 'report', 'writing']:
        analysis_result.update(analyze_text_content(folder_path))
    elif project_type in ['design', 'artwork', 'graphics']:
        analysis_result.update(analyze_designs(folder_path))
    elif project_type in ['software', 'coding', 'development']:
        analysis_result.update(analyze_code_quality(folder_path))
    
    # Calculate progress
    progress = calculate_progress(analysis_result, project_type)
    
    return progress, analysis_result

@app.route('/')
def home():
    return redirect(url_for('manager'))

@app.route('/manager', methods=['GET', 'POST'])
def manager():
    if request.method == 'POST':
        employee = request.form['employee']
        task = request.form['task']
        project_type = request.form.get('project_type', 'general')
        
        if os.path.exists('manager_data.pkl'):
            with open('manager_data.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            data = {}
        
        data[employee] = {
            'task': task,
            'project_type': project_type,
            'progress': 0,
            'last_updated': None,
            'analysis': None
        }
        
        with open('manager_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        return redirect(url_for('manager'))

    try:
        with open('manager_data.pkl', 'rb') as f:
            data = pickle.load(f)
    except:
        data = {}

    return render_template('manager.html', data=data)

@app.route('/employee', methods=['GET', 'POST'])
def employee():
    if request.method == 'POST':
        employee = request.form.get('employee')
        project_type = request.form.get('project_type')
        files = request.files.getlist("files")

        if not employee or not project_type or not files:
            return "Missing employee name, project type or files", 400

        save_path = os.path.join(app.config['UPLOAD_FOLDER'], employee)
        os.makedirs(save_path, exist_ok=True)

        uploaded_files = []
        for file in files:
            if file.filename == '':
                continue
            filename = secure_filename(file.filename)
            filepath = os.path.join(save_path, filename)
            file.save(filepath)
            uploaded_files.append({
                'name': filename,
                'size': os.path.getsize(filepath)
            })

        progress, analysis = analyze_project(save_path, project_type)

        if os.path.exists('manager_data.pkl'):
            with open('manager_data.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            data = {}

        if employee not in data:
            data[employee] = {
                'task': 'Autocreated from upload',
                'project_type': project_type
            }
        
        data[employee].update({
            'progress': progress,
            'last_updated': str(datetime.now()),
            'analysis': analysis,
            'uploaded_files': uploaded_files
        })

        with open('manager_data.pkl', 'wb') as f:
            pickle.dump(data, f)

        return render_template('result.html', 
                             employee=employee, 
                             progress=progress,
                             analysis=analysis,
                             files=uploaded_files,
                             project_type=project_type)

    try:
        with open('manager_data.pkl', 'rb') as f:
            data = pickle.load(f)
    except:
        data = {}

    return render_template('employee.html', employees=data.keys())

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)