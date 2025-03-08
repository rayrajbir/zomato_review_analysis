from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from werkzeug.utils import secure_filename
from flask import send_from_directory
# Download necessary NLTK resources
try:
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    pass

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flashing messages

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def text_processing(text):
    """Process text by converting to lowercase, removing special characters, numbers, 
    and extra spaces, removing stopwords, and lemmatizing."""
    try:
        text = str(text).lower()  # Convert to lowercase
        text = re.sub(r'\W', ' ', text)  # Remove special characters
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        
        # Simple tokenization by splitting on spaces
        tokens = text.split()
        
        # Remove stopwords (if stopwords are available)
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [t for t in tokens if t not in stop_words]
        except:
            pass
        
        # Lemmatization (if WordNet is available)
        try:
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
        except:
            pass
        
        return ' '.join(tokens)
    except:
        return text  # Return original text if processing fails

def analyze_sentiment(text):
    """Analyze sentiment using NLTK's VADER sentiment analyzer."""
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(text)
        
        if sentiment_score['compound'] >= 0.05:
            return 'POSITIVE'
        elif sentiment_score['compound'] <= -0.05:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
    except:
        return 'NEUTRAL'  # Default to neutral if analysis fails

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the uploaded file
        try:
            # Determine the file type and read accordingly
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:  # Excel files
                df = pd.read_excel(file_path)
            
            # Check if the required column exists
            if 'review' not in df.columns:
                flash('File must contain a "review" column')
                return redirect(url_for('index'))
            
            # Process reviews
            df['Processed_Review'] = df['review'].astype(str).apply(text_processing)
            
            # Analyze sentiment
            df['Sentiment'] = df['Processed_Review'].apply(analyze_sentiment)
            
            # Save processed file
            processed_filename = 'processed_' + filename
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            
            if processed_filename.endswith('.csv'):
                df.to_csv(processed_path, index=False)
            else:
                df.to_excel(processed_path, index=False)
            
            # Calculate sentiment distribution
            sentiment_counts = df['Sentiment'].value_counts().to_dict()
            
            # Get sample reviews for each sentiment
            positive_samples = df[df['Sentiment'] == 'POSITIVE']['review'].head(3).tolist()
            negative_samples = df[df['Sentiment'] == 'NEGATIVE']['review'].head(3).tolist()
            neutral_samples = df[df['Sentiment'] == 'NEUTRAL']['review'].head(3).tolist()
            
            return render_template('results.html', 
                                  filename=processed_filename,
                                  sentiment_counts=sentiment_counts,
                                  positive_samples=positive_samples,
                                  negative_samples=negative_samples,
                                  neutral_samples=neutral_samples,
                                  total_reviews=len(df))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('File type not allowed. Please upload a CSV or Excel file.')
        return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    text = request.form.get('text', '')
    if not text:
        flash('Please enter text to analyze')
        return redirect(url_for('index'))
    
    processed_text = text_processing(text)
    sentiment = analyze_sentiment(processed_text)
    
    return render_template('text_result.html', 
                          original_text=text,
                          processed_text=processed_text,
                          sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)