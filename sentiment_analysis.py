import pandas as pd
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load your Excel file
sheets = pd.read_excel("Business_Review_Market_Analysis.xlsx", sheet_name=None)
print(sheets.keys())  # This will list all sheet names

df = sheets["Starbucks_review"]  # Use the correct name shown in your output
print(df.columns)  # Now this will show actual column names

# Clean text column
df['clean_text'] = df['text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# --- VADER Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()
df['compound'] = df['clean_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['sentiment'] = df['compound'].apply(lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral'))

# --- Keyword Frequency Extraction ---
stop_words = set(stopwords.words('english'))
words = ' '.join(df['clean_text']).split()
filtered_words = [word for word in words if word not in stop_words]
word_freq = Counter(filtered_words)
top_keywords = word_freq.most_common(50)
keywords_df = pd.DataFrame(top_keywords, columns=['keyword', 'count'])

# --- Export to Excel ---
with pd.ExcelWriter("Starbucks_Processed.xlsx") as writer:
    df.to_excel(writer, sheet_name="Reviews", index=False)
    keywords_df.to_excel(writer, sheet_name="Top_Keywords", index=False)

print("✅ Sentiment analysis complete and exported to 'Starbucks_Processed.xlsx'")
