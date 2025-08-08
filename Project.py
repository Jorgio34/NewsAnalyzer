import requests
from bs4 import BeautifulSoup
import re
import time
import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import sqlite3
import threading
from dataclasses import dataclass
import logging
from typing import List, Dict, Any, Optional

# Try importing pandas, use fallback if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not installed. Some features will be limited.")
    print("Install with: pip install pandas")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Data structure for news articles"""
    title: str
    summary: str
    source: str
    timestamp: datetime
    url: str
    ticker: Optional[str] = None

class DataCollector:
    """Base class for data collectors"""
    def fetch_news(self, limit: int = 100) -> List[NewsArticle]:
        raise NotImplementedError("Subclasses must implement fetch_news")

class FinvizDataCollector(DataCollector):
    """Phase 1: Data Collection & Input from Finviz"""

    def __init__(self):
        self.base_url = "https://finviz.com/news.ashx"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_news(self, limit: int = 100) -> List[NewsArticle]:
        """Fetch news articles from Finviz"""
        try:
            print(f"Fetching news from Finviz (limit: {limit})...")
            response = self.session.get(self.base_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []

            # --- MODIFIED: More robust selector logic ---
            # Try to find the main news table using multiple, more general selectors
            news_table = soup.find('table', {'class': 'fullview-news-outer'})
            if not news_table:
                news_table = soup.find('table', {'class': 't-content'}) # A common alternative
            if not news_table:
                news_table = soup.find('table', id='news')

            if not news_table:
                logger.warning("Could not find a main news table on Finviz. Looking for any potential tables.")
                # Fallback to a broader search
                news_tables = soup.find_all('table')
                for table in news_tables:
                    if 'news' in str(table).lower() and table.find('a'):
                        news_table = table
                        break

            if not news_table:
                logger.warning("Still could not find a news table, returning sample data.")
                return self._get_sample_articles()

            rows = news_table.find_all('tr')
            print(f"Found {len(rows)} potential news rows")

            for i, row in enumerate(rows[:limit]):
                try:
                    cells = row.find_all('td')
                    if len(cells) < 2:
                        continue

                    # Look for time and news content
                    time_cell = None
                    news_cell = None

                    # Use a more heuristic approach to find the right cells
                    for cell in cells:
                        cell_text = cell.get_text(strip=True)
                        # Check if cell contains time pattern or date
                        if re.match(r'\d{1,2}:\d{2}[AP]M|\w{3}-\d{2}-\d{2}', cell_text) and not time_cell:
                            time_cell = cell
                        # Check if cell contains news link
                        if cell.find('a') and not news_cell:
                            news_cell = cell

                    if not (time_cell and news_cell):
                        continue

                    # Get timestamp
                    time_text = time_cell.get_text(strip=True)
                    timestamp = self._parse_time(time_text)

                    # Get news link and title
                    link = news_cell.find('a')
                    if not link:
                        continue

                    title = link.get_text(strip=True)
                    url = link.get('href', '')

                    if not title or len(title) < 10:
                        continue

                    # Extract source (usually in parentheses or after dash)
                    source = "Finviz"
                    source_patterns = [
                        r'\(([^)]+)\)',  # Text in parentheses
                        r'[-–—]\s*([A-Za-z\s]+)$',  # Text after dash at end
                        r'^([A-Za-z\s]+)[-–—]'  # Text before dash at start
                    ]

                    for pattern in source_patterns:
                        match = re.search(pattern, title)
                        if match:
                            potential_source = match.group(1).strip()
                            if len(potential_source) < 20 and ' ' not in potential_source:  # A reasonable heuristic for source names
                                source = potential_source
                            break

                    # Create article object
                    article = NewsArticle(
                        title=title,
                        summary=title,  # Finviz provides titles, not full summaries
                        source=source,
                        timestamp=timestamp,
                        url=url if url.startswith('http') else f"https://finviz.com{url}"
                    )

                    articles.append(article)

                except Exception as e:
                    logger.error(f"Error parsing article row {i}: {e}")
                    continue

            if not articles:
                logger.warning("No articles found, returning sample data")
                return self._get_sample_articles()

            logger.info(f"Successfully collected {len(articles)} articles from Finviz")
            return articles

        except Exception as e:
            logger.error(f"Error fetching news from Finviz: {e}")
            logger.info("Returning sample articles for testing")
            return self._get_sample_articles()

    def _get_sample_articles(self) -> List[NewsArticle]:
        """Return sample articles for testing when Finviz is unavailable"""
        sample_articles = [
            NewsArticle(
                title="Apple Stock Surges After Strong Earnings Beat Expectations",
                summary="Apple reports better than expected quarterly results",
                source="MarketWatch",
                timestamp=datetime.now() - timedelta(minutes=30),
                url="https://example.com/apple-earnings"
            ),
            NewsArticle(
                title="Fed Signals Potential Rate Cut Amid Economic Concerns",
                summary="Federal Reserve hints at monetary policy changes",
                source="Reuters",
                timestamp=datetime.now() - timedelta(hours=1),
                url="https://example.com/fed-rates"
            ),
            NewsArticle(
                title="Tesla Stock Plunges on Production Miss and Guidance Cut",
                summary="Tesla disappoints with lower production numbers",
                source="Bloomberg",
                timestamp=datetime.now() - timedelta(hours=2),
                url="https://example.com/tesla-production"
            ),
            NewsArticle(
                title="Market Volatility Continues as Investors Await Economic Data",
                summary="Markets remain uncertain ahead of key reports",
                source="CNBC",
                timestamp=datetime.now() - timedelta(hours=3),
                url="https://example.com/market-volatility"
            ),
            NewsArticle(
                title="Google Announces Major AI Investment and Research Expansion",
                summary="Google commits billions to artificial intelligence development",
                source="TechCrunch",
                timestamp=datetime.now() - timedelta(hours=4),
                url="https://example.com/google-ai"
            )
        ]

        print(f"Using {len(sample_articles)} sample articles for testing")
        return sample_articles

    def _parse_time(self, time_text: str) -> datetime:
        """Parse Finviz time format"""
        try:
            # Finviz uses formats like "Dec-18-24 08:30AM"
            if len(time_text) > 10:
                return datetime.strptime(time_text, "%b-%d-%y %I:%M%p")
            else:
                # If just time, assume today
                time_part = datetime.strptime(time_text, "%I:%M%p").time()
                return datetime.combine(datetime.now().date(), time_part)
        except (ValueError, IndexError):
            return datetime.now()

class TextPreprocessor:
    """Phase 2: Data Preprocessing & Cleaning"""

    def __init__(self):
        # Common stop words
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }

        # Negation words
        self.negations = {'not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nor'}

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Fix encoding issues
        text = text.encode('ascii', 'ignore').decode('ascii')

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Convert to lowercase
        text = text.lower()

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, remove stop words, and normalize"""
        # Clean text first
        text = self.clean_text(text)

        # Extract words (alphanumeric only)
        words = re.findall(r'\b[a-zA-Z]+\b', text)

        # Remove stop words
        words = [word for word in words if word not in self.stop_words]

        return words

    def quality_check(self, article: NewsArticle) -> bool:
        """Check if article meets quality criteria"""
        text = f"{article.title} {article.summary}"
        words = self.tokenize(text)

        # Article must have at least 3 meaningful words (flowchart specifies > 50 words but this is a reasonable minimum for a title/summary)
        if len(words) < 3:
            return False

        # Must contain some alphabetic characters
        if not re.search(r'[a-zA-Z]', article.title):
            return False

        return True

class LearningDictionary:
    """Phase 3: Learning Dictionary with sentiment scores and confidence levels"""

    def __init__(self):
        # Initialize with basic financial sentiment words
        self.word_scores: Dict[str, float] = {
            # Positive words
            'up': 0.5, 'rise': 0.5, 'gain': 0.6, 'profit': 0.7, 'growth': 0.6,
            'bull': 0.8, 'bullish': 0.8, 'rally': 0.7, 'surge': 0.8, 'jump': 0.6,
            'strong': 0.5, 'positive': 0.6, 'good': 0.4, 'beat': 0.6, 'exceed': 0.6,
            'outperform': 0.7, 'upgrade': 0.7, 'buy': 0.6, 'recommend': 0.5,

            # Negative words
            'down': -0.5, 'fall': -0.5, 'drop': -0.6, 'loss': -0.7, 'decline': -0.6,
            'bear': -0.8, 'bearish': -0.8, 'crash': -0.9, 'plunge': -0.8, 'sink': -0.6,
            'weak': -0.5, 'negative': -0.6, 'bad': -0.4, 'miss': -0.6, 'disappoint': -0.6,
            'underperform': -0.7, 'downgrade': -0.7, 'sell': -0.6, 'warning': -0.5,

            # Neutral but important
            'earnings': 0.0, 'revenue': 0.0, 'sales': 0.0, 'report': 0.0
        }

        self.word_counts: Dict[str, int] = defaultdict(int)
        self.confidence_levels: Dict[str, float] = defaultdict(float)

        # Initialize confidence levels
        for word in self.word_scores:
            self.confidence_levels[word] = 0.8

    def get_sentiment_score(self, word: str) -> tuple[float, float]:
        """Get sentiment score and confidence for a word"""
        score = self.word_scores.get(word, 0.0)
        confidence = self.confidence_levels.get(word, 0.1)
        return score, confidence

    def update_word_score(self, word: str, context_sentiment: float, learning_rate: float = 0.1):
        """Update word sentiment score based on context"""
        current_score = self.word_scores.get(word, 0.0)

        # Update score using exponential moving average
        new_score = current_score + learning_rate * (context_sentiment - current_score)

        self.word_scores[word] = max(-1.0, min(1.0, new_score))  # Clamp to [-1, 1]

        # Update confidence
        self.word_counts[word] += 1
        self.confidence_levels[word] = min(0.9, 0.1 + 0.1 * (self.word_counts[word] ** 0.5))

    def learn_from_patterns(self, articles: List[NewsArticle], preprocessor: 'TextPreprocessor'):
        """Learn sentiment patterns from articles"""
        for article in articles:
            words = preprocessor.tokenize(article.title + " " + article.summary)

            # Simple heuristic: if article contains strong positive/negative words,
            # update scores for co-occurring words
            strong_positive = sum(1 for w in words if self.word_scores.get(w, 0) > 0.6)
            strong_negative = sum(1 for w in words if self.word_scores.get(w, 0) < -0.6)

            if strong_positive > strong_negative:
                context_sentiment = 0.2
            elif strong_negative > strong_positive:
                context_sentiment = -0.2
            else:
                context_sentiment = 0.0

            # Update scores for neutral/unknown words
            for word in words:
                if abs(self.word_scores.get(word, 0)) < 0.3:
                    self.update_word_score(word, context_sentiment)

class SentimentAnalyzer:
    """Phase 3: Main Sentiment Analysis Engine"""

    def __init__(self, dictionary: LearningDictionary, preprocessor: TextPreprocessor):
        self.dictionary = dictionary
        self.preprocessor = preprocessor

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        words = self.preprocessor.tokenize(text)

        if not words:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'word_count': 0,
                'positive_words': [],
                'negative_words': []
            }

        total_score = 0.0
        total_confidence = 0.0
        positive_words = []
        negative_words = []

        # Handle negations
        negated = False

        for word in words:
            if word in self.preprocessor.negations:
                negated = True
                continue

            score, confidence = self.dictionary.get_sentiment_score(word)

            # Apply negation
            if negated:
                score = -score
                negated = False

            # Weight by confidence
            weighted_score = score * confidence
            total_score += weighted_score
            total_confidence += confidence

            # Track positive/negative words
            if score > 0.3:
                positive_words.append(word)
            elif score < -0.3:
                negative_words.append(word)

        # Calculate final sentiment
        if total_confidence > 0:
            final_sentiment = total_score / len(words)
            avg_confidence = total_confidence / len(words)
        else:
            final_sentiment = 0.0
            avg_confidence = 0.0

        return {
            'sentiment_score': round(final_sentiment, 3),
            'confidence': round(avg_confidence, 3),
            'word_count': len(words),
            'positive_words': positive_words,
            'negative_words': negative_words
        }

class DataManager:
    """Phase 4: Data Storage & Management using SQLite as a stand-in for PostgreSQL/Redis"""

    def __init__(self, db_path="sentiment_analysis.db"):
        self.db_path = db_path
        self.init_database()
        self.cache = {}

    def init_database(self):
        """Initialize SQLite database for articles and dictionary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Articles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                summary TEXT,
                source TEXT,
                url TEXT,
                timestamp DATETIME,
                sentiment_score REAL,
                confidence REAL,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Dictionary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dictionary (
                word TEXT PRIMARY KEY,
                sentiment_score REAL,
                confidence REAL,
                count INTEGER,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def store_article(self, article: NewsArticle, sentiment_result: Dict[str, Any]):
        """Store article and sentiment analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO articles (title, summary, source, url, timestamp, sentiment_score, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            article.title,
            article.summary,
            article.source,
            article.url,
            article.timestamp,
            sentiment_result['sentiment_score'],
            sentiment_result['confidence']
        ))

        conn.commit()
        conn.close()

    def get_recent_articles(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent articles from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM articles
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours))

        # Convert to list of dictionaries
        columns = [description[0] for description in cursor.description]
        articles = []
        for row in cursor.fetchall():
            articles.append(dict(zip(columns, row)))

        conn.close()
        return articles

    def get_sentiment_trends(self) -> Dict[str, Any]:
        """Calculate sentiment trends"""
        if not PANDAS_AVAILABLE:
            # Fallback without pandas
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT
                    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(*) as article_count
                FROM articles
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY strftime('%Y-%m-%d %H:00:00', timestamp)
                ORDER BY hour
            ''')

            trends = []
            total_sentiment = 0
            total_articles = 0

            for row in cursor.fetchall():
                hour, avg_sentiment, count = row
                trend_data = {
                    'hour': hour,
                    'avg_sentiment': float(avg_sentiment) if avg_sentiment else 0.0,
                    'article_count': int(count)
                }
                trends.append(trend_data)
                total_sentiment += trend_data['avg_sentiment'] * trend_data['article_count']
                total_articles += trend_data['article_count']

            conn.close()

            overall_sentiment = total_sentiment / total_articles if total_articles > 0 else 0.0

            return {
                'hourly_trends': trends,
                'overall_sentiment': overall_sentiment,
                'total_articles': total_articles
            }

        else:
            # Use pandas if available
            conn = sqlite3.connect(self.db_path)

            df = pd.read_sql_query('''
                SELECT
                    strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(*) as article_count
                FROM articles
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY strftime('%Y-%m-%d %H:00:00', timestamp)
                ORDER BY hour
            ''', conn)

            conn.close()

            return {
                'hourly_trends': df.to_dict('records'),
                'overall_sentiment': df['avg_sentiment'].mean() if not df.empty else 0.0,
                'total_articles': df['article_count'].sum() if not df.empty else 0
            }

class RealTimeSentimentSystem:
    """Main system orchestrator"""

    def __init__(self):
        self.collector = FinvizDataCollector()
        self.preprocessor = TextPreprocessor()
        self.dictionary = LearningDictionary()
        self.analyzer = SentimentAnalyzer(self.dictionary, self.preprocessor)
        self.data_manager = DataManager()

        self.running = False
        self.processing_thread: Optional[threading.Thread] = None

    def process_batch(self):
        """Process a batch of articles from data collectors"""
        logger.info("Starting batch processing...")

        # Phase 1: Collect data
        articles = self.collector.fetch_news(limit=50)

        if not articles:
            logger.warning("No articles collected")
            return {'processed': 0, 'errors': 0}

        processed = 0
        errors = 0

        for article in articles:
            try:
                # Phase 2: Quality check and preprocessing
                if not self.preprocessor.quality_check(article):
                    continue

                # Phase 3: Analyze sentiment
                text = f"{article.title} {article.summary}"
                sentiment_result = self.analyzer.analyze_sentiment(text)

                # Phase 4: Store results
                self.data_manager.store_article(article, sentiment_result)

                processed += 1

            except Exception as e:
                logger.error(f"Error processing article: {e}")
                errors += 1

        # Phase 3: Learn from new data
        if processed > 0:
            self.dictionary.learn_from_patterns(articles, self.preprocessor)
            logger.info(f"Dictionary updated with {len(self.dictionary.word_scores)} words")

        logger.info(f"Batch complete: {processed} processed, {errors} errors")
        return {'processed': processed, 'errors': errors}

    def start_real_time_processing(self, interval_minutes: int = 15):
        """Start real-time processing loop"""
        self.running = True

        def processing_loop():
            while self.running:
                try:
                    self.process_batch()
                    time.sleep(interval_minutes * 60)  # Convert to seconds
                except Exception as e:
                    logger.error(f"Error in processing loop: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying

        self.processing_thread = threading.Thread(target=processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        logger.info(f"Real-time processing started (interval: {interval_minutes} minutes)")

    def stop_real_time_processing(self):
        """Stop real-time processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Real-time processing stopped")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        trends = self.data_manager.get_sentiment_trends()

        return {
            'system_running': self.running,
            'dictionary_size': len(self.dictionary.word_scores),
            'recent_articles': len(self.data_manager.get_recent_articles(24)),
            'current_sentiment': trends['overall_sentiment'],
            'trends': trends
        }

    def analyze_text_live(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of custom text"""
        return self.analyzer.analyze_sentiment(text)

# Demo usage
if __name__ == "__main__":
    # Initialize system
    system = RealTimeSentimentSystem()

    print("=== Real-Time News Sentiment Analyzer (Finviz) ===\n")

    # Process initial batch
    print("Processing initial batch...")
    result = system.process_batch()
    print(f"Initial batch: {result['processed']} articles processed\n")

    # Show system status
    status = system.get_system_status()
    print(f"System Status:")
    print(f"- Dictionary size: {status['dictionary_size']} words")
    print(f"- Recent articles: {status['recent_articles']}")
    print(f"- Overall sentiment: {status['current_sentiment']:.3f}")

    # Demo text analysis
    print(f"\n=== Live Text Analysis Demo ===")
    test_texts = [
        "Apple stock surges after strong earnings beat expectations",
        "Market crash fears grow as economic indicators disappoint",
        "Tesla announces new factory expansion plans"
    ]

    for text in test_texts:
        result = system.analyze_text_live(text)
        print(f"\nText: '{text}'")
        print(f"Sentiment: {result['sentiment_score']:.3f} (confidence: {result['confidence']:.3f})")
        print(f"Positive words: {result['positive_words']}")
        print(f"Negative words: {result['negative_words']}")

    # Start real-time processing
    print(f"\n=== Starting Real-Time Processing ===")
    system.start_real_time_processing(interval_minutes=5)  # Process every 5 minutes

    print("System is now running in real-time mode.")
    print("Press Ctrl+C to stop...")

    try:
        while True:
            time.sleep(10)
            status = system.get_system_status()
            print(f"Status: {status['recent_articles']} articles, sentiment: {status['current_sentiment']:.3f}")
    except KeyboardInterrupt:
        print("\nStopping system...")
        system.stop_real_time_processing()
        print("System stopped.")