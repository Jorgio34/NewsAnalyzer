# Finviz News Sentiment Analyzer

This Python-based application performs real-time sentiment analysis on financial news headlines scraped from Finviz.com to provide a quick overview of market sentiment.

## Features

* **Real-time News Scraping:** Fetches the latest headlines, sources, and links from Finviz.com.
* **Dynamic Learning Dictionary:** Self-updating sentiment scores that adapt and learn from incoming news data.
* **Context-Aware Analysis:** Handles negation and word relationships for more accurate sentiment detection.
* **Comprehensive Data Storage:** SQLite database for persistent storage of articles, sentiment scores, and dictionary state.
* **Sentiment Trend Analysis:** Tracks sentiment changes over time with hourly breakdowns.
* **Live Demo:** Includes a function to run real-time text analysis on custom phrases.
* **System Status:** Provides detailed performance metrics including dictionary size, processing statistics, and overall market sentiment.

## Getting Started

### Prerequisites

To run this project, you'll need Python and the following libraries:

* `requests`
* `beautifulsoup4`
* `sqlite3` (included with Python)
* `pandas` (optional - provides enhanced analytics)

You can install the required packages using `pip`:

```bash
pip install requests beautifulsoup4 pandas
```

### Running the Application

Simply run the main Python script from your terminal:

```bash
python real_time_sentiment.py
```

The script will:
1. Process an initial batch of news articles from Finviz
2. Display system status and dictionary statistics
3. Run a live sentiment analysis demo on sample financial headlines
4. Enter real-time processing mode (fetches new articles every 15 minutes by default)
5. Show periodic status updates

You can stop the real-time loop by pressing `Ctrl+C`.

### Configuration

The system can be configured for different processing intervals:

```python
# Start real-time processing every 5 minutes
system.start_real_time_processing(interval_minutes=5)

# Process a custom batch size
articles = system.collector.fetch_news(limit=100)
```

## Code Structure

### Core Components

* **`FinvizDataCollector`**: Handles news scraping from Finviz.com with robust HTML parsing and fallback mechanisms for reliable data collection.

* **`TextPreprocessor`**: Manages text cleaning, tokenization, stop word removal, and quality filtering to prepare articles for analysis.

* **`LearningDictionary`**: The heart of the system - a self-updating sentiment dictionary that learns from context and adjusts word scores dynamically based on usage patterns.

* **`SentimentAnalyzer`**: Performs sentiment analysis using the learning dictionary, handling negation, confidence weighting, and detailed sentiment breakdowns.

* **`DataManager`**: Manages SQLite database operations, article storage, sentiment trend calculation, and data retrieval for analysis.

* **`RealTimeSentimentSystem`**: Main orchestrator class that coordinates all components, manages real-time processing, and provides system status monitoring.

### Data Classes

* **`NewsArticle`**: Data structure containing article title, summary, source, timestamp, URL, and optional ticker information.

## Sample Output

```
=== Real-Time News Sentiment Analyzer (Finviz) ===

Processing initial batch...
Fetching news from Finviz (limit: 50)...
Successfully collected 15 articles from Finviz
Dictionary updated with 127 words
Initial batch: 15 articles processed

System Status:
- Dictionary size: 127 words
- Recent articles: 15
- Overall sentiment: 0.124

=== Live Text Analysis Demo ===

Text: 'Apple stock surges after strong earnings beat expectations'
Sentiment: 0.456 (confidence: 0.623)
Positive words: ['surges', 'strong', 'beat']
Negative words: []

=== Starting Real-Time Processing ===
System is now running in real-time mode.
Press Ctrl+C to stop...
```

## Database Schema

The system automatically creates two SQLite tables:

**Articles Table:**
- Stores processed news articles with sentiment scores, confidence levels, and metadata
- Tracks processing timestamps for trend analysis

**Dictionary Table:**
- Maintains the learning dictionary with word sentiment scores
- Records confidence levels and usage counts for each word
- Updates automatically as the system learns from new articles

## Key Advantages Over Static Systems

### Dynamic Learning
Unlike traditional sentiment analyzers that use fixed dictionaries, this system:
- **Adapts to Financial Language:** Learns domain-specific terminology and sentiment patterns
- **Context Awareness:** Understands how words behave differently in financial contexts
- **Temporal Learning:** Adjusts to changing market language over time

### Real-Time Processing
- **Continuous Updates:** Processes new articles every 15 minutes (configurable)
- **Live Analysis:** Provides immediate sentiment analysis for any text input
- **Trend Tracking:** Monitors sentiment changes over time periods

### Robust Architecture
- **Error Handling:** Comprehensive fallback mechanisms when news sources are unavailable
- **Quality Control:** Filters low-quality articles before processing
- **Performance Monitoring:** Tracks system performance and processing statistics

## Research Applications

This system is designed for academic research in:
- **Financial Market Sentiment:** Real-time tracking of news sentiment vs. market performance
- **Natural Language Processing:** Dynamic dictionary learning and adaptation
- **Time Series Analysis:** Sentiment trend analysis and pattern recognition
- **Information Systems:** Real-time data processing and storage architectures

## Academic Project Context

**College of IST Faculty Project - Raahemifar (Analyzer)**

- **Project Duration:** May 5, 2025 - August 10, 2025
- **Minimum Hours:** 250 total hours (25 hours/week minimum)
- **Work Mode:** Remote with weekly meetings required
- **Academic Credit:** Qualifies for IST 495 credit
- **Research Focus:** Real-time dictionary-based sentiment analysis with machine learning capabilities

## Performance Characteristics

- **Processing Speed:** ~50 articles processed in 2-5 seconds
- **Memory Efficient:** Lightweight dictionary storage and SQLite database
- **Scalable:** Handles hundreds of articles per processing cycle
- **Reliable:** Robust error handling and fallback mechanisms

## Future Enhancements

The current system provides a solid foundation for advanced research including:
- Integration with multiple news sources
- Advanced machine learning algorithms for sentiment prediction
- Real-time market correlation analysis
- Multi-language support
- API development for external system integration

## License

This project is developed as part of an academic research initiative at the College of IST under Professor Raahemifar's supervision.
