# FinApp

**A Financial Market Analysis and Portfolio Management Application**

FinApp is a modern web application built with Python and Reflex that provides sophisticated tools for financial market analysis, portfolio optimization, and investment strategy development.

## 🚀 Features

### Market Analysis & Data
- **Real-time Market Data**: Integration with Yahoo Finance, BlackRock, TipRanks, and Zacks
- **Historical Analysis**: Comprehensive historical data retrieval and analysis
- **Multi-Provider Support**: Unified interface across multiple financial data sources
- **Intelligent Caching**: Automatic data caching with date-based organization

### Portfolio Management
- **Portfolio Optimization**: Mathematical optimization using modern portfolio theory (cvxpy, riskfolio-lib)
- **Backtesting Engine**: Historical performance testing of investment strategies
- **Risk Analysis**: Advanced risk metrics and factor analysis
- **Performance Visualization**: Interactive charts and plots using matplotlib and reflex-pyplot

### Screening & Comparison
- **Stock/ETF Screening**: Filter and search financial instruments by custom criteria
- **Side-by-Side Comparison**: Detailed comparison of multiple assets
- **Fundamental Analysis**: Access to company financials and market data
- **Technical Indicators**: Built-in technical analysis tools

### System Management
- **Cache Management Interface**: Web-based cache and log management
- **Structured Logging**: Comprehensive CSV-based logging system
- **Data Storage**: Organized date-based storage with automatic cleanup
- **Workflow Processing**: Background processing using LlamaIndex workflows

## 🏗️ Architecture

### Technology Stack
- **Frontend**: Reflex (Python-to-React) with TailwindV4 styling
- **Backend**: Python 3.10+ with async/await support
- **Data Processing**: pandas, numpy, DuckDB for analytics
- **Visualization**: matplotlib, reflex-pyplot for interactive charts
- **Storage**: Date-based file organization (JSON, Parquet, CSV)
- **Testing**: pytest with 196+ comprehensive unit tests

### Key Components
- **Data Providers**: Modular system for financial API integration
- **Caching Layer**: Automatic response caching with configurable TTL
- **Logging System**: Structured CSV logging with context detection
- **Storage Utilities**: Date-based folder management and cleanup
- **Web Interface**: Multi-page application with reactive state management

## 🛠️ Development

### Quick Start
```bash
# Install dependencies
make install

# Build the application
make build

# Run the application
make run
```

### Development Workflow
```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Full development cycle
make all
```

### Project Structure
```
finapp/
├── app/                   # Main application code
│   ├── flows/             # LlamaIndex workflows
│   ├── lib/               # Core utilities (storage, logging, settings)
│   ├── models/            # Data providers and caching
│   ├── pages/             # Web application pages
│   └── templates/         # Layout templates
├── data/                  # Date-organized cache storage
│   └── YYYYMMDD/          # Daily folders with logs and cached data
├── tests/                 # Comprehensive test suite (196 tests)
├── temp/                  # Temporary workspace for experiments
└── assets/                # Static assets and content
```

## 📊 Data Management

### Storage Organization
- **Date-Based Folders**: All data organized by date (YYYYMMDD format)
- **Automatic Caching**: API responses cached as JSON/Parquet files
- **Log Management**: Daily CSV logs with structured data
- **Cache Cleanup**: Automatic removal of old data with configurable retention

### Logging System
- **Structured Logging**: CSV format with timestamp, level, message, context, file, function
- **Context Detection**: Automatic identification of workflow vs application context
- **Thread-Safe**: Concurrent logging with file locking
- **Web Interface**: Browse and manage logs through the cache management page

## 🔧 Configuration

### Environment Variables
See `app/lib/settings.py` for configuration options including:
- Cache enablement settings
- Provider-specific configurations
- Rate limiting parameters
- Timeout and retry settings

### Data Providers
Currently integrated providers:
- **Yahoo Finance**: Stock prices, company info, historical data
- **BlackRock**: ETF holdings and fund information
- **TipRanks**: Analyst ratings and price targets
- **Zacks**: Research and financial analysis data

## 🧪 Testing

The application includes comprehensive unit testing:
- **196 test cases** covering all major components
- **Provider testing** with mocked HTTP responses
- **Storage testing** with temporary directory isolation
- **Logging testing** with thread safety validation
- **Cache testing** with various data formats

Run tests with: `make test`

## 📝 Contributing

1. Follow the coding guidelines in `CLAUDE.md`
2. Ensure all tests pass before submitting changes
3. Use `make format` and `make lint` for code quality
4. Add tests for new functionality
5. Update documentation for significant changes

## 📄 License

This project is for educational and research purposes.