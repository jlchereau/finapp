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

### Modular Component Design
FinApp follows a sophisticated modular component architecture where complex pages are broken down into self-contained components:

- **Component Isolation**: Each chart or feature component manages its own state and calls exactly one workflow
- **Decentralized Events**: Components communicate through Reflex decentralized event handlers
- **Computed Variables**: Main pages use `@rx.var` for shared computed properties like `base_date`
- **Page Coordination**: Main `page.py` files coordinate multiple components through update methods

**Example Structure**:
```
app/pages/compare/
├── page.py              # Main layout + computed vars + coordination
├── returns_chart.py     # Self-contained returns component
├── volatility_chart.py  # Self-contained volatility component
└── metrics.py           # Combined metrics component
```

Each component exports both a rendering function and an update event handler, enabling clean separation of concerns and improved maintainability.

### Technology Stack
- **Frontend**: Reflex 0.8.12 (Python-to-React) with TailwindV4 styling
- **Backend**: Python 3.12+ with async/await support
- **Data Processing**: pandas 2.3.2, numpy 2.3.3, DuckDB 1.4.0 for analytics
- **Visualization**: matplotlib 3.10.6, plotly 6.3.0, reflex-pyplot for interactive charts
- **Storage**: Date-based file organization (JSON, Parquet, CSV)
- **Testing**: pytest 8.4.2 with 627 comprehensive unit tests

### Key Components
- **Modular Components**: Self-contained chart and feature components with individual state management
- **Data Providers**: Modular system for financial API integration
- **Caching Layer**: Automatic response caching with configurable TTL
- **Logging System**: Structured CSV logging with context detection
- **Storage Utilities**: Date-based folder management and cleanup
- **Web Interface**: Multi-page application with reactive state management and decentralized event handlers

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

### Creating New Components
When adding new functionality, follow the modular component pattern:

1. **Create Component File**: Add `new_component.py` in the appropriate page directory
2. **Component Structure**: Include state class, update method, event handler, and component function
3. **Single Flow Rule**: Each component should call exactly one workflow for data processing
4. **Export Pattern**: Export both component function and update event handler
5. **Page Integration**: Import and coordinate in main `page.py` file
6. **Add Tests**: Create comprehensive unit tests for the new component

See `app/pages/compare/returns_chart.py` for a complete example following this pattern.

### Project Structure
```
finapp/
├── app/                   # Main application code
│   ├── flows/             # LlamaIndex workflows (markets/, compare/)
│   ├── lib/               # Core utilities (storage, logging, settings)
│   ├── providers/         # Data providers and caching
│   ├── pages/             # Web application pages
│   │   ├── markets/       # Modular markets page (page.py + components)
│   │   ├── compare/       # Modular comparison page (page.py + components)
│   │   └── optimize/      # Modular optimization page (page.py + components)
│   └── templates/         # Layout templates
├── data/                  # Date-organized cache storage
│   └── YYYYMMDD/          # Daily folders with logs and cached data
├── tests/                 # Comprehensive test suite (627 tests)
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
- **FRED (Federal Reserve)**: Economic data and indicators
- **Shiller**: CAPE ratio and market valuation data
- **IBKR**: Interactive Brokers integration (placeholder)

## 🧪 Testing

The application includes comprehensive unit testing:
- **627 test cases** covering all major components
- **Modular component testing** with individual state and event handler validation
- **Provider testing** with mocked HTTP responses and Pydantic model validation
- **Storage testing** with temporary directory isolation
- **Logging testing** with thread safety validation
- **Cache testing** with various data formats (JSON, Parquet)
- **Workflow testing** with LlamaIndex integration

Run tests with: `make test`

## 📝 Contributing

1. Follow the coding guidelines in `CLAUDE.md`
2. Ensure all tests pass before submitting changes
3. Use `make format` and `make lint` for code quality
4. Add tests for new functionality
5. Update documentation for significant changes

## 📄 License

This project is for educational and research purposes.