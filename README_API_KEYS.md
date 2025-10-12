# API Keys Configuration Guide

This guide explains how to configure API keys for the Soccer Match Intelligence Pipeline to enable data collection from various sources.

## Overview

The pipeline supports multiple data sources for comprehensive football data collection:

- **Primary Sources**: Football Data API, Odds API, API Football
- **Additional Sources**: RapidAPI Football, SportsData.io, Betfair Exchange
- **Specialized Sources**: Weather data, News data, Social media data
- **Backup Sources**: Multiple fallback options for reliability

## Quick Setup

### 1. Edit API Keys File

Open the `api_keys_template.env` file and replace the placeholder values with your actual API keys:

```env
# Primary Football Data APIs
FOOTBALL_DATA_API_KEY=your_actual_football_data_api_key
ODDS_API_KEY=your_actual_odds_api_key
API_FOOTBALL_KEY=your_actual_api_football_key

# Additional sources (optional)
RAPIDAPI_KEY=your_actual_rapidapi_key
SPORTSDATA_IO_KEY=your_actual_sportsdata_io_key
# ... etc
```

### 3. Test Configuration

```bash
python api_integration_test.py
```

## API Key Sources

### Primary Data Sources

#### 1. Football Data API
- **Website**: https://www.football-data.org/
- **Free Tier**: 10 requests per minute
- **Data**: Matches, standings, teams, players
- **Sign Up**: https://www.football-data.org/client/register

#### 2. Odds API
- **Website**: https://the-odds-api.com/
- **Free Tier**: 500 requests per month
- **Data**: Odds, live odds, historical odds
- **Sign Up**: https://the-odds-api.com/liveapi/guides/v4/

#### 3. API Football
- **Website**: https://www.api-football.com/
- **Free Tier**: 100 requests per day
- **Data**: Fixtures, standings, statistics, predictions
- **Sign Up**: https://rapidapi.com/api-sports/api/api-football/

### Additional Data Sources

#### 4. RapidAPI Football
- **Website**: https://rapidapi.com/api-sports/api/api-football/
- **Free Tier**: 1000 requests per month
- **Data**: Fixtures, odds, predictions, statistics
- **Sign Up**: https://rapidapi.com/

#### 5. SportsData.io
- **Website**: https://sportsdata.io/
- **Free Tier**: 1000 requests per month
- **Data**: Scores, schedules, standings, news
- **Sign Up**: https://sportsdata.io/developers/api-documentation/soccer

#### 6. Betfair Exchange
- **Website**: https://developer.betfair.com/
- **Free Tier**: 200 requests per minute
- **Data**: Odds, market data, prices
- **Sign Up**: https://developer.betfair.com/

### Specialized Data Sources

#### 7. OpenWeatherMap
- **Website**: https://openweathermap.org/api
- **Free Tier**: 1000 requests per day
- **Data**: Weather conditions for match locations
- **Sign Up**: https://openweathermap.org/api

#### 8. NewsAPI
- **Website**: https://newsapi.org/
- **Free Tier**: 1000 requests per day
- **Data**: News articles, headlines
- **Sign Up**: https://newsapi.org/register

#### 9. Twitter API
- **Website**: https://developer.twitter.com/
- **Free Tier**: 300 requests per 15 minutes
- **Data**: Tweets, sentiment analysis
- **Sign Up**: https://developer.twitter.com/en/apply-for-access

## Configuration Details

### Priority System

Data sources are prioritized based on their `priority` value in the configuration:

1. **Priority 1**: Football Data API (most reliable)
2. **Priority 2**: Odds API (best odds data)
3. **Priority 3**: API Football (comprehensive data)
4. **Priority 4+**: Additional sources (backup/alternative)

### Rate Limiting

Each data source has built-in rate limiting:

- **Global Limit**: 1000 requests per hour
- **Per Source Limit**: 100 requests per hour per source
- **Burst Limit**: 50 requests per minute
- **Backoff Strategy**: Exponential backoff on failures

### Caching

Data is cached to improve performance:

- **Cache Duration**: 1 hour (3600 seconds)
- **Cache Types**: Odds, fixtures, standings
- **Max Cache Size**: 1GB
- **Cache Key**: MD5 hash of source + endpoint + parameters

### Fallback Strategy

If primary sources fail:

1. **Timeout**: 30 seconds per request
2. **Retry Attempts**: 3 attempts with 5-second delay
3. **Fallback Sources**: API Football, RapidAPI Football
4. **Data Validation**: Cross-reference multiple sources

## Data Quality Monitoring

The system monitors data quality across sources:

### Metrics Tracked
- **Response Time**: Average API response time
- **Success Rate**: Percentage of successful requests
- **Data Freshness**: How recent the data is
- **Completeness**: Percentage of expected fields present

### Validation Rules
- **Odds Consistency**: Compare odds across sources
- **Team Name Matching**: Ensure team names are consistent
- **Date Validation**: Verify match dates are valid
- **Score Validation**: Check score formats and ranges

### Alerts
- **Low Success Rate**: < 80% success rate
- **High Response Time**: > 5 seconds average
- **Data Staleness**: Data older than 1 hour

## Usage Examples

### Basic Usage

```python
from api_data_manager import APIDataManager

# Initialize manager
manager = APIDataManager()

# Get fixtures
fixtures = manager.get_fixtures(league="E1")
for response in fixtures:
    if response.success:
        print(f"Data from {response.source}: {response.data}")

# Get odds
odds = manager.get_odds(league="E1")
for response in odds:
    if response.success:
        print(f"Odds from {response.source}: {response.data}")
```

### Advanced Usage

```python
# Get standings with validation
standings = manager.get_standings(league="E1")
validation_results = manager.validate_data_quality(standings)

# Get weather data for match location
weather = manager.get_weather_data(city="London")

# Get news data for team
news = manager.get_news_data(query="football", team="Arsenal")

# Check cache statistics
cache_stats = manager.get_cache_stats()
print(f"Cache entries: {cache_stats['total_entries']}")
```

## Troubleshooting

### Common Issues

#### 1. "No valid API key found"
- **Cause**: API key not set in .env file
- **Solution**: Check .env file and ensure API key is correctly set

#### 2. "Rate limit exceeded"
- **Cause**: Too many requests to a single source
- **Solution**: Wait for rate limit to reset or use different source

#### 3. "API request failed: HTTP 401"
- **Cause**: Invalid API key
- **Solution**: Verify API key is correct and active

#### 4. "API request failed: HTTP 403"
- **Cause**: API key doesn't have permission for requested data
- **Solution**: Check API documentation for required permissions

#### 5. "No data sources available"
- **Cause**: All sources disabled or no valid API keys
- **Solution**: Enable at least one source with valid API key

### Debug Mode

Enable debug logging to see detailed API requests:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

manager = APIDataManager()
# All API requests will now be logged
```

### Testing Individual Sources

Test a specific data source:

```python
# Test only Football Data API
manager = APIDataManager()
manager.disable_source("odds_api")
manager.disable_source("api_football")

fixtures = manager.get_fixtures(league="E1")
```

## Security Best Practices

### 1. Environment Variables
- Never commit API keys to version control
- Use `api_keys_template.env` file for local development
- Use environment variables in production

### 2. API Key Rotation
- Rotate API keys regularly
- Monitor API usage for anomalies
- Revoke compromised keys immediately

### 3. Rate Limiting
- Respect API rate limits
- Implement exponential backoff
- Use caching to reduce API calls

### 4. Data Validation
- Validate all incoming data
- Cross-reference multiple sources
- Log suspicious data patterns

## Performance Optimization

### 1. Caching
- Enable caching for frequently accessed data
- Set appropriate cache durations
- Monitor cache hit rates

### 2. Parallel Requests
- Make requests to multiple sources in parallel
- Use connection pooling
- Implement request queuing

### 3. Data Filtering
- Request only needed data fields
- Use date ranges to limit data
- Filter by league/competition

### 4. Monitoring
- Track API response times
- Monitor success rates
- Alert on performance degradation

## Support and Maintenance

### Regular Maintenance
- Update API keys before expiration
- Monitor API documentation for changes
- Test data sources monthly
- Review and update rate limits

### Monitoring
- Set up alerts for API failures
- Monitor data quality metrics
- Track API usage and costs
- Review cache performance

### Updates
- Keep API client libraries updated
- Monitor for new data sources
- Update configuration as needed
- Test new features before deployment

## Conclusion

Proper API key configuration is essential for the Soccer Match Intelligence Pipeline to function effectively. By following this guide, you can:

1. Set up multiple data sources for redundancy
2. Implement proper rate limiting and caching
3. Monitor data quality and performance
4. Troubleshoot common issues
5. Optimize for performance and reliability

For additional support, refer to the individual API documentation or contact the development team.



