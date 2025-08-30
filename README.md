# Premier League Football Match Prediction Model

A comprehensive machine learning system that predicts the probability of football match outcomes (1X2) for the English Premier League, with an automated web deployment. The model lacks real time APIs with access to ongoing and future fixtures and the therefore can only use past matches. To turn it into an accurate model use APIs with access to real-time fixtures.

## 🚀 Features

- **Machine Learning Models**: Logistic Regression and XGBoost for match outcome prediction
- **Feature Engineering**: Advanced statistical features including team form, head-to-head records, and performance metrics
- **API Integration**: Real-time data from API-Football.com
- **Web Interface**: Beautiful, responsive website displaying predictions
- **Automation**: GitHub Actions for scheduled prediction updates
- **Betting Strategy**: Value betting simulation with ROI analysis

## 🏗️ Architecture

```
├── 01_data_acquisition.py      # Historical data collection
├── 02_data_processing.ipynb    # Data cleaning and exploration
├── 03_feature_engineering.py   # Feature creation
├── 04_model_training.py        # Model training and evaluation
├── 05_betting_strategy.py      # Betting strategy simulation
├── 06_update_predictions.py    # Live prediction updates
├── docs/                       # Web interface
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── .github/workflows/          # GitHub Actions automation
└── requirements.txt            # Python dependencies
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8+
- Git
- GitHub account (for automation)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd PredictionModel
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**
   - The API key for api-football.com is already configured
   - For GitHub Actions automation, add `API_FOOTBALL_KEY` as a repository secret

### Data Collection and Model Training

1. **Collect historical data**
   ```bash
   python 01_data_acquisition.py
   ```

2. **Process and clean data**
   ```bash
   jupyter notebook 02_data_processing.ipynb
   ```

3. **Create features**
   ```bash
   python 03_feature_engineering.py
   ```

4. **Train models**
   ```bash
   python 04_model_training.py
   ```

5. **Test betting strategy**
   ```bash
   python 05_betting_strategy.py
   ```

6. **Generate live predictions**
   ```bash
   python 06_update_predictions.py
   ```

## 🌐 Web Deployment

The web interface is automatically deployed to GitHub Pages from the `docs/` folder.

### Manual Deployment

1. **Generate predictions**
   ```bash
   python 06_update_predictions.py
   ```

2. **Commit and push**
   ```bash
   git add docs/predictions.json
   git commit -m "Update predictions"
   git push
   ```

### Automated Deployment

The GitHub Actions workflow automatically runs every Thursday at 12 PM UTC to:
- Fetch latest fixtures
- Generate new predictions
- Update the website
- Commit changes

## 📊 Model Features

### Team Performance Metrics
- Average goals scored/conceded (last 5 matches)
- Team form (points per game)
- Goal difference trends
- Home/away performance

### Head-to-Head Analysis
- Historical match results
- Average goals in previous meetings
- Recent performance against specific opponents

### Temporal Features
- Season progression
- Day of week effects
- Month-based patterns

## 🎯 Prediction Outputs

The model provides:
- **Match outcome probabilities**: Home Win (H), Draw (D), Away Win (A)
- **Confidence scores**: Model certainty for each prediction
- **Detailed breakdowns**: Individual probabilities for all outcomes

## 💰 Betting Strategy

The system includes a value betting simulation that:
- Compares model predictions with implied bookmaker odds
- Identifies value betting opportunities
- Calculates potential ROI
- Uses flat staking strategy

## 🔧 Configuration

### API Configuration
- **API-Football**: Primary data source for fixtures and statistics
- **Rate Limiting**: Respects API limits (100 requests/day free tier)

### Model Parameters
- **Training Split**: Season-based to prevent data leakage
- **Feature Window**: 5-match rolling averages
- **Evaluation Metric**: Log Loss (Brier Score)

## 📈 Performance Metrics

- **Log Loss**: Primary accuracy measure
- **Confusion Matrix**: Per-class performance analysis
- **ROI**: Return on investment for betting strategy
- **Win Rate**: Percentage of successful predictions

## 🚨 Important Notes

### API Limitations
- Free tier has 100 requests per day limit
- Historical statistics may be limited
- Consider upgrading for production use

### Disclaimer
⚠️ **These predictions are for informational purposes only. Do not use for gambling decisions. Football is unpredictable and past performance doesn't guarantee future results.**

### Data Quality
- Only finished matches are used for training
- Missing data is handled with appropriate imputation
- Feature engineering ensures no future information leakage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- API-Football.com for providing match data
- Premier League for the competition structure
- Open source machine learning community

## 📞 Support

For questions or issues:
1. Check the existing issues
2. Create a new issue with detailed description
3. Include error messages and system information

## 🔮 Future Enhancements

- [ ] Additional leagues support
- [ ] Real-time odds integration
- [ ] Advanced betting strategies
- [ ] Mobile app development
- [ ] Social media integration
- [ ] Performance analytics dashboard

---

**Built with ❤️ for football analytics and machine learning enthusiasts**

