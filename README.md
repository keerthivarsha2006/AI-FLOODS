# AI-Based Real-Time Flood Risk Prediction and Early Warning System

This project provides a Streamlit-based flood risk decision-support dashboard that blends:

- location-aware weather snapshots
- rainfall trend analysis
- water-level and rate-of-rise monitoring
- a trained flood-risk model
- explainable feature drivers
- future risk forecasting
- alert and guidance panels
- demo/fallback data when live APIs are unavailable

## Quick start

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Copy the sample env file:
   ```bash
   copy .env.example .env
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

## Important notes

- The app uses a demo dataset by default and labels it as demo/fallback data when the live API is unavailable.
- It does not replace official flood warnings from government agencies or emergency authorities.
- Risk thresholds are configurable and should be validated against site-specific historical data before operational use.
