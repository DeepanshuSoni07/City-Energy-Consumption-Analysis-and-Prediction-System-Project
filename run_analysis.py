import sys
import os
from city_energy_analysis import CityEnergyAnalyzer

def quick_demo():
    """Run a quick demonstration"""
    print("🚀 QUICK DEMO MODE")
    print("="*30)
    
    analyzer = CityEnergyAnalyzer()
    
    # Generate smaller dataset for demo
    analyzer.generate_synthetic_data(num_zones=3, days=90)
    analyzer.clean_and_preprocess()
    analyzer.analyze_patterns()
    analyzer.build_prediction_model()
    
    # Quick prediction example
    print("\n🔮 Sample Predictions:")
    print("-" * 25)
    
    test_cases = [
        (1, 25, 60, 0, "Normal summer day"),
        (2, 35, 70, 1, "Hot day with event"),
        (3, 5, 40, 0, "Cold winter day")
    ]
    
    for zone, temp, humid, event, desc in test_cases:
        pred = analyzer.predict_consumption(zone, temp, humid, event)
        zone_name = analyzer.zone_info.get(zone, {}).get('name', f'Zone {zone}')
        print(f"{zone_name} - {desc}: {pred:.1f} kWh")
    
    print("\n✅ Quick demo completed!")

def prediction_only():
    """Run only the prediction interface"""
    print("🎯 PREDICTION ONLY MODE")
    print("="*30)
    
    if not os.path.exists('energy_model.pkl'):
        print("❌ No saved model found. Running full analysis first...")
        analyzer = CityEnergyAnalyzer()
        analyzer.generate_synthetic_data()
        analyzer.clean_and_preprocess()
        analyzer.build_prediction_model()
        analyzer.save_model_and_data()
    else:
        print("📁 Loading saved model...")
        import pickle
        with open('energy_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        analyzer = CityEnergyAnalyzer()
        analyzer.model = model_data['model']
        analyzer.scaler = model_data['scaler']
        analyzer.feature_columns = model_data['feature_columns']
        analyzer.zone_info = model_data['zone_info']
        
        # Load data for reference
        import pandas as pd
        analyzer.data = pd.read_csv('energy_data.csv')
        analyzer.data['Date'] = pd.to_datetime(analyzer.data['Date'])
    
    analyzer.interactive_console()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "demo":
            quick_demo()
        elif mode == "predict":
            prediction_only()
        else:
            print("Usage: python3 run_analysis.py [demo|predict]")
            print("  demo    - Quick demonstration with smaller dataset")
            print("  predict - Interactive prediction console only")
            print("  (no arg) - Full analysis")
    else:
        # Run full analysis
        from city_energy_analysis import main
        main()
