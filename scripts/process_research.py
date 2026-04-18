import pandas as pd
import json
import os
import glob

def process_acled_research():
    data_dir = os.path.join(os.path.dirname(__file__), '../data/')
    extension_list = ["*.csv", "*.xlsx"]
    all_files = []
    for ext in extension_list:
        all_files.extend(glob.glob(os.path.join(data_dir, ext)))
    
    li = []
    for filename in all_files:
        if "global_conflict_research" in filename or "research_summary" in filename:
            continue
        try:
            df = pd.read_csv(filename) if filename.endswith('.csv') else pd.read_excel(filename)
            if 'WEEK' in df.columns:
                li.append(df)
        except: continue

    if not li: return

    global_df = pd.concat(li, axis=0, ignore_index=True)
    
    # БАРЛЫҚ елдер бойынша статистика жасау (head(15)-ті алып тастадық)
    summary_stats = global_df.groupby('COUNTRY').agg({
        'EVENTS': 'sum',
        'FATALITIES': 'sum'
    }).sort_values(by='FATALITIES', ascending=False)
    
    # ТЕКСЕРУ: Ресей, Иран, Израиль бар ма?
    check_countries = ['Russia', 'Iran', 'Israel', 'USA', 'Ukraine']
    print("🔍 Тексеру нәтижесі:")
    for c in check_countries:
        if c in summary_stats.index:
            print(f"✅ {c}: {summary_stats.loc[c].to_dict()}")
        else:
            print(f"❌ {c} деректер базасында табылмады.")

    # Сақтау
    output_json = os.path.join(data_dir, 'research_summary.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary_stats.to_dict(orient='index'), f, ensure_ascii=False, indent=4)
    
    print(f"🚀 Дайын! Барлығы {len(summary_stats)} мемлекет сақталды.")

if __name__ == "__main__":
    process_acled_research()