import pandas as pd
import json
import os
import glob

def process_acled_research():
    # 1. Жолды анықтау
    data_dir = os.path.join(os.path.dirname(__file__), '../data/')
    
    # CSV және XLSX файлдарын іздеу
    extension_list = ["*.csv", "*.xlsx"]
    all_files = []
    for ext in extension_list:
        all_files.extend(glob.glob(os.path.join(data_dir, ext)))
    
    li = []
    print(f"⏳ Іздеу басталды: {data_dir}")
    
    for filename in all_files:
        # Нәтижелік файлдарды өткізіп жіберу
        if "global_conflict_research" in filename or "research_summary" in filename:
            continue
            
        try:
            # Файл түріне қарай оқу
            if filename.endswith('.csv'):
                df = pd.read_csv(filename)
            else:
                df = pd.read_excel(filename)
                
            # ACLED файлы екенін тексеру (ішінде 'WEEK' бағаны болуы керек)
            if 'WEEK' in df.columns:
                li.append(df)
                print(f"✅ Оқылды: {os.path.basename(filename)}")
            else:
                print(f"⏩ Өткізілді (ACLED емес): {os.path.basename(filename)}")
        except Exception as e:
            print(f"❌ Қате: {os.path.basename(filename)} оқу мүмкін болмады: {e}")

    if not li:
        print("🛑 Жарамды деректер табылмады. Excel файлдарыңыздың 'data' папкасында екенін тексеріңіз.")
        return

    # 2. Біріктіру
    global_df = pd.concat(li, axis=0, ignore_index=True)
    global_df['WEEK'] = pd.to_datetime(global_df['WEEK'])
    
    # 3. AI үшін статистика
    summary_stats = global_df.groupby('COUNTRY').agg({
        'EVENTS': 'sum',
        'FATALITIES': 'sum'
    }).sort_values(by='FATALITIES', ascending=False).head(15)
    
    # 4. Сақтау
    output_csv = os.path.join(data_dir, 'global_conflict_research.csv')
    output_json = os.path.join(data_dir, 'research_summary.json')
    
    global_df.to_csv(output_csv, index=False)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary_stats.to_dict(orient='index'), f, ensure_ascii=False, indent=4)
    
    print("-" * 30)
    print(f"🚀 Сәтті аяқталды! {len(li)} файл өңделді.")
    print(f"📍 Нәтиже: {output_json}")

if __name__ == "__main__":
    process_acled_research()