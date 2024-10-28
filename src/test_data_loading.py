from data.data_loader import DataLoader

def main():
    # 데이터 로더 초기화
    data_loader = DataLoader('../data')
    
    try:
        # 모든 데이터 로드
        data = data_loader.load_all()
        
        # 각 데이터셋의 기본 정보 출력
        for name, df in data.items():
            print(f"\n{name} 데이터셋 정보:")
            print(f"Shape: {df.shape}")
            print("\nColumns:")
            print(df.columns.tolist())
            print("\nSample data:")
            print(df.head(2))
            print("-" * 80)
            
    except Exception as e:
        print(f"데이터 로딩 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
