import pandas as pd
import numpy as np
from processors.simple_processor import SimpleProcessor
from data.data_loader import DataLoader
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_processor_output_shapes(processor_output, n_users, n_items):
    """프로세서 출력 형태 검증"""
    assert 'user_features' in processor_output, "user_features가 없습니다"
    assert 'item_features' in processor_output, "item_features가 없습니다"
    assert 'interaction_matrix' in processor_output, "interaction_matrix가 없습니다"
    
    assert processor_output['user_features'].shape[0] == n_users, f"user_features 행 개수가 맞지 않습니다. Expected: {n_users}, Got: {processor_output['user_features'].shape[0]}"
    assert processor_output['item_features'].shape[0] == n_items, f"item_features 행 개수가 맞지 않습니다. Expected: {n_items}, Got: {processor_output['item_features'].shape[0]}"
    assert processor_output['interaction_matrix'].shape == (n_users, n_items), f"interaction_matrix 크기가 맞지 않습니다. Expected: ({n_users}, {n_items}), Got: {processor_output['interaction_matrix'].shape}"

def test_processor_output_values(processor_output):
    """프로세서 출력값 검증"""
    # 값 범위 체크
    assert np.all(np.isfinite(processor_output['user_features'])), "user_features에 무한값이 있습니다"
    assert np.all(np.isfinite(processor_output['item_features'])), "item_features에 무한값이 있습니다"
    assert np.all(np.isfinite(processor_output['interaction_matrix'])), "interaction_matrix에 무한값이 있습니다"
    
    # 평점 범위 체크 (0-10)
    assert np.all(processor_output['interaction_matrix'] >= 0), "평점에 음수가 있습니다"
    assert np.all(processor_output['interaction_matrix'] <= 10), "평점이 10을 초과합니다"

def analyze_processor_output(processor_output):
    """프로세서 출력 분석"""
    logger.info("\n=== 프로세서 출력 분석 ===")
    
    # 특징 분석
    logger.info("\n[사용자 특징]")
    logger.info(f"Shape: {processor_output['user_features'].shape}")
    logger.info(f"평균: {processor_output['user_features'].mean():.3f}")
    logger.info(f"표준편차: {processor_output['user_features'].std():.3f}")
    logger.info(f"결측치 수: {np.isnan(processor_output['user_features']).sum()}")
    
    logger.info("\n[책 특징]")
    logger.info(f"Shape: {processor_output['item_features'].shape}")
    logger.info(f"평균: {processor_output['item_features'].mean():.3f}")
    logger.info(f"표준편차: {processor_output['item_features'].std():.3f}")
    logger.info(f"결측치 수: {np.isnan(processor_output['item_features']).sum()}")
    
    logger.info("\n[상호작용 행렬]")
    logger.info(f"Shape: {processor_output['interaction_matrix'].shape}")
    logger.info(f"평균 평점: {processor_output['interaction_matrix'][processor_output['interaction_matrix'] > 0].mean():.3f}")
    logger.info(f"평점 분포:\n{pd.Series(processor_output['interaction_matrix'][processor_output['interaction_matrix'] > 0].flatten()).value_counts().sort_index()}")
    logger.info(f"총 상호작용 수: {(processor_output['interaction_matrix'] > 0).sum()}")

def main():
    try:
        # 1. 데이터 로드
        logger.info("데이터 로딩 중...")
        data_loader = DataLoader('../../data/')
        data = data_loader.load_all()
        
        # 2. 프로세서 초기화 및 실행
        logger.info("SimpleProcessor 실행 중...")
        processor = SimpleProcessor()
        processor.fit(data)
        output = processor.transform(data)
        
        # 3. 출력 검증
        logger.info("출력 검증 중...")
        n_users = len(data['users']['user_id'].unique())
        n_items = len(data['books']['isbn'].unique())
        test_processor_output_shapes(output, n_users, n_items)
        test_processor_output_values(output)
        
        # 4. 출력 분석
        analyze_processor_output(output)
        
        logger.info("\n모든 테스트가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
