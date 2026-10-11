import type { PageName } from '../components/AppLayout';
import './OverviewPage.css';

type OverviewPageProps = {
  onNavigate: (page: PageName) => void;
};

function OverviewPage({ onNavigate }: OverviewPageProps) {
  return (
    <main className="overview-page">
      <div className="overview-heading">
        <h1>Overview</h1>
        <p>로봇을 준비하고, 데이터를 수집하고, 정책을 학습하세요.</p>
      </div>

      <div className="overview-system-status" aria-label="시스템 상태">
        <span className="overview-system-status-item">
          Robot <strong>미확인</strong>
        </span>

        <span className="overview-system-status-item">
          GPU 서버 <strong>미확인</strong>
        </span>

        <span className="overview-system-status-item">
          Policy <strong>미확인</strong>
        </span>

        <span className="overview-system-status-note">상태 조회 연결 예정</span>
      </div>

      <div className="overview-panels">
        <section className="overview-panel">
          <div className="overview-panel-header">
            <h2>Robot</h2>
            <span className="overview-status">연결 전</span>
          </div>

          <p className="overview-value">SO-101</p>
          <p className="overview-description">Leader · Follower · 카메라</p>

          <div className="overview-actions">
            <button type="button" onClick={() => onNavigate('Devices')}>
              장치 설정
            </button>

            <button
              type="button"
              className="overview-button-accent"
              onClick={() => onNavigate('Teleoperate')}
            >
              Teleoperation
            </button>
          </div>
        </section>

        <section className="overview-panel">
          <div className="overview-panel-header">
            <h2>Dataset</h2>
            <span className="overview-status">선택 전</span>
          </div>

          <p className="overview-value">데이터셋 선택</p>
          <p className="overview-description">
            시연 녹화 · episode 탐색 · 데이터 검사
          </p>

          <div className="overview-actions">
            <button type="button" onClick={() => onNavigate('Datasets')}>
              데이터셋 탐색
            </button>

            <button type="button" onClick={() => onNavigate('Collect')}>
              Collect episodes
            </button>
          </div>
        </section>

        <section className="overview-panel">
          <div className="overview-panel-header">
            <h2>Create a policy</h2>
            <span className="overview-status">실행 전</span>
          </div>

          <p className="overview-value">π0.5</p>
          <p className="overview-description">H0 · expert-only fine-tuning</p>

          <div className="overview-actions">
            <button
              type="button"
              className="overview-button-green"
              onClick={() => onNavigate('Train')}
            >
              Training
            </button>

            <button type="button" onClick={() => onNavigate('Run & Evaluate')}>
              Run & Evaluate
            </button>
          </div>
        </section>
      </div>

      <section className="overview-jobs">
        <div className="overview-jobs-header">
          <h2>Jobs</h2>
          <button type="button" onClick={() => onNavigate('Jobs')}>
            작업 관리로 이동 →
          </button>
        </div>

        <p>작업 목록과 로그는 Jobs 화면에서 확인할 수 있습니다.</p>
      </section>
    </main>
  );
}

export default OverviewPage;
