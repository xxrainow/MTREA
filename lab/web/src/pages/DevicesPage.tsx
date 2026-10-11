import './DevicesPage.css';

const arms = [
  { name: 'Follower', description: '명령에 따라 움직이는 로봇 팔' },
  { name: 'Leader', description: '사람이 움직이는 입력용 로봇 팔' },
];

function DevicesPage() {
  return (
    <main className="devices-page">
      <div className="devices-heading">
        <h1>Devices</h1>
        <p>로봇 연결, 캘리브레이션과 카메라를 관리합니다.</p>
      </div>

      <div className="devices-grid">
        {arms.map((arm) => (
          <section className="device-panel" key={arm.name}>
            <div className="device-panel-header">
              <h2>{arm.name}</h2>
              <span className="device-status">미확인</span>
            </div>

            <p className="device-description">{arm.description}</p>

            <dl className="device-details">
              <div>
                <dt>로봇 종류</dt>
                <dd>SO-101</dd>
              </div>
              <div>
                <dt>장치 ID</dt>
                <dd>설정 연결 예정</dd>
              </div>
              <div>
                <dt>시리얼 포트</dt>
                <dd>설정 연결 예정</dd>
              </div>
              <div>
                <dt>캘리브레이션</dt>
                <dd>파일 확인 예정</dd>
              </div>
            </dl>

            <div className="device-actions">
              <button type="button" disabled>
                연결 확인
              </button>
              <button type="button" disabled>
                캘리브레이션
              </button>
            </div>
          </section>
        ))}
      </div>

      <section className="device-panel devices-calibration">
        <h2>Calibration</h2>

        <ol className="calibration-steps">
          <li>
            <strong>기준 자세</strong>
            <span>선택한 팔을 안내된 기준 자세에 놓습니다.</span>
          </li>
          <li>
            <strong>동작 범위 기록</strong>
            <span>관절별 움직임 범위를 기록합니다.</span>
          </li>
          <li>
            <strong>확인 · 저장</strong>
            <span>결과를 확인하고 캘리브레이션 파일로 저장합니다.</span>
          </li>
        </ol>

        <p className="device-description">
          단계 안내 예시입니다. 실제 절차는 기능 연결 시 표시합니다.
        </p>
      </section>

      <div className="devices-section-heading">
        <h2>Cameras</h2>
        <span className="device-description">미리보기 연결 예정</span>
      </div>

      <div className="devices-grid">
        {['wrist', 'scene'].map((camera) => (
          <section className="device-panel" key={camera}>
            <div className="device-panel-header">
              <h2>{camera}</h2>
              <span className="device-status">미확인</span>
            </div>

            <div
              className="device-camera-placeholder"
              aria-label={`${camera} 카메라 미리보기 연결 전`}
            >
              카메라 미리보기
            </div>

            <button type="button" disabled>
              카메라 확인
            </button>
          </section>
        ))}
      </div>
    </main>
  );
}

export default DevicesPage;
