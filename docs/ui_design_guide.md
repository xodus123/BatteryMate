사용자님의 요청에 따라 **텐서보드 스타일의 전문적인 세부 분석 대시보드** 기능을 포함하여 `UI_DESIGN_SPEC.md` 파일을 최종 업데이트해 드립니다.

이 설계는 단순히 결과만 보여주는 것이 아니라, 텐서보드처럼 **데이터의 분포, AI의 확신도, 시각적 근거**를 한 화면에서 분석할 수 있도록 구성되었습니다.

---

# 📊 UI_DESIGN_SPEC.md (v1.2 - TensorBoard Style)

## 1. 개요

본 문서는 배터리 결함 분석 시스템의 UI 구조를 정의한다. 특히 **세부사항 결과 페이지**는 TensorBoard의 데이터 시각화 방식을 채택하여 전문가가 AI의 판단 근거를 정밀하게 검토할 수 있도록 설계한다.

---

## 2. 페이지 전환 플로우

1. **Home (Upload)**: 이미지 업로드
2. **Processing**: 3개 모델 추론 중 (애니메이션)
3. **Summary**: 3개 모델 결과 요약 -> 불량 위치 이미지, 텍스트 결과 (클릭 시 상세 이동)
4. **Detailed Dashboard**: 모델별 텐서보드 스타일 상세 분석

---

## 3. 페이지별 세부 설계

### [Page 3] 통합 리포트 (Summary Report)

* **3-Way 카드**: 각 모델의 결과(Normal/Defect)와 신뢰도를 카드 형태로 출력.
* **Interactions**: 카드 클릭 시 `st.session_state`를 해당 모델 ID로 업데이트하고 상세 페이지로 화면 전환.

### [Page 4-A] 🛡️ 통합 검사 상세 (Scalars & Images Style)

* **Images 섹션**: 원본 이미지와 Grad-CAM 히트맵을 텐서보드 이미지 피드처럼 나란히 배치.
* **Scalars 섹션**: 5개 클래스별 확률 분포를 막대 그래프(Bar Chart)로 표시.
* **Distributions 섹션**: AutoEncoder의 이상 탐지 점수(Anomaly Score)를 텐서보드 히스토그램처럼 시각화하여 정상군과의 거리 표시.

### [Page 4-B] 🤖 VLM 상세 (Reasoning & Text Style)

* **Text 섹션**: AI가 생성한 심층 소견서를 텐서보드 `text` 로그 형태로 출력.
* **Grounding 섹션**: 텍스트 소견에서 언급된 결함 부위를 이미지 내 BBox로 연결하여 시각화.
* **Logs**: 모델의 프롬프트 구성 및 추론 파라미터 정보 제공.

### [Page 4-C] 🎯 VLG 상세 (Detection & PR Curve Style)

* **Detection 섹션**: GroundingDINO가 검출한 결함 부위들을 텐서보드 `Object Detection` 뷰어처럼 확대 이미지(Crop)로 리스트업.
* **Metrics 섹션**: 검출된 객체들의 신뢰도 점수 분포(Confidence Distribution) 차트 제공.
* **Thresholding**: 사용자가 임계값을 조절하면 화면의 BBox가 실시간으로 필터링되는 인터랙티브 기능.

---

## 4. 텐서보드 스타일 구현 가이드 (기술 사양)

| 기능 | 구현 방식 (Streamlit/Python) | 텐서보드 대응 탭 |
| --- | --- | --- |
| **확률/점수** | `st.plotly_chart` (Interactive Line/Bar) | **Scalars** |
| **히트맵/BBox** | `st.image` + `PIL.ImageDraw` (Overlay) | **Images** |
| **데이터 분포** | `st.altair_chart` (Histogram) | **Histograms** |
| **AI 소견서** | `st.chat_message` 또는 `st.code` | **Text** |

---

## 5. 디자인 시스템

* **배경**: `#0E1117` (Deep Dark) - 텐서보드 다크모드와 유사한 환경 제공.
* **그래프 컬러**: 텐서보드 고유의 비비드한 컬러 팔레트 사용 (Orange, Blue, Pink).
* **컴포넌트**: `st.expander`를 활용하여 상세 로그를 접고 펼 수 있도록 구성.

---

