# THU-v4 · Distill & Solve

2026년 울름공과대학교(THU) AICOSS 하계 프로그램에서 개발한 장문 문서 질의응답 팀 프로젝트의 Pipeline v4입니다. 여러 구간에 흩어진 근거를 모아 개수 집계, 조건 비교, 문서 전체 요약에 답합니다.

## 처리 흐름

1. **문서 준비** — TXT 또는 PDF에서 텍스트를 읽습니다. 정규식으로 Figure·Box·Table 라벨을 탐지해 목록을 만들고, 문서를 청크로 나눕니다.
2. **질문 분석 · Demand Planner** — 질문 유형을 분류하고, 필요한 엔티티 유형, 수치 항목, 주제와 키워드를 정합니다.
3. **근거 추출 · Chunk Distiller** — 각 청크에서 엔티티, 수치·사실, 사건, 주장, 주제를 JSON으로 추출합니다. 청크는 병렬로 처리할 수 있으며 결과를 `KnowledgeStore`에 모읍니다.
4. **답변 생성 · Solvers** — 집계나 정렬은 가능한 범위에서 Python으로 계산하고, 근거를 결합해 설명할 때는 LLM을 사용합니다. 답변과 함께 질문별 처리 계획과 청크 정보를 저장합니다.

### 청크 구성

- **제목 기반 분할** — Markdown 제목이나 Chapter·Section 패턴에서 분할 경계를 찾습니다. 기본 설정에서 300단어 미만의 섹션은 다음 섹션과 합치고, 10,000단어를 넘는 섹션은 문단이나 문장 단위로 재분할합니다. 표는 가능한 한 함께 두고, 큰 표는 행 경계에서 나눕니다.
- **단어 수 기준 분할** — 제목을 충분히 찾지 못하면 약 10,000단어를 기준으로 나눕니다. 문장과 표를 고려해 경계를 조정합니다.
- **본문과 문맥** — 각 청크의 핵심 본문은 서로 겹치지 않습니다. 제목 기반 분할에서는 앞뒤 각각 최대 150단어, 대체 분할에서는 앞뒤 각각 최대 100단어를 참고 문맥으로 붙입니다. `chunk_manifest.json`에는 청크 ID, 핵심 본문의 시작·끝 위치와 단어 수, 섹션 제목이 기록됩니다.

### 근거 저장 방식

`KnowledgeStore`는 같은 이름의 엔티티를 합치고, 사실·사건·주장·주제에 출처 청크 ID를 붙입니다. 서로 다른 수치가 발견되면 두 기록을 보존해 모순 후보로 확인할 수 있습니다. 비교에 필요한 경우 수치 단위를 맞추고, 구조화된 근거가 부족할 때는 원문 청크를 검색합니다.

### 질문 유형별 접근 방식

| 유형 | 질문 예시 | 처리 방식 |
| --- | --- | --- |
| 집계 · Aggregation | 장마다 그림이 몇 개인가? | 그림·표·박스는 사전 목록에서 조회합니다. 일반 엔티티는 조건에 맞는 항목을 모아 중복을 제거하고 Python으로 셉니다. |
| 최댓값·최솟값 · Superlative | 면적이 가장 넓은 공원은 어디인가? | 수치를 비교 가능한 단위로 맞춰 정렬하고, 필요한 경우 원문에서 근거를 보완합니다. |
| 부재 확인 · Absence | 보기 중 문서에서 다루지 않는 주제는 무엇인가? | 원문에서 각 보기와 표현 변형을 찾아 등장한 후보를 걸러내고, 필요한 경우 LLM이 문맥을 확인합니다. |
| 모순 탐지 · Contradiction | 같은 항목에 서로 다른 수치가 적힌 곳은 어디인가? | 같은 대상에 대한 값 차이와 단위 차이를 단서로 모은 뒤 각각의 문맥으로 충돌 여부를 설명합니다. |
| 섹션 간 연결 · Cross-section | 여러 장에 흩어진 두 조건을 모두 만족하는 대상은 무엇인가? | 조건별로 저장소와 원문을 검색하고 동일한 대상에 대한 근거를 연결합니다. |
| 세부 정보 조회 · Needle | 문서의 ISBN은 무엇인가? | 구조화된 근거와 원문 문장 검색 결과를 함께 사용합니다. |
| 전체 종합 · Global synthesis | 문서 전체에서 강조하는 주제는 무엇인가? | 문서에서 추출한 주제·사건·주장·사실을 종합합니다. |

## 실행

저장소 루트에서 아래 명령을 실행합니다.

```bash
cd pipeline_v4
python -m pip install -r requirements_app.txt
python -m streamlit run app.py
```

- **Demo · Load outputs**에서 이전 실행에서 생성한 `answers.json` 또는 `answers_debug.json`을 불러올 수 있습니다.
- **Live · Run pipeline**에서 TXT 또는 PDF 문서와 YAML, JSON 또는 TXT 질문을 업로드합니다. 먼저 실행 환경에 OpenAI 호환 서버 주소를 `COURSE_API_BASE_URL`, 발급받은 키를 `COURSE_API_KEY`로 설정하세요. 서버가 [설정 파일](pipeline_v4/src_v4/config.py)에 지정된 Mistral 모델 이름을 지원해야 합니다.
- [노트북](pipeline_v4/main_v4.ipynb)에서 사용할 수 있는 [합성 문서](pipeline_v4/nationalparks_europe.txt)와 [예시 질문](pipeline_v4/dev_questions.yaml)을 포함했습니다.

## 주요 파일

| 파일 | 역할 |
| --- | --- |
| [app.py](pipeline_v4/app.py) | Streamlit 실행 화면과 결과 조회 |
| [pipeline.py](pipeline_v4/src_v4/pipeline.py) | 문서·질문 처리 단계 연결 |
| [universal_chunker.py](pipeline_v4/src_v4/universal_chunker.py) · [chunker.py](pipeline_v4/src_v4/chunker.py) | 제목 기반 분할과 단어 수 기준 분할 |
| [demand_planner.py](pipeline_v4/src_v4/demand_planner.py) · [document_inventory.py](pipeline_v4/src_v4/document_inventory.py) | 질문별 근거 계획과 그림·표·박스 목록 생성 |
| [chunk_distiller.py](pipeline_v4/src_v4/chunk_distiller.py) · [knowledge_store.py](pipeline_v4/src_v4/knowledge_store.py) | 청크별 근거 추출과 저장 |
| [solvers.py](pipeline_v4/src_v4/solvers.py) | 질문 유형별 답변 처리 |

## 기여 및 결과

정재서는 집계형 질문에서 문서 일부만 참고할 때 항목이 빠지는 원인을 분석했습니다. 청크 앞뒤의 문맥을 보존하고, 추출 근거를 공통 저장소에 구조화한 뒤 Python으로 중복을 제거하고 개수를 세는 처리 방식을 제안했습니다. Python 질의응답 파이프라인 개발을 총괄하며 코드를 수정하고 매개변수를 조정했으며, 실험 응답을 비교했습니다. 현지 실무자와 설계의 적용 가능성을 논의하고 최종 발표를 맡았습니다.

팀은 처음 접한 문서에 대한 질의응답 평가에서 가장 높은 정확도를 기록해 **Best Generalization Award**를 받았습니다.
