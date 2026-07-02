# VERA · M2 — Báo cáo Rule-based Report Parser

*Cập nhật: 2026-06-30 · `phase_2/src/rule_parser.py`*

---

## 1. Bối cảnh & động cơ

Module 2 của VERA cần biến **báo cáo X-quang ngực** thành **scene graph theo vùng** (flat schema:
mỗi trong 29 vùng giải phẫu → danh sách `{finding, presence, uncertain?, progression?}` trên 69
concept). Phương án ban đầu là **LLM** (Qwen2.5 3B/7B QLoRA). Kết quả khảo sát:

- **7B zero-shot**: region-F1 **0.090** — gần như đoán mò.
- **3B finetune**: chậm (≈3h/epoch trên 2×T4 cho 25k mẫu), tốn GPU, có nguy cơ hallucinate, và
  **không nằm trong train-path của MIMIC** (silver scene graph đã phủ MIMIC; LLM chỉ dùng cho
  CheXplus weak-label + parse prior report lúc launch).

→ Đặt câu hỏi: **có cần LLM không?** ImaGenome (Chest ImaGenome, Wu et al. 2021) xây silver bằng
**pipeline NLP rule-based** (lexicon 271 entity + NegEx + SpaCy + ontology), không dùng model. Nếu
tái lập được thì có một parser **CPU, deterministic, auditable, không hallucinate** — đúng tinh thần
*glass-box* của VERA. Báo cáo này tổng kết parser đó.

---

## 2. Kiến trúc

`parse_report(report, regions) -> flat schema`, gồm các tầng (tất cả deterministic):

| # | Tầng | Vai trò |
|---|------|---------|
| 1 | **Section filter** | Chỉ giữ câu trong **FINDINGS / IMPRESSION**; bỏ INDICATION / HISTORY / COMPARISON / TECHNIQUE / EXAMINATION. Các mục này chứa *câu hỏi lâm sàng* ("assess for pneumonia") chứ không phải finding. |
| 2 | **Chuẩn hoá whitespace** | Gộp xuống dòng (MIMIC wrap giữa câu) thành khoảng trắng — nếu không, danh sách phủ định bị cắt ("No a,⏎b or c" rò b,c thành dương). |
| 3 | **Tách segment** | Theo dấu câu `.;:` + liên từ tương phản (`but / however / otherwise / aside from / except`). |
| 4 | **Tách clause + NegEx forward-scope** | Tách theo dấu phẩy; phủ định chỉ áp cho trigger **đứng SAU** cue trong clause ("X with no Y" → X dương, Y âm); danh sách "No a, b or c" lan phủ định cho tới khi gặp khẳng định mới. Loại trừ "no … change" (= ổn định, không phủ định). |
| 5 | **Hedge / uncertainty** | `hedge.py::is_hedged` (regex high-precision dùng chung phase 2/3/4) → cờ `uncertain`, giữ nguyên polarity. |
| 6 | **Progression** | Lexicon cue → improved / stable / worsened (ưu tiên worsened > improved > stable). |
| 7 | **Finding lexicon** | 69 concept; trigger = (a) **curated** + (b) **mined** từ silver `phrases` (xem §4). |
| 8 | **Gán vùng** | **footprint dữ liệu** `P(region|finding)≥0.55` (mined) **∪ vị trí tường minh** (anatomy lexicon: "left base", "retrocardiac"…), rồi **lọc bằng mặt nạ hợp-lý-giải-phẫu** `P(region|finding)≥0.05` để chặn rò vùng của finding lân cận. Lateral hoá theo bên được nêu. |
| 9 | **Lan truyền cha-con (ontology)** | finding con khẳng định finding cha cùng vùng (consolidation → lung opacity), mined từ silver, đóng bao truyền. |

---

## 3. Kết quả

Đánh giá trên **full val 22k** (so silver) và **784 ảnh GOLD** (held-out, nhãn người), cùng bộ
metric (eval_sg_llm / eval_rule_parser). Cell = (vùng, finding) với presence="yes".

| Metric | vs SILVER (22k) | vs GOLD (784) |
|---|---|---|
| **presence F1 (có vùng)** | **0.860** | **0.771** |
| precision | 0.894 | 0.732 |
| recall | 0.829 | 0.815 |
| **macro-F1** | 0.779 | 0.662 |
| finding-only F1 (bỏ vùng) | 0.924 | 0.840 |
| localization gap | 0.064 | 0.069 |
| uncertain (hedge) F1 | 0.746 | 0.653 |
| progression acc (trên cell có cue) | 0.877 | 0.871 |

→ **~9.6× so với 7B zero-shot (0.090).**

### Per-finding (vs silver, P/R/F1) — nhóm thường gặp đã ngang ImaGenome

| finding | P/R/F1 | | finding | P/R/F1 |
|---|---|---|---|---|
| pleural effusion | 0.96/0.91/**0.94** | | enlarged cardiac silhouette | 0.96/0.93/**0.95** |
| lung opacity | 0.93/0.86/**0.89** | | tortuous aorta | 0.90/0.94/**0.92** |
| atelectasis | 0.92/0.84/**0.88** | | aspiration | 0.93/0.85/**0.89** |
| pulmonary edema/hazy | 0.93/0.82/**0.87** | | enteric tube | 0.86/0.93/**0.90** |
| pneumonia | 0.89/0.77/**0.82** | | airspace opacity | 0.89/0.82/**0.85** |
| consolidation | 0.91/0.74/**0.82** | | vascular congestion | 0.87/0.84/**0.85** |

**Nhóm yếu (recall-limited):** linear/patchy atelectasis 0.62, lobar/segmental collapse 0.50,
lung lesion 0.70, picc 0.57 — chủ yếu do subtype mơ hồ của silver ("volume loss" = collapse vs
atelectasis) hoặc localization tube (carina/svc).

---

## 4. Mining lexicon (đòn long-tail)

`scripts/rule_parser/2-mine_finding_lexicon.py` học trigger từ câu nguồn của silver (`phrases`),
**chỉ ảnh TRAIN** (không leak val):

- Trigger **NHẬN DIỆN finding nào** (polarity để NegEx lo), chấm theo **độ tập trung**
  `P(concept|n-gram) = #phrase nhắc concept / #phrase chứa n-gram` → giữ từ đặc trưng
  ("layering", "atelectatic"), loại từ chung ("lung", "base").
- 2 chốt chặn: **pos-rate ≥ 0.35** (loại từ phủ định/normal kiểu "clear") + **loại n-gram toàn
  anatomy/stopword**.
- Mỗi n-gram gán cho **concept đặc trưng nhất** (con, không phải cha co-tag).
- Knob khoá: concentration ≥ 0.90, pos-rate ≥ 0.35, count ≥ 40, ≤ 25/concept → **697 trigger / 66
  concept**, bổ sung lexicon curated. Tái-ngưỡng nhanh bằng `rethreshold_lexicon.py` (đọc rich
  sidecar, không cần đọc lại 148k scene).

Đóng góp: F1 0.717→0.729, recall 0.763→0.842, **macro 0.546→0.628**.

---

## 5. Hành trình tinh chỉnh (diagnostic-driven)

Mỗi bước được dẫn dắt bởi `5-diagnose_rule_parser.py` (tách lỗi LOCALIZE vs DETECT + tính trần):

| Bước | F1 |
|---|---|
| curated lexicon (NegEx + footprint + parent-child) | 0.717 |
| + mined lexicon | 0.729 |
| **+ section filter** (bỏ INDICATION/HISTORY) | **0.784** |
| + footprint threshold 0.45→0.55 | ~0.79 |
| **+ loc allowed-mask** (lọc vị trí theo hợp-lý-giải-phẫu) | **0.831** |
| + negation: bỏ "is present"/"demonstrated" khỏi reset, +"clear of"/"rather than" | 0.839 |
| **+ forward-scope NegEx** ("X with no Y" → X dương) | 0.845 |
| + bỏ trigger generic (pleural/vascular/fractures) | 0.855 |
| + flexible cardiomegaly, bidirectional vascular-calc, "no change", reticular | **0.860** |

**Hai đòn lớn nhất:** *section filter* (+0.044, precision 0.642→0.736, pneumonia P 0.51→0.84) và
*loc allowed-mask* (+0.046, precision 0.761→0.862). Cả hai đến từ việc soi mẫu lỗi cụ thể, không
phải đoán.

---

## 6. Phân tích lỗi & trần

Ở F1 0.860 (vs silver): FP ≈ 25k (localize 40% / detect 60%), FN ≈ 43k (localize 60% / miss 40%).

**Trần "vùng hoàn hảo" = F1 0.936** (nếu sửa hết lỗi localization). Khoảng cách còn lại 0.860→0.936
**thuần là localization** — chủ yếu **laterality (trái/phải)**. Đây là giới hạn của footprint
*marginal*: nó không biết per-ảnh silver gán bên nào nếu câu không nói rõ.

> Paper ImaGenome (trang 8) thừa nhận: *"phần lớn false-positive là do không phát hiện đúng
> trái/phải, vì thông tin này thường vắt qua ranh giới câu — **nằm ngoài khả năng của NLP
> pipeline**."* Tức laterality là bức tường **với cả pipeline gốc của họ**; vá nó cần **image
> grounding** (họ lấy từ link bbox↔câu), không phải lexicon.

Các hướng đã **thử và loại** (gated OFF, ghi null trong code): `RULE_LOC_ONLY` (0.674 — `_locate`
quá yếu để thay footprint), `RULE_ZONE_CLIP` (≈null), `RULE_REGION_CONTAINMENT` (âm),
`RULE_LAST_WINS` (0.857 — dù silver dùng last-mention, yes>no priority khớp tốt hơn).

---

## 7. So sánh với ImaGenome & cách đọc con số

ImaGenome báo **F1 0.939 (object-attribute, report-level)** — nhưng đó là **silver-vs-GOLD**, mà
gold được tạo BẰNG CÁCH sửa silver → tự nhiên cao. Parser của ta là **reimplementation độc lập**,
nên:

- vs silver 0.860 = "khớp pipeline của họ 86%".
- vs gold 0.771 = thấp hơn vì **gold bảo thủ hơn silver** (bác sĩ bỏ bớt finding hedged), còn parser
  trung thành tái lập silver "rộng tay" → over-predict, precision rớt 0.89→0.73.

→ Không nên so trực tiếp 0.771 với 0.939 (khác hệ quy chiếu + metric của ta nghiêm hơn, khớp đúng
(vùng,finding)). Ở mức **finding-level**, parser đạt **0.92 (silver) / 0.84 (gold)** — đã rất gần
nhãn người về mặt *phát hiện đúng bệnh*; phần thua là *gán đúng vùng*.

---

## 8. Hạn chế

- **Laterality / localization** là trần (như trên) — cần image grounding để vượt.
- **Subtype mơ hồ của silver**: linear/patchy atelectasis, lobar collapse ("volume loss") — recall
  thấp, khó tách bằng quy tắc mà không hại precision.
- **Tube localization** (ETT→carina, picc→svc) precision ~0.5-0.7 do footprint.
- Đánh giá vs silver kế thừa **lỗi của chính silver** (laterality); vs gold mới là chuẩn người
  nhưng chỉ 784 mẫu (không tune được — leakage).

---

## 9. Tái lập & cấu trúc file

```
src/rule_parser.py            # parser (env-flags khoá ở best defaults)
src/rule_finding_triggers.json, rule_region_priors.json, rule_concept_parents.json   # data bundled
scripts/rule_parser/
  1-mine_region_priors.py     # -> footprint + parent-child priors
  2-mine_finding_lexicon.py   # -> trigger lexicon (cần scene-root chest-imagenome)
  3-eval_rule_parser.py       # vs silver  -> 0.860
  4-eval_gold.py              # vs gold    -> 0.771
  5-diagnose_rule_parser.py   # tách lỗi + trần
  rethreshold_lexicon.py      # tinh chỉnh ngưỡng lexicon (giây)
```

Env-flags (locked): `RULE_SECTION_FILTER=1`, `RULE_DEFAULT_THRESH=0.55`, `RULE_ALLOW_THRESH=0.05`,
`RULE_CLAUSE_LOC=1`, `RULE_FWD_NEG=1`. Mọi thứ CPU, deterministic, reproducible.

---

## 10. Phân tích khoảng cách & quy kết nguồn lỗi

Câu hỏi: **vì sao F1 chưa cao như kỳ vọng, và mỗi phần lỗi đến từ TA hay từ mô tả mơ hồ của HỌ?**
Dùng số từ `5-diagnose_rule_parser.py` (mức F1 ≈ 0.857): **FP ≈ 24.5k** (localize 10k / detect 14k),
**FN ≈ 47k** (localize 28k / miss 19k), trần "vùng hoàn hảo" = **0.936**.

| Nguồn | Lỗi | Cỡ | Sửa được? |
|---|---|---|---|
| **A. Khác hệ quy chiếu** | "0.9" là silver-vs-gold; ta đo parser-vs-silver/gold | định khung, không phải lỗi | — |
| **B. Họ có ẢNH, ta chỉ có TEXT** | localization/laterality (FN-loc 28k + FP-loc 10k) | lớn nhất, ≈0.08 tới trần | không, bằng text |
| **C. Từ phía TA (kỹ thuật)** | `_locate` regex thay vì dependency parse; không coref; vài lỗ trigger/negation | thiểu số của detect/miss | có, nhưng nặng/rủi ro |
| **D. Mô tả MƠ HỒ của họ** | luật disambiguation/lexicon không công bố; subtype; implied-from-comparison | recall subtype + 1 phần miss | không tái lập chính xác |
| **E. Metric ta nghiêm hơn** | họ tính relation report-level; ta khớp đúng (vùng,finding) | làm số ta trông thấp hơn | một phần |

**A — Kỳ vọng đặt sai (không phải lỗi ai).** Con số 0.939 của ImaGenome là **silver-vs-GOLD**, mà
gold được tạo BẰNG CÁCH sửa silver → hai bên gần nhau sẵn. Parser của ta là bản **tái lập độc lập**;
0.86 (silver) / 0.77 (gold) **không cùng thước đo** với 0.939.

**B — Bản chất lớn nhất: thiếu modality ảnh.** 38k cell sai là **sai VÙNG**, không sai bệnh
(finding-only đã 0.92). Silver gán vùng nhờ **detect bbox + link câu↔object — tức dùng ẢNH**. Ta chỉ
đọc text: câu không nêu bên ("hazy opacities") thì silver vẫn biết nhờ ảnh, ta buộc đoán footprint →
rò sang bên kia. Đây **không phải lỗi ta, cũng không phải mơ hồ của họ — là thiếu thông tin ảnh.**
Chính paper (trang 8) thừa nhận laterality "vắt qua câu, **ngoài khả năng của NLP pipeline**" — *cả
pipeline gốc của họ cũng thua chỗ này*. Tường cứng của hướng text-only; vượt được phải **grounding
bằng ảnh** (việc của phase 3).

**C — Phần do TA, fixable nhưng mắc kỹ thuật (thiểu số).** (1) `_locate` dùng regex keyword thay vì
**SpaCy dependency parse** như họ → gắn "right" vào đúng danh từ trong câu nhiều finding chưa chuẩn
(câu nối "with/and" không phẩy vẫn rò; ta đã xấp xỉ bằng tách clause + allowed-mask). (2) Không
**coreference** ("There is an opacity. It is on the right.") — bên ở câu sau; cả ta lẫn họ đều chịu.
(3) Còn vài lỗ **negation phối hợp / hedge phức tạp** — đã vá phần lớn (is-present, clear-of,
forward-scope, no-change). Các điểm này làm được nhưng nặng/rủi ro precision, ROI thấp.

**D — Phần do mô tả MƠ HỒ của họ (không tái lập chính xác).** (1) **Luật disambiguation không công
bố**: paper chỉ nói "a set of sentence-level filtering rules, e.g. collapse = lung vs vertebral"
nhưng không đưa luật → ta không biết khi nào "volume loss" thành *lobar collapse* vs *atelectasis*
(recall collapse 0.33, linear/patchy 0.49). (2) **Subtype convention** nội bộ (atelectasis vs
linear/patchy). (3) **Implied-from-comparison**: silver suy ra finding *có mặt* từ "no change from
previous" dù câu không mô tả — quy tắc suy luận không tả đủ. (4) **Lexicon 271-entity** curate bởi 2
bác sĩ, không phát hành đầy đủ → ta mine lại 697 trigger, không phủ hết long-tail. (5) **Silver tự nó
có nhiễu** (laterality): đo vs silver bị phạt cả ở chỗ *silver sai*; đo vs gold thì precision rớt
0.89→0.73 vì silver gán **rộng tay hơn người**, ta tái lập trung thành silver → over-emit. Phần
precision-loss-vs-gold này **là nhiễu silver của họ, không phải lỗi ta**.

**E — Metric.** Ta khớp đúng cell **(vùng, finding, presence)**; Table 3 của họ đo object-attribute
**relation report-level** (rollup last-mention), lỏng hơn ở chi tiết vùng → một phần "thua" là định
nghĩa metric, không phải chất lượng.

**Một câu:** khoảng cách 0.86→0.94 *không phải do parser kém* — đa số là **laterality/vùng cần ảnh**
(bản chất, cả họ cũng thua) + **luật/lexicon họ giấu** (không tái lập được); phần *thực sự do ta và
sửa được* (dependency parse, coref) là **thiểu số** và mắc vì nặng/rủi ro. Ở mức *phát hiện bệnh*,
parser đã **0.92 (silver) / 0.84 (gold)** — gần nhãn người.

---

## 11. Kết luận / khuyến nghị

Rule parser đạt **F1 0.860 (vs silver) / 0.771 (vs gold)**, finding-level **0.92 / 0.84**, vượt xa
LLM zero-shot (0.090) và **không cần GPU, không hallucinate, audit được** — hợp glass-box thesis của
VERA. **Khuyến nghị dùng rule parser** cho nhánh attribute của M2: với MIMIC v1 dùng silver trực
tiếp (không cần cả hai), parser chỉ phục vụ CheXplus weak-label + prior report. Nếu 3B finetune chỉ
ngang parser thì chọn parser. Muốn đẩy region-F1 lên >0.9 thì lever duy nhất còn lại là **laterality
grounding bằng ảnh** — ngoài phạm vi NLP, để dành cho hướng nghiên cứu sau.
