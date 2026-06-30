# VERA — Tổng hợp Concern phương pháp luận (từ câu hỏi YOLO → hiện tại)

> Mỗi mục ghi theo bốn phần: **Vấn đề** (concern) · **Nguyên nhân** · **Nghi vấn** (điều chưa
> trả lời được bằng số) · **Giải pháp**. Đọc Phần A trước — nó là sợi chung khiến mọi mục bên dưới
> thực ra là *cùng một loại lỗi* lặp lại ở các vị trí khác nhau.

---

## A. NĂM NGUYÊN TẮC XUYÊN SUỐT (sợi chung của mọi concern)

1. **Train trên đúng thứ sẽ có lúc inference.** Scene-graph (gold box, nhãn-từ-report) chỉ tồn tại
   lúc train; lúc launch chỉ có ảnh → detector box → M3 → M4. Mọi thứ "biến mất lúc launch" KHÔNG
   được là *input* lúc inference, chỉ được là *nhãn/giám sát* lúc train.
2. **Đừng để output của một thành phần đóng vai "sự thật" cho thành phần khác ở chỗ sẽ báo cáo kết quả.**
   Nhãn yếu M2 không được là ground-truth lúc eval; concept không được làm "why" nếu ảnh→concept chưa
   tin cậy; teacher xem report không được dạy student bịa.
3. **Faithfulness sống ở mắt xích yếu nhất, không ở con số headline.** Một AUC cao ở tầng *sau* (vd
   finding→bệnh) có thể che một tầng *trước* yếu (ảnh→finding). Luôn truy về mắt xích đầu tiên.
4. **Kỷ luật scope: "ý hay, nhưng là bài khác".** Concept-bottleneck đầy đủ, prior-report bất đối xứng,
   privileged-information distillation — đều giá trị, đều bị hoãn để v1 giữ một câu chuyện không dấu hoa thị.
5. **Đo, đừng cãi.** Gần như mọi concern dưới đây giải bằng một ablation/phép đo rẻ, không bằng tranh luận.

---

## B. DANH SÁCH CONCERN

### B1. Detector/YOLO chỉ "đoán template" hay thật sự nhìn?
- **Vấn đề:** YOLO có thể đặt 29 box dựa vào *vị trí trung bình* của giải phẫu thay vì đọc nội dung ảnh.
- **Nguyên nhân:** Giải phẫu ngực rất có cấu trúc (tim giữa-trái, đỉnh trên, góc sườn hoành dưới-ngoài)
  → bài toán định vị "dễ một cách lừa"; một detector lười dựa prior-vị-trí vẫn ăn IoU cao.
- **Nghi vấn:** (i) Box dựa-prior có **sụp ở ca bất thường** (tràn dịch lớn đẩy trung thất, xẹp phổi,
  tim rất to) — đúng ca cần định vị nhất — không? (ii) Điều đó có **quan trọng** không, khi box chỉ là
  mask để attention-pool gom feature (M3 mới là chỗ "nhìn")?
- **Giải pháp (đo, 4 phép):**
  1. **Static-prior baseline:** box trung bình cố định (bỏ ảnh) áp lên test; YOLO phải vượt rõ baseline này.
  2. **Phân tầng theo độ bất thường giải phẫu:** đo IoU trên tầng điển hình vs lệch-chuẩn → lộ chỗ detector sụp.
  3. **Oracle ablation gold-box vs detector-box ở mức M3:** chạy M3 hai lần, so macro-F1 → lỗi định vị
     lan xuống bao nhiêu. Chênh nhỏ ⇒ box không phải nút thắt; chênh lớn ⇒ đầu tư detector.
  4. **Perturbation test:** che/đảo nội dung vùng → box có đổi không (không đổi = bỏ qua ảnh).
- **Quyết định:** dùng **predicted box (Phase 2)** cho cả train lẫn infer (khớp phân phối); gold box chỉ
  để train detector + oracle ablation. **Tuyệt đối không** train-trên-gold / infer-trên-predicted.
- **Lưu ý faithfulness:** box đúng là *điều kiện cần*, không đủ — `α`/`softmax_r` vẫn phải tự chứng minh
  faithful qua deletion/insertion (B6).

### B2. Nguồn ground-truth temporal cho M4 — TỬ HUYỆT LỚN NHẤT
- **Vấn đề:** Nhãn improved/stable/worsened để **đánh giá** M4 đến từ đâu.
- **Nguyên nhân:** Nếu từ scene-graph LLM / pseudo-label step-7 → đang **đo faithfulness của model bằng
  nhãn do model khác sinh**.
- **Nghi vấn:** Cả claim "temporal faithful" (điểm bán chính của VERA) đứng hay sụp ở đây.
- **Giải pháp:** Tập **test temporal người-gán** (loại MS-CXR-T — cần xác minh phiên bản/độ phủ), tách
  bạch khỏi nhãn train. Nhãn yếu chỉ được dùng để **train**, **eval-never**. Khởi động tìm nguồn này
  *sớm, song song* với train M3/M4 (tìm nhãn temporal người-gán mất thời gian).

### B3. 69 concept: định nghĩa, mapping, và cái bẫy AUC 0.9
- **Vấn đề:** 69 concept là gì, có đủ không, và AUC 0.9 (cây quyết định) chứng minh điều gì.
- **Nguyên nhân:** 69 concept = **finding của Chest ImaGenome**. Cây quyết định đạt AUC 0.9 được nuôi
  bằng finding **ground-truth**, nên nó đo *finding → bệnh*, KHÔNG đo *ảnh → finding*.
- **Nghi vấn:** Faithfulness của "why-bằng-concept" sống ở mắt xích **ảnh → finding** (chưa kiểm). Nếu
  ảnh→finding yếu nhưng finding tương quan mạnh với bệnh, hệ vẫn ra AUC bệnh cao **vì lý do sai** →
  "bệnh d vì finding c" thành plausible-mà-không-faithful (đúng bẫy VERA tránh). Ngoài ra: 69 có **đủ**
  để suy mọi bệnh không (có bệnh nào "mù" với mọi concept?); ImaGenome **không áp trực tiếp** lên CheXplus.
- **Giải pháp:**
  1. AUC 0.9 chứng minh **sufficiency** (bottleneck đủ rộng) — giữ, đưa vào paper, nhưng đừng dừng ở đó.
  2. **Đo ảnh→finding per-concept** (F1/AUC trên ImaGenome người-gán) — go/no-go thật. **Nhìn phân phối,
     không nhìn trung bình**: chia 69 thành nhóm tốt / trung bình / tệ.
  3. **Intervention test:** can thiệp bật/tắt concept → bệnh đổi đúng hướng? Có ⇒ bottleneck thật; không
     ⇒ concept trang trí (tương quan, không nhân quả).
  4. **Gating mức-từng-finding:** tách **vai tính-toán** (cả 69, giữ sufficiency) khỏi **vai giải-thích**
     (chỉ subset pass cả bước 2+3). Report chỉ nói "vì finding c" khi c thuộc subset đáng tin.
  5. **Incompleteness (không chắc chỉ 69):** dùng **bất đồng giữa direct-head và concept-routed** làm
     tín hiệu abstention — lệch nhiều ⇒ "69 finding không giải thích được ca này" ⇒ hedge/nhường radiologist.
  6. **CheXplus:** train/eval ảnh→finding **trên ImaGenome người-gán**; finding-trên-CheXplus chỉ pseudo
     để pretrain/augment, **không eval**.

### B4. Calibration mới là *phương pháp*, chưa phải *bằng chứng*
- **Vấn đề:** M5 dựa nặng vào confidence-calibrate để assert/hedge/omit và abstention.
- **Nguyên nhân:** "temperature scaling" là cách làm, chưa chứng minh calibration thật sự đạt — nhất là
  trên dữ liệu **mất cân bằng nặng** (long-tail rất dễ hỏng calibration).
- **Nghi vấn:** Nếu confidence chưa đáng tin thì ngưỡng `τ_assert/τ_uncertain/τ_prog` và cả tầng
  abstention của M5 xây trên cát.
- **Giải pháp:** Báo cáo **reliability diagram + ECE**, tách per-class và đặc biệt trên lớp hiếm; nếu
  calibration kém ở lớp nào thì ngưỡng phải đặt riêng hoặc lớp đó mặc định hedge.

### B5. Nhánh global: cơ chế rõ, nhưng grounding cho finding quan hệ chưa rõ
- **Vấn đề:** Finding quan hệ (cardiomegaly, phù lan toả, low lung volumes) đi qua GlobalHead, gộp ở
  image-logit qua gate — vậy chúng grounding "ở đâu" để bác sĩ verify?
- **Nguyên nhân:** Chúng không thuộc một box nào → câu chuyện "mọi claim truy về một ô vùng" có lỗ hổng
  cho lớp finding này; và cái gate `g=σ(...)` quyết global/local thắng tự nó có cần giải thích được không.
- **Nghi vấn:** Khi report khẳng định một global finding, faithfulness của nó nằm ở đâu.
- **Giải pháp:** Định nghĩa **grounding mức toàn-ảnh tường minh** cho lớp finding quan hệ, **dán nhãn
  khác** với finding theo vùng (đừng giả vờ chúng có grounding vùng). Trung thực về việc đây là lớp
  "grounding toàn cục", không phải "ô vùng".

### B6. Deletion/insertion: baseline thay thế & tín hiệu audit chưa chốt
- **Vấn đề:** Kế hoạch ablate bằng zero `region_feat` để dựng đường cong faithfulness.
- **Nguyên nhân:** Kết quả deletion/insertion **phụ thuộc mạnh vào "giá trị thay thế"** (zero vs mean vs
  noise); và chưa chốt audit faithfulness của *tín hiệu nào* (α / softmax_r / gradient — mỗi cái một đường).
- **Nghi vấn:** AUC đẹp có thể là artifact của lựa chọn baseline; "đang đo faithful của cái gì" chưa rõ.
- **Giải pháp:** Luận chứng vì sao chọn baseline (và kiểm robustness với ≥1 baseline khác); nói rõ đang
  audit tín hiệu nào. **Đặt deletion/insertion làm bằng chứng *phụ*** — đảm bảo cấp construction (out-of-table,
  temporal-guard) mới là claim faithfulness *chính* (deletion/insertion chỉ đo importance ngoài, cần-không-đủ).

### B7. So sánh baseline xuyên paradigm — dễ khập khiễng
- **Vấn đề:** VERA hi sinh fluency lấy faithfulness; nếu chỉ báo CheXbert-F1/BLEU thì **thua trên sân
  của baseline sinh** và câu chuyện "operating point khác" không tự hiện.
- **Nguyên nhân:** Faithfulness metric của VERA (out-of-table, temporal-halluc) định nghĩa trên "bảng";
  baseline sinh tự do **không có bảng** → đo out-of-table cho chúng kiểu gì.
- **Nghi vấn:** Cách operationalize đo faithfulness *xuyên các paradigm khác nhau* một cách công bằng.
- **Giải pháp:** Thiết kế khung so sánh có **faithfulness là một trục đo cho cả baseline** (vd dùng
  round-trip CheXbert trên *output của baseline* để ước lượng out-of-table/temporal-halluc của chúng).
  Chủ động **sở hữu trade-off**: báo cả ba nhóm metric (clinical-efficacy · faithfulness · fluency) và
  lập luận VERA là một *điểm vận hành cố ý* vì patient-safety. (Đây là concern đặc thù MICCAI — quyết
  định bài có *trông như* so sánh công bằng không.)

### B8. Claim "bác sĩ ít kinh nghiệm" — động cơ trung tâm nhưng chưa đo được
- **Vấn đề:** Lập luận an toàn của VERA ("faithful giúp bác sĩ ít kinh nghiệm không bị dẫn sai") hiện là
  khẳng định, không phải phép đo.
- **Nguyên nhân:** Không có dữ liệu người để chứng minh tác động này.
- **Nghi vấn:** Reviewer chấp nhận *động cơ* nhưng không cho *credit* như một kết quả.
- **Giải pháp:** Hoặc làm **reader study nhỏ** (vài bác sĩ ít kinh nghiệm, faithful vs free-prose, đo tỉ lệ
  bị dẫn sai) → biến khẳng định thành bằng chứng; hoặc **hạ xuống motivation** và không over-claim trong
  contribution. Quyết sớm vì nó ảnh hưởng cách viết phần đóng góp.

### B9. Vòng lặp nhãn yếu M2 (MIMIC → CheXplus → train M3/M4) — distillation thầm lặng
- **Vấn đề:** M2 fine-tune trên MIMIC → sinh pseudo scene-graph cho CheXplus → lấy chính nhãn yếu đó
  train M3/M4. Đây là knowledge distillation ngầm: M3/M4 học bắt chước M2, không học từ sự thật.
- **Nguyên nhân & ba tầng hại:** (i) **trần hiệu năng bị khoá bởi M2** (lỗi hệ thống của M2 thành "sự
  thật"); (ii) **lỗi tương quan, không ngẫu nhiên** — vì M2 là LLM bóc finding, lỗi của nó *chính là
  hallucination* ⇒ nguy cơ **dạy M3/M4 hallucinate theo kiểu M2**, mâu thuẫn điểm bán của bài; (iii)
  **domain shift MIMIC→CheXplus** làm nhãn yếu tệ nhất ở đúng phần dữ liệu mới mà ta thêm vào.
- **Nghi vấn:** Nhãn yếu có làm hỏng model so với chỉ dùng MIMIC không (chính câu hỏi của bạn).
- **Giải pháp:**
  1. **Lằn ranh bất biến:** nhãn yếu → **train: được; eval/báo cáo: tuyệt đối không.**
  2. **Đo chất lượng nhãn yếu trước khi xài:** so M2-weak vs subset người-gán, **F1 per-class**; lớp nào
     tệ thì không train trên CheXplus / hạ trọng số.
  3. **Pretrain-rồi-finetune** thay vì trộn đồng thời: pretrain trên CheXplus-weak, **kết thúc** trên
     MIMIC-clean để dữ liệu sạch "có tiếng nói cuối", ghi đè bias nhãn yếu.
  4. **Soft/weighted loss** theo confidence của M2 (đừng coi mọi nhãn yếu như nhau).
  5. **Sơ đồ data-provenance**: mỗi tập đóng vai gì ở train/val/test, nhãn đến từ đâu.

### B10. Vai trò CheXplus & NIH (giữ hay bỏ)
- **Vấn đề:** Hai dataset ngoài MIMIC nên đóng vai gì, có nên bỏ khỏi pipeline không.
- **Nguyên nhân & phân biệt:**
  - **NIH**: chỉ ảnh, **không report** → không parse finding → không sinh scene-graph → **không đóng góp
    được cho phần region/temporal** (phần mới của bài). Có nhãn 14 bệnh ảnh-level (yếu).
  - **CheXplus**: có ảnh + report → đi qua detector + parser LLM ra scene-graph yếu → train được cả phần
    region. Nhưng nhãn yếu = rủi ro nhiễm cao nhất (xem B9).
- **Nghi vấn:** "Nhiều data" với nhãn yếu là tài sản hay nợ; giá trị thật là *diversity phân phối* hay
  *volume*.
- **Giải pháp:**
  - **NIH → cross-dataset eval cho M3-disease** (chứng minh generalize), **không train**. Không giúp M4/M5.
  - **CheXplus → mặc định pretrain/augment (giá trị = diversity, không phải volume), quyết bằng ablation:**
    train có/không CheXplus, đo trên **test MIMIC sạch**. Cải thiện ⇒ giữ; không đổi ⇒ bỏ (không mang rủi
    ro để đổi số 0); tệ đi ⇒ bỏ là kết quả củng cố luận điểm faithfulness. **Nghiêng "bỏ sau khi đo".**
  - **Lằn ranh:** cả hai chỉ train/pretrain, eval-never; mọi số trong bảng từ nhãn người-gán.
  - Rút về **MIMIC + ImaGenome** vẫn **giữ nguyên toàn bộ novelty** — phạm vi hẹp hơn nhưng sạch, đúng VERA.

### B11. Prior có thể có report lúc launch — bất đối xứng thông tin (quyết định kiến trúc, không phải chi tiết)
- **Vấn đề:** Ràng buộc nền tảng cũ ("launch chỉ có ảnh, không report") chưa chính xác: **prior thường vẫn
  có report** (lần khám cũ đã được đọc). Tình huống thật: **current chỉ ảnh; prior ảnh + (thường) report.**
- **Nguyên nhân:** Report prior là **dữ liệu thật, verify được** (khác bẫy GAN của OsteoGA) → có thể parse
  ra scene-graph *thật* cho phía prior, làm M4 chính xác hơn ở ca có prior-report.
- **Nghi vấn:** Nếu cho M3/M4 xem scene-graph-prior lúc train nhưng **không phải prior nào cũng có report**
  lúc launch → model học dựa vào input *đôi khi vắng mặt* → giòn, hoặc **hallucinate phần prior bị thiếu**.
- **Giải pháp (chọn có ý thức, đừng để trôi):**
  - **A (khuyên cho v1):** M3/M4 **thuần ảnh** lúc inference; report/scene-graph chỉ làm **nhãn train**.
    Sạch, đối xứng, không dấu hoa thị. ← chốt cho v1.
  - **B (future/section riêng):** mô hình hoá bất đối xứng tường minh (cờ "prior-report-available" +
    đường xử lý hai chế độ + degrade gracefully) — một đóng góp riêng, cần ablation "có/không prior-report".
  - **C (CẤM):** để M3/M4 "tiện thì xem scene-graph prior" mà không tách chế độ/không cờ/không ablation —
    tạo model mà faithfulness phụ thuộc dữ liệu có-sẵn-hay-không, không kiểm soát.

### B12. Transfer learning / Privileged Information — khả quan, nhưng là *bài khác*
- **Vấn đề:** Hướng "train với thông tin đầy đủ, lúc launch thiếu thì giữ tri thức để không drop" — chính
  là **LUPI / distillation / modality-dropout**.
- **Nguyên nhân & cái bẫy riêng:** Teacher (xem scene-graph/report) dạy student (chỉ ảnh) **bắt chước kết
  luận**. Nếu teacher kết luận nhờ *đọc report* mà **ảnh không đủ tín hiệu**, thì đang **dạy student nói X
  mà không nhìn thấy X** — hallucination *được chưng cất vào trọng số*, nhất quán, tự tin, không cờ nào bật.
- **Nghi vấn:** Tri thức distill là "đọc ảnh tốt hơn" (an toàn) hay "bịa theo report" (độc)? Hai loại
  trộn lẫn trong scene-graph-từ-report.
- **Giải pháp:** **Tách thành study riêng (VERA-privileged).** Lý do: (i) đủ lớn để tự đứng; (ii) đòi một
  nhánh evaluation riêng (test "ảnh không có X nhưng report prior có X — model có nói X không"); (iii)
  đứng ở *thế đối lập triết học* với điểm bán v1 ("model biết nhiều hơn cái nó nhìn") → trộn làm loãng cả
  hai. **Câu hỏi gác cổng để bài-sau vẫn faithful:** *chỉ distill những finding chứng minh được là suy ra
  từ ảnh.* Kế thừa toàn bộ hạ tầng M1–M5 của v1 → hoãn không mất gì.

---

## C. VIỆC CẦN ĐO NGAY (gần như mọi concern giải bằng phép đo rẻ)

Ưu tiên theo mức "điều kiện cần để số chính có nghĩa":

1. **[B2] Nguồn ground-truth temporal người-gán** — khởi động sớm, song song train. *Điều kiện tồn tại
   của đóng góp chính.*
2. **[B3] Bảng ảnh→finding per-concept + intervention test** — quyết số phận hướng concept (faithful "why").
3. **[B9/B10] Ablation có/không CheXplus & NIH trên test MIMIC sạch** + bảng kiểm-định-nhãn-yếu
   (M2-weak vs người-gán per-class) + sơ đồ data-provenance.
4. **[B4] Calibration thực chứng** (reliability diagram + ECE, per-class, lớp hiếm).
5. **[B1] Static-prior baseline + oracle ablation gold-box vs detector-box** — chặn nghi ngờ "YOLO đoán
   template" và đo lỗi định vị lan xuống M3.
6. **[B6] Chốt baseline thay thế + tín hiệu audit cho deletion/insertion** (kèm robustness ≥1 baseline khác).
7. **[B7] Khung so sánh faithfulness xuyên paradigm** (round-trip trên output baseline) — quyết phần
   experiment có công bằng không.
8. **[B8] Quyết reader study: làm (→ bằng chứng) hay hạ xuống motivation.**

## D. CHỐT SCOPE v1 (để không trôi)
- M3/M4 **thuần ảnh** lúc inference (B11-A). Report/scene-graph = nhãn train.
- Concept-bottleneck đầy đủ, prior-report bất đối xứng (B11-B), privileged-information (B12) → **future work
  có tên rõ**, không nhét vào v1.
- Nhãn yếu (M2-weak, CheXplus, NIH) → **train/pretrain only, eval-never**.
- Dataset eval chính: **MIMIC + ImaGenome người-gán**; NIH = cross-dataset eval cho M3-disease.
