"""Build the thesis evaluation document from the computed JSON results.

Every number in the document is read from a file under DATA, produced on the
training host by the thesis_* scripts, except the handful of RECORDED results whose
checkpoints no longer exist; those are constants below and are labelled as recorded
in the text itself.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from fmt import Thesis, f, sf, ci, ci_plain, pct, n

DATA = sys.argv[1]
OUTPUT = sys.argv[2]

def J(*p):
    return json.load(open(os.path.join(DATA, *p)))

CFG = J("configs.json")
S1 = lambda tag: J("stage1", tag, "metrics.json")["classification"]
AGG = lambda tag: S1(tag)["aggregates"]
PAIR = lambda name: J("stage1", f"paired_{name}.json")
PL = J("stage1", "perlabel_ft.json")
STD = {k: J("standard", f"{k}.json") for k in ("test_main", "stop_probe_25", "cue_rules_100", "pilot_300")}
LEN = J("length_diagnostics.json")
DS = J("data_stats.json")
BLANK_TR = pct(DS["splits"]["train"]["blank_fraction"])
REPO_TEST = {k: J("repo_protocol", f"eval_{k}", "metrics.json")["generation"] for k in ("zeroshot", "armA", "armC")}

LABELS_VI = {
    "No Finding": "Không có bất thường (No Finding)",
    "Enlarged Cardiomediastinum": "Trung thất tim to (Enlarged Cardiomediastinum)",
    "Cardiomegaly": "Tim to (Cardiomegaly)",
    "Lung Opacity": "Mờ phổi (Lung Opacity)",
    "Lung Lesion": "Tổn thương phổi (Lung Lesion)",
    "Edema": "Phù phổi (Edema)",
    "Consolidation": "Đông đặc (Consolidation)",
    "Pneumonia": "Viêm phổi (Pneumonia)",
    "Atelectasis": "Xẹp phổi (Atelectasis)",
    "Pneumothorax": "Tràn khí màng phổi (Pneumothorax)",
    "Pleural Effusion": "Tràn dịch màng phổi (Pleural Effusion)",
    "Pleural Other": "Bất thường màng phổi khác (Pleural Other)",
    "Fracture": "Gãy xương (Fracture)",
    "Support Devices": "Thiết bị hỗ trợ (Support Devices)",
}

# References, in order of first citation.
REFS = [
    'A. Johnson, M. Lungren, Y. Peng, Z. Lu, R. Mark, S. Berkowitz, and S. Horng, "MIMIC-CXR-JPG, chest radiographs with structured labels," version 2.1.0, PhysioNet, 2024, doi: 10.13026/jsn5-t979.',
    'A. E. W. Johnson, T. J. Pollard, S. J. Berkowitz, N. R. Greenbaum, M. P. Lungren, C.-Y. Deng, R. G. Mark, and S. Horng, "MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports," Scientific Data, vol. 6, no. 1, p. 317, 2019.',
    'J. Irvin et al., "CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison," in Proc. AAAI Conference on Artificial Intelligence, vol. 33, 2019.',
    'D. Edirisinghe, W. Nimalsiri, M. Hennayake, D. Meedeniya, and G. Lim, "Chest X-ray report generation using abnormality guided vision language model," IEEE Access, vol. 13, pp. 157651-157673, 2025, doi: 10.1109/ACCESS.2025.3606961.',
    'Z. Chen, Y. Song, T.-H. Chang, and X. Wan, "Generating radiology reports via memory-driven Transformer," in Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020.',
    'S. Bannur et al., "Learning to exploit temporal structure for biomedical vision-language processing," in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.',
    'S. Eslami, C. Meinel, and G. de Melo, "PubMedCLIP: How much does CLIP benefit visual question answering in the medical domain?," in Findings of the Association for Computational Linguistics: EACL, 2023.',
    'J. Li, D. Li, S. Savarese, and S. Hoi, "BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models," in Proc. International Conference on Machine Learning (ICML), 2023.',
    'L. Yang et al., "MedGemma technical report," arXiv:2507.05201, 2025.',
    'T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," in Advances in Neural Information Processing Systems (NeurIPS), 2023.',
    'E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," in Proc. International Conference on Learning Representations (ICLR), 2022.',
    'B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman and Hall, 1993.',
    'K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "BLEU: A method for automatic evaluation of machine translation," in Proc. Annual Meeting of the Association for Computational Linguistics (ACL), 2002.',
    'S. Banerjee and A. Lavie, "METEOR: An automatic metric for MT evaluation with improved correlation with human judgments," in Proc. ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, 2005.',
    'C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in Text Summarization Branches Out, 2004.',
    'R. Vedantam, C. L. Zitnick, and D. Parikh, "CIDEr: Consensus-based image description evaluation," in Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.',
    'T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, "BERTScore: Evaluating text generation with BERT," in Proc. International Conference on Learning Representations (ICLR), 2020.',
    'X. Chen, H. Fang, T.-Y. Lin, R. Vedantam, S. Gupta, P. Dollar, and C. L. Zitnick, "Microsoft COCO captions: Data collection and evaluation server," arXiv:1504.00325, 2015.',
    'Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining approach," arXiv:1907.11692, 2019.',
    'L. Seyyed-Kalantari, G. Liu, M. McDermott, I. Y. Chen, and M. Ghassemi, "CheXclusion: Fairness gaps in deep chest X-ray classifiers," in Proc. Pacific Symposium on Biocomputing (PSB), 2021, arXiv:2003.00827.',
    'Z. Wang, L. Liu, L. Wang, and L. Zhou, "METransformer: Radiology report generation by Transformer with multiple learnable expert tokens," in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, arXiv:2304.02211.',
    'H. Jin, H. Che, Y. Lin, and H. Chen, "PromptMRG: Diagnosis-driven prompts for medical report generation," in Proc. AAAI Conference on Artificial Intelligence, 2024, arXiv:2308.12604.',
    'R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in Proc. IEEE International Conference on Computer Vision (ICCV), 2017.',
    'H. Chefer, S. Gur, and L. Wolf, "Transformer interpretability beyond attention visualization," in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.',
    'J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, "Sanity checks for saliency maps," in Advances in Neural Information Processing Systems (NeurIPS), 2018.',
]
R = {k: i + 1 for i, k in enumerate([
    "jpg", "mimic", "chexpert", "metacxr", "r2gen", "biovilt", "pubmedclip", "blip2", "medgemma",
    "qlora", "lora", "bootstrap", "bleu", "meteor", "rouge", "cider", "bertscore", "coco", "roberta",
    "chexclusion", "metransformer", "promptmrg", "gradcam", "chefer", "adebayo"])}
def c(*keys):
    return "[" + ", ".join(str(R[k]) for k in keys) + "]"

T = Thesis()
T.title("Kết quả thực nghiệm và đánh giá hệ thống META-CXR trên bộ dữ liệu MIMIC-CXR-JPG")
T.para(
    "Tài liệu này trình bày toàn bộ kết quả đánh giá của hệ thống sinh báo cáo X-quang ngực hai giai đoạn "
    "được xây dựng trong đề tài, dựa trên nhánh mã nguồn chính (main, revision 79139fc). Giai đoạn 1 phân loại "
    "14 nhãn bất thường; Giai đoạn 2 sinh phần FINDINGS của báo cáo. Mọi số liệu Giai đoạn 1 được tính lại từ "
    "tệp dự đoán đã lưu và tái lập chính xác đến bốn chữ số thập phân; mọi số liệu Giai đoạn 2 trên tập kiểm "
    "tra được sinh lại sau khi sửa lỗi điều kiện dừng sinh và được chấm lại theo giao thức chuẩn của lĩnh vực "
    "để có thể đặt cạnh các công trình đã công bố. Các thử nghiệm huấn luyện gần nhất không cải thiện kết quả "
    "chỉ được đưa vào như nghiên cứu cắt bỏ, không thay thế mô hình chính.")

# ======================================================================== CH1
T.chapter_heading("Bộ dữ liệu và thiết lập thực nghiệm", new_page=False)
T.section_heading("Bộ dữ liệu MIMIC-CXR-JPG")
T.sub_heading("Nguồn gốc và quan hệ với MIMIC-CXR")
T.para(
    f"Toàn bộ thực nghiệm sử dụng MIMIC-CXR-JPG phiên bản 2.1.0 {c('jpg')}, bộ ảnh X-quang ngực định dạng JPG "
    f"được dẫn xuất từ cơ sở dữ liệu MIMIC-CXR {c('mimic')}. Bộ dữ liệu gồm {n(DS['raw']['metadata_images'])} ảnh thuộc {n(DS['raw']['metadata_studies'])} nghiên cứu của "
    f"{n(DS['raw']['metadata_subjects'])} bệnh nhân, "
    "trong đó mỗi nghiên cứu (study) ứng với một lần chụp, một báo cáo văn bản, và có thể gồm nhiều ảnh ở các tư thế "
    "khác nhau. "
    f"Đi kèm là nhãn có cấu trúc do bộ gán nhãn CheXpert {c('chexpert')} trích xuất tự động từ văn bản báo cáo, "
    "với bốn trạng thái cho mỗi bệnh lý: dương tính (1), âm tính (0), không chắc chắn (-1) và để trống khi báo "
    "cáo không nhắc tới bệnh lý đó.")
T.para(
    f"Theo trang mô tả chính thức {c('jpg')}, MIMIC-CXR-JPG sử dụng chính tệp phân chia chính thức "
    "mimic-cxr-2.0.0-split.csv.gz của MIMIC-CXR. Hệ quả quan trọng cho việc so sánh là các công trình sinh báo cáo "
    "đánh giá trên tập kiểm tra theo phân chia chính thức của MIMIC-CXR đang chấm trên cùng tập nghiên cứu và cùng "
    "văn bản báo cáo với đề tài này; khác biệt nằm ở định dạng lưu ảnh (DICOM so với JPG) chứ không ở tập dữ liệu "
    f"đánh giá. Điều này thể hiện trực tiếp qua số lượng: công trình R2Gen {c('r2gen')} báo cáo tập kiểm định và "
    "kiểm tra gồm 1.808 và 3.269 báo cáo, trùng khớp về số lượng với 1.808 và 3.269 nghiên cứu trong tập đánh giá "
    "Giai đoạn 1 của đề tài. Sự trùng khớp về số lượng là bằng chứng mạnh nhưng không phải phép kiểm tra đồng nhất "
    "từng nghiên cứu, vì các công trình đó không công bố danh sách định danh.")

import ch1_data
DSX = ch1_data.write(T, DS, CFG, c)
T.sub_heading("Vai trò của các tập trong thực nghiệm")
T.table("Vai trò của các tập dữ liệu",
        ["Tập", "Số nghiên cứu", "Vai trò"],
        [["Huấn luyện", n(DS['splits']['train']['studies']), "Huấn luyện Giai đoạn 1"],
         ["Huấn luyện, có FINDINGS hợp lệ", n(CFG['armC']['training_config']['train_samples']), "Huấn luyện Giai đoạn 2"],
         ["Kiểm định", n(DS['splits']['val']['studies']), "Chọn checkpoint, hiệu chuẩn ngưỡng quyết định"],
         ["Kiểm tra", n(DS['splits']['test']['studies']), "Báo cáo kết quả, chỉ dùng một lần sau khi đã cố định mô hình"]],
        widths=[5.0, 3.0, 8.4],
        note="Số nghiên cứu huấn luyện Giai đoạn 2 lấy từ tệp manifest của adapter đã huấn luyện.")

T.sub_heading("Các tập đánh giá")
T.para(
    "Mỗi phép so sánh trong tài liệu được thực hiện trên đúng một tập nghiên cứu cố định, và danh sách định danh "
    "băm (sample_key) của các hệ thống được kiểm tra trùng khớp cả về tập hợp lẫn thứ tự trước khi tính điểm. Tập "
    "kiểm tra của Giai đoạn 2 nhỏ hơn Giai đoạn 1 vì hai lý do: chỉ các nghiên cứu có phần FINDINGS hợp lệ mới có "
    "báo cáo tham chiếu, và hệ thống kết hợp đọc dữ liệu qua đường Giai đoạn 1 trong khi hai hệ thống còn lại đọc "
    "qua manifest ảnh thẳng; tập 2.772 nghiên cứu là giao của hai đường này.")
T.table("Các tập nghiên cứu dùng để đánh giá",
        ["Giai đoạn", "Tập", "Số nghiên cứu", "Mục đích"],
        [["1", "Kiểm định", "1.808", "Hiệu chuẩn ngưỡng quyết định"],
         ["1", "Kiểm tra", "3.269", "Kết quả chính và nghiên cứu cắt bỏ"],
         ["2", "Kiểm tra", n(STD['test_main']['n_studies']), "Kết quả chính của ba hệ thống"],
         ["2", "Kiểm định", n(STD['pilot_300']['n_studies']), "Cắt bỏ kênh gợi ý, token phát hiện, đổi Giai đoạn 1"],
         ["2", "Kiểm định", n(STD['cue_rules_100']['n_studies']), "Cắt bỏ quy tắc tạo gợi ý"],
         ["2", "Kiểm định", n(STD['stop_probe_25']['n_studies']), "Thí nghiệm lỗi điều kiện dừng sinh"]],
        widths=[2.2, 2.6, 3.0, 8.6])

T.section_heading("Môi trường thực nghiệm")
T.table("Phần cứng và phần mềm",
        ["Thành phần", "Cấu hình"],
        [["GPU", "Một NVIDIA GeForce RTX 5060 Ti, 16 GB (15,48 GiB khả dụng cho PyTorch)"],
         ["Hệ điều hành", "Ubuntu 26.04 LTS"],
         ["Python và PyTorch", "Python 3.11.16; PyTorch 2.9.1 với CUDA 12.9"],
         ["Thư viện mô hình", "transformers 4.53.2; PEFT; bitsandbytes"],
         ["Chấm điểm sinh báo cáo", "pycocoevalcap 1.2 (METEOR 1.5 chạy trên OpenJDK 25); bert_score 0.3.12"]],
        widths=[4.5, 11.9])

T.section_heading("Cấu hình huấn luyện")
r1, m1 = CFG["stage1_run"], CFG["stage1_model"]
L = m1["loss"]
T.para(
    "Giai đoạn 1 huấn luyện một lần theo cấu hình được trích trực tiếp từ nhật ký của lần chạy (run_20260820_ft), "
    "không lấy từ tệp cấu hình hiện hành, để bảng dưới đây mô tả đúng mô hình đã được đánh giá.")
T.table("Siêu tham số huấn luyện Giai đoạn 1",
        ["Siêu tham số", "Giá trị"],
        [["Kích thước lô", f"{r1['batch_size_train']} nghiên cứu, tích lũy gradient {r1['accum_grad_iters']} bước (lô hiệu dụng {r1['batch_size_train'] * r1['accum_grad_iters']})"],
         ["Tốc độ học", f"{r1['init_lr']:g} cho mô-đun huấn luyện; {r1['init_lr_enc']:g} cho phần bộ mã hóa được mở khóa; tối thiểu {r1['min_lr']:g}"],
         ["Lịch tốc độ học", f"Khởi động tuyến tính {r1['warmup_steps']} bước cập nhật, sau đó cosine"],
         ["Số epoch", f"{r1['max_epoch']}; bắt đầu chấm kiểm định từ epoch thứ {r1['eval_start_epoch']}"],
         ["Suy giảm trọng số", str(r1['weight_decay']).replace('.', ',')],
         ["Độ chính xác số", "Hỗn hợp bfloat16"],
         ["Tiêu chí chọn checkpoint", "Hàm mất mát trên tập kiểm định; checkpoint tốt nhất là epoch 9"],
         ["Bộ mã hóa ảnh", "BioViL-T (ResNet-50, ảnh 448, 196 token) và PubMedCLIP (ViT-B/32, ảnh 224, 50 token)"],
         ["Mở khóa bộ mã hóa", CFG['stage1_facts']['encoder_ft_line'][0].split(': ', 1)[1].replace('.', ',')],
         ["Thời gian huấn luyện", CFG['stage1_facts']['training_time'][0].replace('Training time ', '') + " (giờ:phút:giây)"],
         ["Bộ nhớ GPU cực đại", f"{n(CFG['stage1_facts']['max_mem'][0])} MiB"]],
        widths=[4.5, 11.9])
T.table("Trọng số các thành phần hàm mất mát Giai đoạn 1",
        ["Thành phần", "Trọng số", "Vai trò"],
        [["Phân loại học viên (student)", f(L['lambda_cls'], 2), "Phân loại 14 nhãn, 3 lớp, chỉ từ ảnh"],
         ["Phân loại giáo viên (teacher)", f(L['lambda_teacher_cls'], 2), "Giáo viên đọc thêm văn bản báo cáo, chỉ khi huấn luyện"],
         ["Chưng cất tri thức", f(L['lambda_distill'], 2), "Truyền phân bố của giáo viên sang học viên"],
         ["Cổng nhắc tới (mention gate)", f(L['lambda_gate'], 2), "Dự đoán báo cáo có nhắc tới bệnh lý hay không"],
         ["Tương phản MHCAC", f(L['lambda_mhcac_contrastive'], 2), "Tách biệt biểu diễn các bệnh lý"],
         ["Trực giao", f(L['lambda_orthogonality'], 2), "Giảm dư thừa giữa các đầu chú ý"],
         ["Thưa", f(L['lambda_sparsity'], 2), "Làm thưa bản đồ chú ý"],
         ["Tương phản đa dương tính", f(L['lambda_mpc'], 2), "Nhất quán giữa các góc nhìn của một nghiên cứu"],
         ["Nhất quán góc nhìn", f(L['lambda_view_consistency'], 2), "Ràng buộc mềm giữa dự đoán hợp nhất và ảnh chính"],
         ["ITC, ITM, LM của Q-Former", f(L['lambda_itc'], 2), "Tắt; bốn lần đo độc lập đều ở mức ngẫu nhiên (Mục 2.6)"],
         ["Giải thích (Grad-CAM)", f(L['lambda_explanation'], 2), "Tắt; thử nghiệm có đối chứng không cải thiện (Mục 2.6)"]],
        widths=[5.0, 2.2, 9.2])
tA, tC = CFG["armA"]["training_config"], CFG["armC"]["training_config"]
T.table("Siêu tham số huấn luyện Giai đoạn 2",
        ["Siêu tham số", "Hệ thống A (ảnh gốc)", "Hệ thống C (kết hợp)"],
        [["Mô hình ngôn ngữ", "MedGemma 1.5 4B-it", "MedGemma 1.5 4B-it"],
         ["Lượng tử hóa và tinh chỉnh", "NF4 4-bit, QLoRA r=16, alpha=32", "NF4 4-bit, QLoRA r=16, alpha=32"],
         ["Đầu vào thị giác", "Tháp ảnh của MedGemma", "Tháp ảnh của MedGemma, 32 token mềm Q-Former, gợi ý P/N/U"],
         ["Lô hiệu dụng", f"{tA['effective_batch_size']} ({tA['batch_size']} x {tA['grad_accum']})", f"{tC['effective_batch_size']} ({tC['batch_size']} x {tC['grad_accum']})"],
         ["Tốc độ học LoRA", f"{tA['lora_lr']:g}", f"{tC['lora_lr']:g}"],
         ["Tốc độ học bộ chiếu", "không áp dụng", f"{tC['projector_lr']:g}"],
         ["Độ dài chuỗi tối đa", str(tA['max_length']), str(tC['max_length'])],
         ["Tham số huấn luyện được", n(tA['parameters']['trainable_parameters']), n(tC['parameters']['trainable_parameters'])],
         ["Mức hoàn thành", f"{f(CFG['armA']['completion']['fraction_of_epoch'], 4)} epoch (dừng do tắt máy theo lịch)", "1 epoch đầy đủ"],
         ["Chọn checkpoint", "Không có (checkpoint cuối cùng)", "Hàm mất mát kiểm định, tốt nhất 1,0019"]],
        widths=[4.2, 5.8, 6.4],
        note=f"Hệ thống A dừng ở {n(CFG['armA']['completion']['updates_done'])} trên {n(CFG['armA']['completion']['updates_planned'])} bước cập nhật và chưa từng chạy kiểm định, nên mọi kết quả của hệ thống A phải được hiểu là kết quả của 0,86 epoch.")

T.section_heading("Nguyên tắc đánh giá")
T.para(
    "Năm nguyên tắc áp dụng cho mọi số liệu trong tài liệu. Thứ nhất, tập kiểm tra chỉ được dùng một lần để báo "
    "cáo; mọi lựa chọn ngưỡng quyết định và chọn checkpoint đều thực hiện trên tập kiểm định. Thứ hai, khoảng tin "
    f"cậy 95% được ước lượng bằng bootstrap {c('bootstrap')} với đơn vị lấy mẫu là nghiên cứu; so sánh giữa hai hệ "
    "thống dùng bootstrap ghép cặp trên cùng một tập nghiên cứu, 2.000 lần lấy mẫu lại. Thứ ba, một chênh lệch được "
    "coi là có ý nghĩa thống kê khi khoảng tin cậy không chứa 0; khi một cận của khoảng nằm cách 0 dưới 0,0005 thì "
    "kết quả được coi là chưa thiết lập được theo cả hai chiều, vì trong thực nghiệm dấu của cận đó thay đổi theo "
    "nhiễu Monte Carlo khi đổi luồng lấy mẫu. Thứ tư, mọi số liệu Giai đoạn 1 được tính lại từ tệp dự đoán đã lưu "
    "bằng mã nguồn nhánh chính và trùng khớp đến bốn chữ số thập phân với kết quả ghi nhận ban đầu. Thứ năm, đề tài "
    "không có bộ chấm lâm sàng đã được kiểm chứng (CheXbert, RadGraph); các mục này được ghi là không có chứ không "
    "ghi bằng 0, và các độ đo từ vựng không được trình bày như độ chính xác lâm sàng.")

# ======================================================================== CH2
def prows(name):
    return {r["metric"]: r for r in PAIR(name)["results"]}

def dcell(r, d=4):
    """Delta with CI; bold only when the CI excludes zero and is not at the zero boundary."""
    if r["ci_lower"] == r["ci_upper"] == r["delta"] == 0:
        # Both sides score the identical probabilities; only thresholds differ, so a
        # threshold-free metric cannot move. Not a boundary result.
        return ("không đổi (cùng điểm số)", False)
    txt = f"{sf(r['delta'], d)} {ci(r['ci_lower'], r['ci_upper'], d)}"
    return (txt, bool(r["ci_excludes_zero"] and not r["at_zero_boundary"]))

def sig_word(r):
    if r["at_zero_boundary"]:
        return "nằm sát biên 0 nên chưa thiết lập được"
    return "có ý nghĩa thống kê" if r["ci_excludes_zero"] else "không có ý nghĩa thống kê"

def baseline(tag, name):
    for b in S1(tag)["baselines"]:
        if b["baseline"] == name:
            return b
    return {}

T.chapter_heading("Giai đoạn 1: phân loại bất thường trên 14 nhãn")
T.section_heading("Kiến trúc được đánh giá")
T.para(
    f"Giai đoạn 1 nhận một nghiên cứu gồm một ảnh chính và tối đa một ảnh phụ. Hai bộ mã hóa ảnh chuyên ngành y tế "
    f"là BioViL-T {c('biovilt')} (nền ResNet-50, ảnh 448 điểm ảnh, lưới 14 nhân 14 gồm 196 token) và PubMedCLIP "
    f"{c('pubmedclip')} (ViT-B/32, ảnh 224 điểm ảnh, một token toàn cục và 49 token vùng) được giữ nguyên thang đo "
    "gốc; đầu ra của chúng được hợp nhất góc nhìn theo từng bộ mã hóa, chiếu về cùng chiều 1.408 và nối lại thành "
    "chuỗi 246 token. Mô-đun phân loại chú ý chéo đa đầu (MHCAC) dự đoán cho mỗi nhãn một phân bố ba lớp dương tính, "
    "âm tính, không chắc chắn, có điều kiện là báo cáo nhắc tới bệnh lý đó; song song, một đầu cổng nhắc tới dự đoán "
    "xác suất báo cáo có nhắc tới bệnh lý. Khi huấn luyện, một nhánh giáo viên đọc thêm văn bản báo cáo và chưng cất "
    "phân bố của nó sang nhánh học viên chỉ dùng ảnh; khi suy luận chỉ nhánh học viên được dùng. Phần Q-Former của "
    f"BLIP-2 {c('blip2')} vẫn hiện diện nhưng các mục tiêu căn chỉnh ảnh văn bản (ITC, ITM, LM) được tắt vì lý do "
    "trình bày ở Mục 2.6.")

T.section_heading("Độ đo đánh giá")
T.para(
    "Mỗi nhãn được đánh giá như một bài toán nhị phân dương tính so với không dương tính. Các độ đo không phụ thuộc "
    "ngưỡng gồm diện tích dưới đường cong ROC (AUROC) và diện tích dưới đường cong precision recall (AUPRC). Các độ "
    "đo phụ thuộc ngưỡng gồm precision, recall (độ nhạy), F1 và specificity (độ đặc hiệu), tính tại ngưỡng đã hiệu "
    "chuẩn trên tập kiểm định. Trung bình macro lấy trung bình theo nhãn với trọng số bằng nhau; trung bình micro gộp "
    "mọi cặp nghiên cứu và nhãn thành một bảng xếp hạng duy nhất. Hai nhãn tổng hợp No Finding và Support Devices "
    "luôn được báo cáo riêng theo từng nhãn nhưng không tham gia các trung bình macro phụ thuộc ngưỡng. AUROC macro "
    "là độ đo chính của đề tài vì nó không bị ảnh hưởng bởi lựa chọn ngưỡng.")

T.section_heading("Hai cách diễn giải ma trận nhãn")
T.para(
    f"Vì {BLANK_TR} ô nhãn của tập huấn luyện để trống, cùng một tệp dự đoán có thể được chấm theo hai cách cho kết quả rất khác nhau, và "
    "việc không nêu rõ cách chấm là nguồn gốc phổ biến của các so sánh không hợp lệ. Cách thứ nhất, phân cực có che "
    "(masked polarity), bỏ qua ô trống và chỉ hỏi phân cực của bệnh lý khi báo cáo đã nhắc tới nó. Cách thứ hai, hiện "
    "diện theo nghiên cứu (study presence), coi ô trống, âm tính và không chắc chắn đều là không hiện diện, tức trả "
    "lời câu hỏi lâm sàng nghiên cứu này có bệnh lý hay không. Với cách thứ hai, điểm dự đoán có thể là xác suất "
    "dương tính có điều kiện (q) hoặc điểm hiện diện biên, bằng tích của xác suất được nhắc tới với xác suất dương "
    "tính có điều kiện; điểm biên đúng với câu hỏi được đặt ra vì phân bố q chỉ có nghĩa khi bệnh lý đã được nhắc tới.")
fm = [("Phân cực có che, điểm q", "ft_masked_q"), ("Hiện diện, điểm q", "ft_presence_q"), ("Hiện diện, điểm biên (chính)", "ft_main")]
rows = []
for label, tag in fm:
    a = AGG(tag)
    rows.append([label, f(a["macro_auroc"]), f(a["micro_auroc"]), f(a["positive_macro_f1"]),
                 f(a["positive_macro_precision"]), f(a["positive_macro_recall"]), f(a["macro_specificity"])])
for label, tag in fm[:1] + fm[2:]:
    for bname, bvi in (("all_positive", "luôn dương tính"), ("majority_class", "lớp đa số")):
        b = baseline(tag, bname)
        if b:
            rows.append([f"Baseline {bvi}, {label.split(',')[0].lower()}", "không áp dụng", "không áp dụng",
                         f(b["positive_macro_f1"]), "không áp dụng", f(b["positive_macro_recall"]), "không áp dụng"])
T.table("Cùng một mô hình chấm theo ba cách, kèm các bộ dự đoán tầm thường (tập kiểm tra, 3.269 nghiên cứu)",
        ["Cách chấm", "AUROC macro", "AUROC micro", "F1 macro", "Precision", "Recall", "Specificity"],
        rows, widths=[5.4, 1.9, 1.9, 1.7, 1.8, 1.7, 2.0], font_size=10,
        note="Mỗi cách chấm dùng ngưỡng hiệu chuẩn riêng trên tập kiểm định theo đúng cách chấm đó. Bộ dự đoán tầm thường không có điểm liên tục nên không có AUROC.")
mq, mm = AGG("ft_masked_q"), AGG("ft_main")
ap_m = baseline("ft_masked_q", "all_positive")
T.para(
    f"Bảng trên cho thấy lý do đề tài chỉ báo cáo F1 theo cách hiện diện. Theo cách phân cực có che, mô hình đạt F1 "
    f"{f(mq['positive_macro_f1'])}, nhưng bộ dự đoán luôn trả lời dương tính đã đạt {f(ap_m.get('positive_macro_f1'))}, "
    f"nên mô hình chỉ hơn một hằng số {f(mq['positive_macro_f1'] - ap_m.get('positive_macro_f1', 0))}. Nguyên nhân là việc "
    "che ô trống đảo ngược tỉ lệ lớp: trong các ô được nhắc tới, dương tính chiếm đa số ở phần lớn các nhãn. Theo "
    f"cách hiện diện, cả hai bộ dự đoán tầm thường về F1 bằng 0, và F1 {f(mm['positive_macro_f1'])} của mô hình là "
    "phần kỹ năng thực. Do đó mọi con số F1 khoảng 0,8 đến 0,9 trên MIMIC-CXR cần được đọc cùng cách chấm và cùng "
    "baseline tầm thường của chính cách chấm đó.")

T.section_heading("Hiệu chuẩn ngưỡng quyết định")
T.para(
    "Ngưỡng của từng nhãn được chọn trên tập kiểm định để tối đa F1, rồi áp dụng nguyên vẹn lên tập kiểm tra. Đề tài "
    "dùng quy tắc bình nguyên: thay vì lấy ngưỡng tại đỉnh F1, ngưỡng được đặt tại trung vị của mọi ngưỡng đạt ít "
    "nhất 95% giá trị đỉnh, vì đỉnh trên 1.808 nghiên cứu phụ thuộc vào vài nghiên cứu ngẫu nhiên nằm quanh nó còn "
    "vùng gần tối ưu thì chuyển giao tốt hơn. Nhãn có ít nhất 5 ca dương tính trên tập kiểm định mới được hiệu chuẩn; "
    "nhãn ít hơn giữ ngưỡng 0,5. Tác động của hai lựa chọn này được đo ở Mục 2.6.")

T.section_heading("Kết quả chính")
iv = S1("ft_main")["intervals"]
def ivc(k):
    x = iv[k]; return ci_plain(x["lower"], x["upper"])
T.table("Kết quả phân loại Giai đoạn 1 trên tập kiểm tra (hiện diện, điểm biên, 3.269 nghiên cứu)",
        ["Độ đo", "Giá trị", "Khoảng tin cậy 95%"],
        [["AUROC macro", f(mm["macro_auroc"]), ivc("macro_auroc")],
         ["AUROC micro", f(mm["micro_auroc"]), "không tính"],
         ["AUPRC macro", f(mm["macro_auprc"]), ivc("macro_auprc")],
         ["F1 macro (dương tính)", f(mm["positive_macro_f1"]), ivc("positive_macro_f1")],
         ["Precision macro", f(mm["positive_macro_precision"]), ivc("positive_macro_precision")],
         ["Recall macro", f(mm["positive_macro_recall"]), ivc("positive_macro_recall")],
         ["Specificity macro", f(mm["macro_specificity"]), "không tính"]],
        widths=[6.0, 3.0, 5.5],
        note="Khoảng tin cậy bootstrap theo nghiên cứu, 1.000 lần lấy mẫu. Kết quả trùng khớp đến bốn chữ số với lần đánh giá ban đầu của cùng checkpoint.")
rows = []
for r in PL["per_label_study_presence_marginal"]:
    rows.append([LABELS_VI.get(r["label"], r["label"]), n(r["support_positive"]), pct(r["prevalence"]),
                 f"{f(r['auroc'])} {ci_plain(r['auroc_lo'], r['auroc_hi'])}", f(r["auprc"]),
                 f(r["precision"]), f(r["recall"]), f(r["f1"]), f(r["specificity"])])
T.table("Kết quả theo từng nhãn trên tập kiểm tra",
        ["Nhãn", "Số ca dương", "Tỉ lệ", "AUROC [KTC 95%]", "AUPRC", "Precision", "Recall", "F1", "Specificity"],
        rows, widths=[3.5, 1.2, 1.4, 3.1, 1.3, 1.4, 1.3, 1.3, 1.5], font_size=9,
        note="KTC: khoảng tin cậy, bootstrap theo nghiên cứu 1.000 lần. Tỉ lệ là tỉ lệ dương tính theo cách hiện diện trên tập kiểm tra.")
pl = sorted(PL["per_label_study_presence_marginal"], key=lambda r: r["auroc"])
T.para(
    f"AUROC macro đạt {f(mm['macro_auroc'])}. Nhãn phân biệt tốt nhất là {LABELS_VI[pl[-1]['label']]} "
    f"({f(pl[-1]['auroc'])}) và {LABELS_VI[pl[-2]['label']]} ({f(pl[-2]['auroc'])}); nhãn khó nhất là "
    f"{LABELS_VI[pl[0]['label']]} ({f(pl[0]['auroc'])}) và {LABELS_VI[pl[1]['label']]} ({f(pl[1]['auroc'])}). "
    "Các nhãn hiếm có khoảng tin cậy rộng và F1 dao động mạnh theo ngưỡng, vì vậy khi so sánh theo nhãn nên ưu tiên "
    "AUROC kèm tỉ lệ dương tính thay vì F1.")

T.section_heading("Nghiên cứu cắt bỏ")
T.sub_heading("Các thành phần được đo lại có đối chứng")
T.para(
    "Mỗi dòng trong hai bảng dưới đây so sánh hai lần chấm trên cùng 3.269 nghiên cứu kiểm tra, mỗi bên dùng ngưỡng "
    "hiệu chuẩn trên tập kiểm định của chính nó, và chỉ khác nhau đúng một thành phần. Cấu hình của các lần chạy "
    "được đối chiếu từng khóa từ nhật ký huấn luyện trước khi so sánh. Chênh lệch được tính là hệ thống sau trừ hệ "
    "thống trước; giá trị in đậm là khoảng tin cậy không chứa 0 và không nằm sát biên. Bảng thứ nhất báo cáo các độ "
    "đo không phụ thuộc ngưỡng, phản ánh khả năng phân biệt của mô hình; bảng thứ hai báo cáo các độ đo tại ngưỡng, "
    "phản ánh điểm vận hành.")
ABL = [("unfreeze", "Mở khóa phần trên bộ mã hóa (đóng băng sang mở khóa nông)"),
       ("depth", "Mở khóa sâu hơn (nông sang sâu)"),
       ("hier_shallow", "Hàm mất mát phân cấp, cấu hình nông"),
       ("hier_deep", "Hàm mất mát phân cấp, cấu hình sâu"),
       ("gate", "Dùng cổng nhắc tới khi chấm (điểm q sang điểm biên)"),
       ("plateau", "Chọn ngưỡng theo bình nguyên thay vì đỉnh"),
       ("minpos", "Hạ số ca dương tối thiểu để hiệu chuẩn từ 20 xuống 5")]
rows1, rows2 = [], []
for key, label in ABL:
    r = prows(key)
    rows1.append([label, dcell(r["macro_auroc"]), dcell(r["micro_auroc"]), dcell(r["macro_auprc"])])
    rows2.append([label, dcell(r["positive_macro_f1"]), dcell(r["positive_macro_precision"]),
                  dcell(r["positive_macro_recall"]), dcell(r["macro_specificity"])])
T.table("Chênh lệch khả năng phân biệt (tập kiểm tra, bootstrap ghép cặp 2.000 lần)",
        ["Thành phần thay đổi", "AUROC macro", "AUROC micro", "AUPRC macro"],
        rows1, widths=[6.4, 3.4, 3.3, 3.3], font_size=9.5)
T.table("Chênh lệch tại điểm vận hành (tập kiểm tra, bootstrap ghép cặp 2.000 lần)",
        ["Thành phần thay đổi", "F1 macro", "Precision", "Recall", "Specificity"],
        rows2, widths=[5.0, 2.9, 2.9, 2.9, 2.8], font_size=9,
        note="Dòng mở khóa bộ mã hóa đồng thời đổi hệ số kappa của trọng số lớp về 1, nên hai yếu tố này không tách được trong phép so sánh này.")

ru, rd, rhs, rg = prows("unfreeze"), prows("depth"), prows("hier_shallow"), prows("gate")
T.para(
    f"Trong các thay đổi về huấn luyện, việc mở khóa phần trên của hai bộ mã hóa là thay đổi duy nhất cải thiện khả năng phân biệt: AUROC macro thay "
    f"đổi {sf(ru['macro_auroc']['delta'])} và kết quả {sig_word(ru['macro_auroc'])}. Lập luận gián tiếp để quy tác "
    "động cho việc mở khóa thay vì hệ số kappa là kappa chỉ dịch chuyển điểm vận hành, vốn đã được ngưỡng hiệu chuẩn "
    "hấp thụ, còn AUROC là đại lượng kappa không thể thay đổi về nguyên tắc; tuy vậy đây là suy luận chứ chưa phải "
    f"phép cắt bỏ riêng. Mở khóa sâu hơn cho AUROC macro {sf(rd['macro_auroc']['delta'])} ({sig_word(rd['macro_auroc'])}), "
    "trong khi các thay đổi lớn nằm ở recall và specificity, là dấu hiệu điểm vận hành dịch chuyển chứ không phải "
    "mô hình tốt hơn.")
T.para(
    f"Dùng cổng nhắc tới khi chấm, tức thay xác suất dương tính có điều kiện bằng điểm hiện diện biên, thay đổi AUROC "
    f"macro {sf(rg['macro_auroc']['delta'])} ({sig_word(rg['macro_auroc'])}). Đây là bằng chứng rằng đầu cổng nhắc "
    f"tới, vốn là thành phần duy nhất học từ {BLANK_TR} ô nhãn để trống, mang thông tin mà nhánh phân loại ba lớp không có.")
T.para(
    f"Hàm mất mát phân cấp, huấn luyện chung xác suất nhắc tới và phân bố phân cực thành một hàm hợp lý duy nhất, cho "
    f"AUROC macro {sf(rhs['macro_auroc']['delta'])} ({sig_word(rhs['macro_auroc'])}) nhưng AUROC micro "
    f"{sf(rhs['micro_auroc']['delta'])} ({sig_word(rhs['micro_auroc'])}), và kết quả cùng dạng lặp lại trên cấu hình "
    "sâu. AUROC micro gộp mọi nhãn vào một bảng xếp hạng nên nhạy với việc điểm số có cùng ý nghĩa giữa các nhãn hay "
    "không, còn AUROC theo từng nhãn không đổi dưới mọi phép co giãn đơn điệu riêng của từng nhãn. Vì vậy hàm mất mát "
    "phân cấp cải thiện tính hiệu chuẩn giữa các nhãn chứ không cải thiện khả năng phân biệt từng nhãn; trong một "
    "quy trình đã hiệu chuẩn ngưỡng riêng cho từng nhãn, tính chất này không mở rộng tập điểm vận hành đạt được.")
rpl, rmp = prows("plateau"), prows("minpos")
def _pp(tag):
    return {r["pathology"]: r for r in S1(tag)["per_pathology"]}
_mp20, _mp5 = _pp("ft_plateau_mp20"), _pp("ft_main")
_stuck = [k for k, r in _mp20.items() if r["threshold"] == 0.5 and r["tp"] == 0 and r["fp"] == 0]
rare_sentence = (
    " và ".join(LABELS_VI[k] for k in _stuck) + " giữ ngưỡng mặc định 0,5 và không được dự đoán dương tính ở bất kỳ "
    "nghiên cứu kiểm tra nào; khi hạ xuống 5 ca, " +
    " và ".join(f"{LABELS_VI[k]} nhận ngưỡng {f(_mp5[k]['threshold'], 3)} với recall {f(_mp5[k]['recall'])}" for k in _stuck)
) if _stuck else "không nhãn nào bị giữ ngưỡng mặc định"
T.para(
    f"Hai lựa chọn về ngưỡng tách được tác động của nhau. Chọn ngưỡng theo bình nguyên thay vì đỉnh thay đổi F1 macro "
    f"{sf(rpl['positive_macro_f1']['delta'])} ({sig_word(rpl['positive_macro_f1'])}), còn hạ số ca dương tối thiểu "
    f"để hiệu chuẩn từ 20 xuống 5 thay đổi F1 macro {sf(rmp['positive_macro_f1']['delta'])} "
    f"({sig_word(rmp['positive_macro_f1'])}). Phần cải thiện F1 của quy trình hiệu chuẩn vì vậy đến từ việc cho phép "
    f"hiệu chuẩn các nhãn hiếm: với ngưỡng tối thiểu 20 ca, {rare_sentence}. "
    f"Cả hai lựa chọn đều đổi precision lấy recall (precision {sf(rmp['positive_macro_precision']['delta'])}, recall "
    f"{sf(rmp['positive_macro_recall']['delta'])} với lựa chọn thứ hai), và cả hai giữ nguyên AUROC vì chỉ ngưỡng thay "
    "đổi còn điểm số của mô hình thì không.")

T.sub_heading("Các thành phần chỉ còn kết quả ghi nhận")
T.para(
    "Ba thí nghiệm cắt bỏ khác được thực hiện trong quá trình phát triển nhưng không thể chấm lại, vì checkpoint của "
    "chúng bị mất khi máy huấn luyện được cài đặt lại hoặc tệp dự đoán tương ứng không được lưu. Kết quả dưới đây là "
    "số liệu ghi nhận tại thời điểm thí nghiệm, được trình bày để đầy đủ và không có cùng mức bảo đảm tái lập như Mục "
    "2.6.1. Riêng thí nghiệm hàm mất mát giải thích được chấm theo cách phân cực có che với bộ mã hóa đóng băng, nên "
    "không so trực tiếp được với các bảng khác trong chương này.")
T.table(f"Hàm mất mát giải thích dựa trên Grad-CAM {c('gradcam')}, 5 epoch, bật so với tắt (số liệu ghi nhận)",
        ["Độ đo", "Bật (0,05 và 0,25)", "Tắt"],
        [["F1 macro (phân cực có che)", "0,8757", "0,8767"],
         ["AUROC macro", "0,7850", "0,7879"],
         ["Specificity macro", "0,3840", "0,4127"],
         ["Thời gian huấn luyện", "5 giờ 06 phút 44 giây", "4 giờ 25 phút 27 giây"]],
        widths=[6.0, 5.0, 5.4],
        note="Mọi khoảng tin cậy 95% chồng lấn; mọi chênh lệch nghiêng về phía tắt; chi phí huấn luyện tăng 15,5%. Độ chính xác bản đồ nổi bật ở cả hai nhánh đều xấp xỉ mức ngẫu nhiên bằng tỉ lệ diện tích mặt nạ.")
T.table("Cổng kiểm tra học tương phản ảnh văn bản của Q-Former trên tập kiểm định (số liệu ghi nhận)",
        ["Lần đo", "Nhiệt độ", "Hạng trung bình ảnh sang văn bản", "Hạng trung bình văn bản sang ảnh", "Chênh lệch so với ngẫu nhiên (nats)"],
        [["Chưa huấn luyện", "0,07 cố định", "130,68", "130,30", "-0,0833"],
         ["525 bước cập nhật", "0,00796 học được", "127,43", "127,65", "-1,1168"],
         ["500 bước cập nhật", "0,07 cố định", "128,38", "127,45", "-0,0025"],
         ["Khoảng 525 bước, lần đo thứ tư", "0,00944 học được", "127,49", "127,76", "-0,0118"]],
        widths=[3.8, 2.8, 3.3, 3.3, 3.2], font_size=9.5,
        note="256 cặp hợp lệ, hạng kỳ vọng khi ngẫu nhiên là 127,5; ngưỡng đạt yêu cầu là chênh lệch tối thiểu +0,10 nats. Cả bốn lần đều ở mức ngẫu nhiên.")
T.table("Huấn luyện thêm 5 epoch (epoch 14 so với epoch 9, tập kiểm tra, số liệu ghi nhận)",
        ["Độ đo", "Chênh lệch", "Kết luận"],
        [["AUROC macro", "-0,0005 [-0,0043; +0,0031]", "Không có ý nghĩa"],
         ["F1 macro", "-0,0067", "Khoảng tin cậy chứa 0"],
         ["Precision macro", "+0,0015", "Khoảng tin cậy chứa 0"],
         ["Recall macro", "-0,0826", "Có ý nghĩa, do ngưỡng dịch chuyển"],
         ["Specificity macro", "+0,0244", "Có ý nghĩa, do ngưỡng dịch chuyển"]],
        widths=[4.5, 6.0, 5.9],
        note="AUROC cải thiện ở 6 trên 14 nhãn. Trên tập kiểm định, AUROC macro tăng +0,0082 [+0,0015; +0,0148] nhưng không lặp lại trên tập kiểm tra.")
T.para(
    "Kết quả Q-Former ở mức ngẫu nhiên phù hợp với giải thích từ tài liệu: BLIP-2 huấn luyện học tương phản với lô "
    f"1.680 đến 2.320 mẫu {c('blip2')}, trong khi cấu hình khả thi trên một GPU 16 GB của đề tài chỉ có lô 8. Mã nguồn "
    f"tham chiếu của bài báo gốc {c('metacxr')} cũng không huấn luyện các mục tiêu này. Vì vậy các trọng số ITC, ITM, "
    "LM được đặt bằng 0 trong mô hình chính, và 32 token mềm của Q-Former là một phép đọc cố định, chưa được thích "
    "nghi, của đặc trưng ảnh.")

T.section_heading("So sánh với công trình liên quan")
cx = {"Lung Opacity": 0.782, "Atelectasis": 0.837, "Cardiomegaly": 0.828, "Consolidation": 0.844, "Edema": 0.904,
      "Pleural Effusion": 0.933, "Enlarged Cardiomediastinum": 0.757, "Fracture": 0.718, "Lung Lesion": 0.772,
      "No Finding": 0.868, "Pleural Other": 0.848, "Pneumonia": 0.748, "Pneumothorax": 0.903, "Support Devices": 0.927}
rows = []
ours = {r["label"]: r for r in PL["per_label_study_presence_marginal"]}
wins = 0
for lab in ours:
    o = ours[lab]; d = o["auroc"] - cx[lab]; wins += d > 0
    rows.append([LABELS_VI[lab], f(cx[lab], 3), f"{f(o['auroc'])} {ci_plain(o['auroc_lo'], o['auroc_hi'])}", sf(d)])
cx_mean = sum(cx.values()) / len(cx)
rows.append(["Trung bình 14 nhãn", "0,834", f(sum(o["auroc"] for o in ours.values()) / len(ours)),
             sf(sum(o["auroc"] for o in ours.values()) / len(ours) - 0.834)])
T.table(f"AUROC theo nhãn: CheXclusion {c('chexclusion')} và đề tài",
        ["Nhãn", "CheXclusion", "Đề tài [KTC 95%]", "Chênh lệch"],
        rows, widths=[6.0, 2.6, 5.0, 2.8], font_size=9.5,
        note="Số của CheXclusion lấy từ Bảng 2 của công trình, cột MIMIC-CXR, trung bình 5 lần chạy. Trung bình 14 nhãn của đề tài ở đây gồm cả hai nhãn tổng hợp để khớp cách tính của CheXclusion, nên khác AUROC macro chính.")
T.table("Khác biệt giao thức giữa CheXclusion và đề tài",
        ["Khía cạnh", "CheXclusion", "Đề tài"],
        [["Dữ liệu", "MIMIC-CXR, 371.858 ảnh", "MIMIC-CXR-JPG 2.1.0"],
         ["Chính sách nhãn", "Mọi nhãn không dương tính là âm tính", "Hiện diện theo nghiên cứu, tương đương"],
         ["Đơn vị đánh giá", "Ảnh", "Nghiên cứu (ảnh chính và tối đa một ảnh phụ)"],
         ["Phân chia", "Ngẫu nhiên 80, 10, 10 theo bệnh nhân", "Phân chia chính thức"],
         ["Mô hình", "DenseNet-121 khởi tạo ImageNet", "BioViL-T và PubMedCLIP, hợp nhất góc nhìn, MHCAC"],
         ["Chọn ngưỡng", "Tối đa F1 trên tập kiểm định", "Bình nguyên F1 trên tập kiểm định"],
         ["Số lần chạy", "5 hạt giống, báo cáo trung bình", "1 lần chạy"]],
        widths=[3.6, 6.0, 6.8], font_size=10)
T.para(
    f"Đề tài có AUROC thấp hơn CheXclusion ở {len(ours) - wins} trên {len(ours)} nhãn, chênh lệch trung bình "
    f"{sf(sum(o['auroc'] for o in ours.values()) / len(ours) - 0.834, 3)}. Chính sách nhãn của hai bên tương đương, "
    "nhưng ba khác biệt giao thức khiến hai bộ số không phải phép so sánh ngang hàng: đơn vị ảnh so với nghiên cứu, "
    "phân chia ngẫu nhiên so với phân chia chính thức, và trung bình 5 lần chạy so với một lần chạy. Kết luận trung "
    "thực là trên cùng câu hỏi hiện diện bệnh lý, bộ phân loại của đề tài chưa đạt mức của một đường cơ sở CNN chuyên "
    "phân loại; đóng góp của Giai đoạn 1 nằm ở việc cung cấp đầu ra có cấu trúc và cổng nhắc tới cho Giai đoạn 2, "
    "không phải ở độ chính xác phân loại vượt trội.")

# ======================================================================== CH3
def spair(group, comparison):
    for p in STD[group]["pairs"]:
        if p["comparison"] == comparison:
            return p
    raise KeyError(f"{group}: {comparison}")

SYS_VI = {"zeroshot": "MedGemma không tinh chỉnh", "armA": "Hệ thống A (ảnh gốc, tinh chỉnh)",
          "armC": "Hệ thống C (kết hợp, tinh chỉnh)"}

T.chapter_heading("Giai đoạn 2: sinh báo cáo")
T.section_heading("Các hệ thống được đánh giá")
T.para(
    f"Giai đoạn 2 dùng mô hình ngôn ngữ thị giác MedGemma 1.5 4B {c('medgemma')} tinh chỉnh bằng QLoRA "
    f"{c('qlora', 'lora')} chỉ trên tháp ngôn ngữ, sinh phần FINDINGS của báo cáo. Ba hệ thống được so sánh trên cùng "
    "tập kiểm tra. Hệ thống không tinh chỉnh là MedGemma nguyên bản, dùng làm mốc đối chứng. Hệ thống A tinh chỉnh "
    "MedGemma chỉ với ảnh gốc qua tháp ảnh của chính nó, không dùng bất kỳ đầu ra nào của Giai đoạn 1. Hệ thống C là "
    "kiến trúc thiết kế ban đầu của đề tài: MedGemma giữ tháp ảnh của mình, đồng thời nhận thêm 32 token mềm từ "
    "Q-Former được chiếu vào không gian nhúng qua một lớp tuyến tính huấn luyện được, và nhận các gợi ý dương tính, "
    "âm tính, không chắc chắn từ Giai đoạn 1 dưới dạng văn bản. Hệ thống C được suy luận với đúng quy tắc tạo gợi ý "
    "mà nó đã được huấn luyện.")

T.section_heading("Cấu hình giải mã")
T.para(
    "Cả ba hệ thống dùng giải mã tham lam, tối đa 160 token mới, không dùng hình phạt lặp và không chặn n-gram lặp. "
    "Điều kiện dừng gồm cả hai mã kết thúc mà cấu hình sinh của MedGemma khai báo: mã kết thúc chuỗi (1) và mã kết "
    "thúc lượt hội thoại end_of_turn (106). Chi tiết này quyết định kết quả và được phân tích ở Mục 3.5: trước khi sửa, "
    "mã nguồn chỉ dừng ở mã 1, trong khi báo cáo đích dùng khi huấn luyện kết thúc bằng mã 106.")

T.section_heading("Độ đo và giao thức chấm điểm")
T.para(
    f"Các độ đo từ vựng gồm BLEU-1 đến BLEU-4 {c('bleu')}, METEOR {c('meteor')}, ROUGE-L {c('rouge')} và CIDEr "
    f"{c('cider')}, tính bằng bộ công cụ coco-caption {c('coco')} (thư viện pycocoevalcap) sau khi tách từ bằng "
    f"PTBTokenizer; đây là giao thức mà R2Gen {c('r2gen')} và các công trình kế tiếp sử dụng. METEOR là phiên bản "
    f"1.5 chạy trên Java. Độ tương đồng ngữ nghĩa được đo bằng BERTScore {c('bertscore')} với mô hình RoBERTa-large "
    f"{c('roberta')}, mặc định của thư viện bert_score cho tiếng Anh, ở cả dạng thô và dạng đã co giãn theo đường "
    "cơ sở; hai dạng này khác nhau rất xa về giá trị tuyệt đối nên luôn được ghi rõ. Khoảng tin cậy của METEOR, "
    "ROUGE-L, CIDEr và BERTScore được tính trên trung bình điểm từng nghiên cứu; của BLEU-4 được tính bằng cách tính "
    "lại BLEU cấp kho văn bản trên mỗi mẫu bootstrap, vì BLEU không phải trung bình điểm từng câu.")
T.para(
    "Bộ chấm nội bộ của mã nguồn tính METEOR bằng thư viện nltk, BLEU bằng cài đặt riêng có làm trơn cộng một, và "
    "BERTScore bằng DistilBERT dạng thô. Các lựa chọn này hợp lệ cho so sánh nội bộ nhưng cho giá trị khác hẳn giao "
    "thức chuẩn, nên không được dùng để so với công trình khác. Bảng dưới đây đặt hai giao thức cạnh nhau trên cùng các "
    "báo cáo sinh ra, để minh họa mức chênh lệch.")
rows = []
for k in ("zeroshot", "armA", "armC"):
    rp, st = REPO_TEST[k]["corpus"], STD["test_main"]["results"][k]
    rows.append([SYS_VI[k], f(rp["meteor"]), f(st["meteor"]), f(rp["bleu_4"]), f(st["bleu_4"]),
                 f(rp["bertscore_f1"]), f(st["bertscore_f1_roberta_large_raw"])])
T.table("Cùng báo cáo sinh ra, chấm bằng bộ chấm nội bộ và bằng giao thức chuẩn (tập kiểm tra)",
        ["Hệ thống", "METEOR nltk", "METEOR 1.5", "BLEU-4 nội bộ", "BLEU-4 coco", "BERTScore DistilBERT", "BERTScore RoBERTa-L"],
        rows, widths=[4.2, 1.8, 1.8, 1.9, 1.8, 2.5, 2.4], font_size=9.5)

T.section_heading("Kết quả trên tập kiểm tra")
tm = STD["test_main"]["results"]
rows = []
for k in ("zeroshot", "armA", "armC"):
    r = tm[k]
    rows.append([SYS_VI[k], f(r["bleu_1"]), f(r["bleu_2"]), f(r["bleu_3"]), f(r["bleu_4"]), f(r["meteor"]),
                 f(r["rouge_l"]), f(r["cider"]), f(r["bertscore_f1_roberta_large_raw"]),
                 f(r["bertscore_f1_roberta_large_rescaled"])])
T.table(f"Kết quả sinh báo cáo trên tập kiểm tra ({n(STD['test_main']['n_studies'])} nghiên cứu, giao thức chuẩn)",
        ["Hệ thống", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "BERTS thô", "BERTS co giãn"],
        rows, widths=[3.1, 1.3, 1.3, 1.3, 1.3, 1.75, 1.75, 1.35, 1.5, 1.75], font_size=9,
        note="BERTS: BERTScore F1 với RoBERTa-large. Độ đo lâm sàng (CheXbert, RadGraph) không có trong đề tài và không được thay bằng giá trị 0.")
rows = []
for k in ("zeroshot", "armA", "armC"):
    r = tm[k]
    rows.append([SYS_VI[k], f(r["gen_median_words"], 0), f(r["ref_median_words"], 0), pct(r["empty_reports"]),
                 pct(r["repeated_5gram_3x"])])
T.table("Độ dài và hiện tượng lặp của báo cáo sinh ra trên tập kiểm tra",
        ["Hệ thống", "Trung vị số từ sinh ra", "Trung vị số từ tham chiếu", "Báo cáo rỗng", "Có 5-gram lặp từ 3 lần"],
        rows, widths=[4.6, 3.0, 3.0, 2.6, 3.2], font_size=10)
pa, pc, pac = spair("test_main", "armA - zeroshot"), spair("test_main", "armC - zeroshot"), spair("test_main", "armC - armA")
rows = []
for label, p in (("Hệ thống A trừ không tinh chỉnh", pa), ("Hệ thống C trừ không tinh chỉnh", pc), ("Hệ thống C trừ hệ thống A", pac)):
    rows.append([label, dcell(p["bleu_4"]), dcell(p["meteor"]), dcell(p["rouge_l"]), dcell(p["cider"]),
                 dcell(p["bertscore_raw"])])
T.table("Chênh lệch ghép cặp trên tập kiểm tra (bootstrap 2.000 lần; BLEU-4 500 lần)",
        ["So sánh", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "BERTScore thô"],
        rows, widths=[3.8, 2.6, 2.5, 2.5, 2.5, 2.5], font_size=9,
        note="Giá trị in đậm là khoảng tin cậy không chứa 0 và không nằm sát biên.")
rA, rZ, rC = tm["armA"], tm["zeroshot"], tm["armC"]
def cmp_sentence(p, metric, name):
    r = p[metric]
    return f"{name} {sf(r['delta'])} ({sig_word(r)})"
T.para(
    "So với MedGemma không tinh chỉnh, hệ thống A thay đổi " + cmp_sentence(pa, "cider", "CIDEr") + ", "
    + cmp_sentence(pa, "meteor", "METEOR") + " và " + cmp_sentence(pa, "bertscore_raw", "BERTScore thô")
    + ". Hệ thống C so với hệ thống A thay đổi " + cmp_sentence(pac, "cider", "CIDEr") + " và "
    + cmp_sentence(pac, "bertscore_raw", "BERTScore thô") + f". Trung vị độ dài báo cáo sinh ra của hệ thống A là "
    f"{f(rA['gen_median_words'], 0)} từ so với {f(rA['ref_median_words'], 0)} từ của báo cáo tham chiếu. Báo cáo "
    "ngắn hơn đáng kể so với tham chiếu làm giảm BLEU qua hệ số phạt độ ngắn và giảm METEOR vốn thưởng cho độ phủ, "
    "nên độ ngắn là một hạn chế cần được đọc cùng các độ đo này.")

T.sub_heading("Hiện tượng sụp về mẫu câu ngắn")
rows = []
for k in ("zeroshot", "armA", "armC"):
    d = LEN[k]
    rows.append([SYS_VI[k], f(d["median_words"], 0), pct(d["share_le10"]), pct(d["share_le5"]), n(d["distinct_outputs"]),
                 pct(d["top1_share"]), pct(d["top5_share"])])
T.table(f"Phân bố độ dài và mức độ lặp mẫu câu của báo cáo sinh ra trên tập kiểm tra ({n(LEN['armC']['n'])} nghiên cứu)",
        ["Hệ thống", "Trung vị số từ", "Tỉ lệ từ 10 từ trở xuống", "Tỉ lệ từ 5 từ trở xuống", "Số đầu ra khác nhau",
         "Tỉ lệ đầu ra phổ biến nhất", "Tỉ lệ 5 đầu ra phổ biến nhất"],
        rows, widths=[4.0, 1.9, 2.1, 2.1, 2.1, 2.2, 2.2], font_size=9,
        note="Hai đầu ra được coi là giống nhau khi trùng khớp hoàn toàn về chuỗi ký tự. Nội dung văn bản không được trình bày vì là dữ liệu dẫn xuất từ MIMIC-CXR.")
dA, dC, ov, dV = LEN["armA"], LEN["armC"], LEN["overlap"], LEN["armC_val_conditional"]
T.para(
    f"Trung vị độ dài của hệ thống C trên tập kiểm tra chỉ là {f(dC['median_words'], 0)} từ, trong khi báo cáo tham chiếu "
    f"có trung vị {f(ov['ref_median'], 0)} từ. Phân bố độ dài có dạng hai đỉnh: {pct(dC['share_le10'])} báo cáo của hệ "
    f"thống C có từ 10 từ trở xuống, và riêng đầu ra phổ biến nhất, một chuỗi {dC['top1_words']} từ, chiếm "
    f"{pct(dC['top1_share'])} tổng số nghiên cứu; năm đầu ra phổ biến nhất chiếm {pct(dC['top5_share'])}. Hệ thống A có "
    f"cùng hiện tượng ở mức nhẹ hơn, với {pct(dA['share_le10'])} báo cáo từ 10 từ trở xuống và đầu ra phổ biến nhất chiếm "
    f"{pct(dA['top1_share'])}. MedGemma không tinh chỉnh không sinh báo cáo nào ngắn như vậy. Có ba kiểm tra cho thấy đây "
    "là hành vi của mô hình chứ không phải lỗi của quy trình đánh giá. Thứ nhất, độ ngắn không đi theo độ dài của báo cáo "
    f"tham chiếu: trung vị độ dài tham chiếu của nhóm báo cáo sinh ngắn là {f(dC['ref_median_when_short'], 0)} từ và của "
    f"nhóm còn lại là {f(dC['ref_median_when_long'], 0)} từ, trong khi chỉ {pct(ov['ref_share_le15'])} báo cáo tham chiếu "
    f"có từ 15 từ trở xuống. Thứ hai, {n(ov['short_both'])} nghiên cứu cho đầu ra ngắn ở cả hai hệ thống A và C, tức là "
    "cùng một nhóm ảnh kéo cả hai mô hình tinh chỉnh về câu mẫu. Thứ ba, tệp tóm tắt quá trình sinh xác nhận quy tắc gợi "
    "ý, danh sách mã dừng và số nghiên cứu đúng như cấu hình. Trên 100 nghiên cứu kiểm định với cùng quy tắc gợi ý, trung "
    f"vị độ dài là {f(dV['median_words'], 0)} từ và {pct(dV['share_le10'])} báo cáo có từ 10 từ trở xuống, nên mức sụp trên "
    "tập kiểm tra nặng hơn trên tập kiểm định. Hiện tượng này giải thích vì sao hệ thống C có CIDEr cao nhất nhưng BLEU, "
    "METEOR, ROUGE-L và BERTScore thấp nhất: các độ đo dựa trên độ phủ và độ dài phạt nặng báo cáo ngắn, còn CIDEr không "
    "phản ánh cùng chiều. Vì vậy CIDEr không được dùng riêng lẻ để kết luận hệ thống C tốt hơn.")

T.section_heading("Lỗi điều kiện dừng sinh")
T.para(
    "Trong quá trình phát triển, mã nguồn ghi đè danh sách mã dừng của mô hình bằng mã kết thúc của bộ tách từ, làm "
    "mất mã end_of_turn mà báo cáo đích dùng để kết thúc. Thí nghiệm có đối chứng dưới đây dùng 25 nghiên cứu kiểm định "
    "được chọn theo hạt giống trước khi xem đầu ra, chạy bốn nhánh trong cùng một phiên nạp mô hình với thứ tự luân "
    "phiên; số token là số đếm chính xác chứ không phải ước lượng.")
T.table("Số token sinh ra trước và sau khi sửa điều kiện dừng (25 nghiên cứu, số liệu ghi nhận)",
        ["Nhánh", "Tổng token", "Chạm giới hạn 160", "Có sinh end_of_turn", "Token sau end_of_turn"],
        [["Không gợi ý, mã dừng cũ", "4.000", "25 trên 25", "25 trên 25", "3.112"],
         ["Không gợi ý, mã dừng đúng", "888", "0 trên 25", "25 trên 25", "0"],
         ["Gợi ý chọn lọc, mã dừng cũ", "4.000", "25 trên 25", "25 trên 25", "3.096"],
         ["Gợi ý chọn lọc, mã dừng đúng", "904", "0 trên 25", "25 trên 25", "0"]],
        widths=[4.8, 2.4, 3.0, 3.0, 3.2], font_size=10)
so = STD["stop_probe_25"]["results"]
rows = []
for k, label in (("none_old", "Không gợi ý, mã dừng cũ"), ("none_fixed", "Không gợi ý, mã dừng đúng"),
                 ("selective_old", "Gợi ý chọn lọc, mã dừng cũ"), ("selective_fixed", "Gợi ý chọn lọc, mã dừng đúng")):
    r = so[k]
    rows.append([label, f(r["bleu_4"]), f(r["meteor"]), f(r["rouge_l"]), f(r["cider"]),
                 f(r["bertscore_f1_roberta_large_raw"]), f(r["gen_median_words"], 0), pct(r["repeated_5gram_3x"])])
T.table("Chất lượng báo cáo trước và sau khi sửa điều kiện dừng (25 nghiên cứu, giao thức chuẩn)",
        ["Nhánh", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "BERTS thô", "Trung vị từ", "Lặp 5-gram"],
        rows, widths=[4.4, 1.6, 1.7, 1.7, 1.6, 1.7, 1.7, 1.9], font_size=9)
T.para(
    "Mô hình phát tín hiệu kết thúc ở mọi lần sinh, nhưng mã nguồn buộc nó viết tiếp: 3.112 trên 4.000 token, tức "
    "78%, nằm sau mã end_of_turn. Sau khi sửa, không lần sinh nào chạm giới hạn độ dài và không còn token nào sau mã "
    "kết thúc. Lỗi này tác động đối xứng lên mọi hệ thống từng được so sánh, nên các kết luận tương đối trước đó nhiều "
    "khả năng vẫn đứng vững, nhưng mọi con số tuyệt đối của Giai đoạn 2 sinh trước khi sửa đều không dùng được. Toàn bộ "
    "kết quả kiểm tra trong chương này được sinh lại sau khi sửa.")

T.section_heading("Nghiên cứu cắt bỏ các kênh thông tin từ Giai đoạn 1")
T.sub_heading("Quy tắc tạo gợi ý văn bản")
T.para(
    "Hệ thống C nhận đầu ra Giai đoạn 1 dưới dạng danh sách bệnh lý dương tính, âm tính, không chắc chắn. Thí nghiệm "
    "này giữ nguyên checkpoint hệ thống C, cùng 100 nghiên cứu kiểm định và cùng cấu hình giải mã đã sửa điều kiện "
    "dừng, chỉ đổi quy tắc chuyển xác suất thành gợi ý, bao gồm các quy tắc có precision cao hơn quy tắc gốc ở cùng "
    "mức recall.")
cr = STD["cue_rules_100"]
rows = []
for k, label in (("conditional", "Quy tắc gốc theo xác suất có điều kiện"), ("marginal", "Ngưỡng trên điểm biên"),
                 ("selective", "Chọn lọc, precision 0,70"), ("bal50", "Cân bằng, precision 0,50"),
                 ("rec35", "Ưu tiên recall, precision 0,35")):
    p = spair("cue_rules_100", f"{k} - none")
    rows.append([label, dcell(p["meteor"]), dcell(p["rouge_l"]), dcell(p["cider"]), dcell(p["bertscore_raw"])])
T.table("Chênh lệch so với không dùng gợi ý (100 nghiên cứu kiểm định, giao thức chuẩn)",
        ["Quy tắc tạo gợi ý", "METEOR", "ROUGE-L", "CIDEr", "BERTScore thô"],
        rows, widths=[5.2, 2.8, 2.8, 2.8, 2.8], font_size=9)
T.para(
    "Không quy tắc nào cải thiện có ý nghĩa thống kê so với việc không đưa gợi ý, kể cả quy tắc ưu tiên recall vốn "
    "trội hơn quy tắc gốc đồng thời về precision và recall. Do đó việc gợi ý văn bản không giúp ích không bắt nguồn "
    "từ việc chọn sai điểm vận hành.")

T.sub_heading("Token phát hiện học được và kiểm tra việc sử dụng kênh")
T.para(
    "Để kiểm tra giả thuyết rằng vấn đề nằm ở kênh truyền dạng văn bản chứ không ở thông tin, đề tài thử một kênh "
    "khác: 13 token học được, mỗi token mang danh tính một bệnh lý cùng các giá trị liên tục của cổng nhắc tới và "
    "phân bố phân cực. Bốn nhánh được huấn luyện trên cùng 10.000 nghiên cứu, cùng hạt giống và cùng cấu hình, rồi "
    "sinh trên cùng 300 nghiên cứu kiểm định: A không gợi ý, B gợi ý văn bản, C token chỉ mang phân cực, D token mang "
    "đầy đủ cổng nhắc tới và phân cực. Ba can thiệp khi suy luận trên nhánh D kiểm tra mô hình có thực sự đọc kênh "
    "này hay không; đây là can thiệp ngoài phân phối, chỉ dùng như phép thăm dò cơ chế.")
pl3 = STD["pilot_300"]
rows = []
for comp, label in (("B - A", "B trừ A: gợi ý văn bản"), ("C - A", "C trừ A: token phân cực"),
                    ("D - A", "D trừ A: token đầy đủ"), ("D - B", "D trừ B: token so với văn bản"),
                    ("D - C", "D trừ C: đóng góp của cổng nhắc tới"),
                    ("D_zero - D", "Xóa giá trị token"), ("D_shuffle - D", "Đảo token giữa các bệnh lý"),
                    ("D_permute - D", "Tráo token giữa các nghiên cứu")):
    p = spair("pilot_300", comp)
    rows.append([label, dcell(p["meteor"]), dcell(p["rouge_l"]), dcell(p["cider"]), dcell(p["bertscore_raw"])])
T.table("Kênh gợi ý văn bản và token phát hiện (300 nghiên cứu kiểm định, giao thức chuẩn)",
        ["So sánh", "METEOR", "ROUGE-L", "CIDEr", "BERTScore thô"],
        rows, widths=[5.2, 2.8, 2.8, 2.8, 2.8], font_size=9,
        note="Ba dòng cuối là can thiệp khi suy luận trên checkpoint D, ngoài phân phối huấn luyện.")
T.para(
    "Kênh token học được, mang nhiều thông tin hơn gợi ý văn bản, không tốt hơn gợi ý văn bản cũng như không tốt hơn "
    "việc không dùng gợi ý; thêm cổng nhắc tới vào token cũng không tạo khác biệt. Ba phép can thiệp khi suy luận, "
    "gồm xóa hoàn toàn giá trị, gán giá trị cho sai bệnh lý và dùng dự đoán của bệnh nhân khác, đều không làm thay đổi "
    "đầu ra một cách có ý nghĩa. Mô hình sau huấn luyện không sử dụng kênh này.")

T.sub_heading("Thay mô hình Giai đoạn 1")
rows = []
for comp, label in (("A_prime - A", "Không gợi ý: Giai đoạn 1 mới trừ cũ"),
                    ("B_prime - B", "Gợi ý văn bản: Giai đoạn 1 mới trừ cũ"),
                    ("B_prime - A_prime", "Gợi ý văn bản trừ không gợi ý, Giai đoạn 1 mới")):
    p = spair("pilot_300", comp)
    rows.append([label, dcell(p["meteor"]), dcell(p["rouge_l"]), dcell(p["cider"]), dcell(p["bertscore_raw"])])
T.table("Giai đoạn 2 với Giai đoạn 1 dùng hàm mất mát phân cấp (300 nghiên cứu kiểm định)",
        ["So sánh", "METEOR", "ROUGE-L", "CIDEr", "BERTScore thô"],
        rows, widths=[5.2, 2.8, 2.8, 2.8, 2.8], font_size=9)
T.para(
    "Thay Giai đoạn 1 bằng mô hình huấn luyện với hàm mất mát phân cấp, vốn có điểm hiện diện được hiệu chuẩn tốt hơn "
    "giữa các nhãn, không cải thiện Giai đoạn 2 qua kênh token mềm lẫn kênh gợi ý văn bản.")

T.section_heading("Kiểm tra độ tin cậy của lời giải thích")
T.para(
    f"Bản đồ giải thích của Giai đoạn 2 được tính bằng lan truyền chú ý có trọng số gradient theo Chefer và cộng sự "
    f"{c('chefer')} trên hệ thống A, 300 nghiên cứu kiểm tra. Hai phép kiểm tra được chạy trước khi xuất bất kỳ bản đồ "
    "nào và có quyền hủy cả lần chạy. Phép kiểm tra thứ nhất thay ảnh của nghiên cứu bằng ảnh của nghiên cứu khác và đo "
    "mức tăng log-likelihood âm của báo cáo tham chiếu. Phép kiểm tra thứ hai khởi tạo lại ngẫu nhiên các lớp của mô "
    f"hình ngôn ngữ từ lớp cuối trở xuống theo Adebayo và cộng sự {c('adebayo')} và đo tương quan hạng Spearman giữa "
    "bản đồ mới và bản đồ gốc.")
T.table("Kiểm tra độ tin cậy lời giải thích trên hệ thống A (tập kiểm tra, số liệu ghi nhận)",
        ["Phép kiểm tra", "Số mẫu", "Kết quả"],
        [["Thay ảnh bằng ảnh nghiên cứu khác, tăng log-likelihood âm", "100", "+0,2160 [+0,1860; +0,2468], 94% nghiên cứu xấu đi"],
         ["Khởi tạo lại 1, 2, 4, 8, 16, 34 lớp, tương quan Spearman", "1", "0,986; 0,981; 0,893; 0,769; 0,694; 0,295"],
         ["Tỉ lệ câu gán được nhãn bằng từ điển", "1.728 câu", "0,6846"]],
        widths=[7.0, 2.0, 7.4], font_size=10)
T.para(
    "Cả hai phép kiểm tra đều đạt: mô hình dùng đúng ảnh của nghiên cứu, và bản đồ suy biến dần khi trọng số bị phá. "
    "Điểm đầu tiên của đường suy biến (0,986 sau khi khởi tạo lại một lớp) cho thấy phương pháp ít nhạy với phần trên "
    "cùng của mô hình, và cần được nêu rõ. Từ điển gán nhãn câu là danh sách từ khóa, không phải bộ gán nhãn lâm sàng.")

T.section_heading("So sánh với công trình liên quan")
rows = [
    ["R2Gen " + c("r2gen"), "0,353", "0,103", "0,142", "0,277", "không báo cáo", "không báo cáo", "0,276"],
    ["METransformer " + c("metransformer"), "0,386", "0,124", "0,152", "0,291", "0,362", "không báo cáo", "không báo cáo"],
    ["PromptMRG " + c("promptmrg"), "0,398", "0,112", "0,157", "0,268", "không báo cáo", "không báo cáo", "0,476"],
    ["META-CXR gốc " + c("metacxr"), "không báo cáo", "0,102", "0,173", "0,280", "0,291", "0,426", "không báo cáo"],
]
for k in ("zeroshot", "armA", "armC"):
    r = tm[k]
    rows.append([SYS_VI[k], f(r["bleu_1"], 3), f(r["bleu_4"], 3), f(r["meteor"], 3), f(r["rouge_l"], 3),
                 f(r["cider"], 3), f(r["bertscore_f1_roberta_large_rescaled"], 3) + " (co giãn)", "không có"])
T.table("Kết quả sinh báo cáo trên tập kiểm tra MIMIC-CXR của các công trình và của đề tài",
        ["Công trình", "BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "BERTScore", "CE F1"],
        rows, widths=[4.2, 1.5, 1.5, 1.6, 1.6, 1.8, 2.3, 1.9], font_size=9,
        note="Số của các công trình khác lấy trực tiếp từ bảng kết quả trong chính công trình đó; số của META-CXR gốc lấy từ trang mã nguồn của nhóm tác giả. CE F1: F1 lâm sàng tính bằng bộ gán nhãn trên báo cáo sinh ra, đề tài không có.")
T.table("Khác biệt giao thức cần lưu ý khi đọc bảng so sánh",
        ["Công trình", "Tập kiểm tra", "Ghi chú"],
        [["R2Gen", "3.269 báo cáo, phân chia chính thức", "coco-caption; beam 3; trung bình 5 lần chạy; CE bằng CheXpert"],
         ["METransformer", "3.269 mẫu, phân chia như R2Gen", "Không nêu tên bộ công cụ trong phần trích xuất"],
         ["PromptMRG", "3.858 mẫu theo tiền xử lý của R2Gen", "CE bằng CheXbert, 14 nhãn, trung bình theo mẫu"],
         ["META-CXR gốc", "Không nêu trong trang mã nguồn", "Chưa rõ BERTScore có co giãn hay không; cần đối chiếu bản đầy đủ"],
         ["Đề tài", f"{n(STD['test_main']['n_studies'])} nghiên cứu, tập con của phân chia chính thức", "coco-caption; tham lam; một lần chạy; chỉ phần FINDINGS"]],
        widths=[3.2, 5.6, 7.6], font_size=9.5)
T.para(
    f"Trên các độ đo được tính cùng giao thức, hệ thống của đề tài thấp hơn các công trình chuyên sinh báo cáo đã công "
    f"bố. Ba khác biệt giới hạn phép so sánh: tập kiểm tra của đề tài là tập con {n(STD['test_main']['n_studies'])} "
    "nghiên cứu thay vì toàn bộ 3.269; các công trình kia dùng beam search và trung bình nhiều lần chạy; và báo cáo "
    "sinh ra của đề tài ngắn hơn đáng kể so với tham chiếu, làm giảm trực tiếp BLEU và METEOR. Báo cáo kỹ thuật của "
    f"MedGemma {c('medgemma')} chỉ công bố RadGraph F1 29,5 trên tập 306 ca với cả hai phần FINDINGS và IMPRESSION, "
    "là độ đo và tập đánh giá khác, nên không được đặt vào bảng trên.")

# ======================================================================== CH4
T.chapter_heading("So sánh với bài báo gốc META-CXR")
T.section_heading("Khác biệt về kiến trúc và quy trình")
T.para(
    f"Đề tài kế thừa mã nguồn công bố cùng bài báo META-CXR {c('metacxr')} và thay đổi nhiều thành phần. Bảng dưới đây "
    "liệt kê các khác biệt chính. Thông tin về bài báo gốc lấy từ trang mã nguồn của nhóm tác giả và từ chính mã nguồn "
    "tham chiếu; bản toàn văn trên IEEE Xplore không truy cập được bằng công cụ tự động trong quá trình thực hiện, nên "
    "các mục không nêu trên trang mã nguồn được ghi là không nêu.")
T.table("Khác biệt giữa bài báo gốc và đề tài",
        ["Khía cạnh", "META-CXR gốc", "Đề tài"],
        [["Bộ mã hóa ảnh", "ResNet-50, ViT, Swin", "BioViL-T (ResNet-50) và PubMedCLIP (ViT-B/32); tắt Swin"],
         ["Mô hình ngôn ngữ", "Vicuna-7B với LoRA theo mã nguồn tham chiếu", "MedGemma 1.5 4B, QLoRA NF4"],
         ["Kênh thị giác vào mô hình ngôn ngữ", "Token mềm Q-Former thay cho ảnh", "Tháp ảnh MedGemma, thêm token mềm và gợi ý ở hệ thống C"],
         ["Mục tiêu căn chỉnh Q-Former", "Bị vô hiệu trong mã nguồn tham chiếu", "Tắt sau bốn lần đo ở mức ngẫu nhiên"],
         ["Đơn vị lấy mẫu", "Ảnh, trong mã nguồn kế thừa ban đầu", "Nghiên cứu, một ảnh chính và tối đa một ảnh phụ"],
         ["Nhãn để trống", "Điền âm tính, trong mã nguồn kế thừa ban đầu", "Che theo từng ô, không coi là âm tính"],
         ["Cổng nhắc tới", "Không có", "Có, học từ các ô để trống"],
         ["Hợp nhất nhiều góc nhìn", "Không có", "Có, theo từng bộ mã hóa"],
         ["Cách chấm phân loại", "Ba lớp P/N/U, 13 bệnh lý và No Finding", "Hiện diện theo nghiên cứu, điểm biên; kèm ba lớp P/N/U để đối chiếu"],
         ["Chọn ngưỡng", "Không nêu", "Hiệu chuẩn trên tập kiểm định, quy tắc bình nguyên"]],
        widths=[4.0, 5.6, 6.8], font_size=9.5)

T.section_heading("Đặt kết quả cạnh nhau")
pnu = PL["pnu_three_class_masked"]
def pnu_std(block):
    """Macro over the three classes, undefined precision counted as 0 (the usual zero_division=0)."""
    pc = block["per_class"]
    val = lambda x: 0.0 if (x is None or x != x) else x
    return (sum(val(pc[k]["precision"]) for k in pc) / 3, sum(val(pc[k]["recall"]) for k in pc) / 3,
            sum(val(pc[k]["f1"]) for k in pc) / 3, {k: val(pc[k]["f1"]) for k in pc})
mdl, apos, maj = (pnu_std(pnu[k]) for k in ("model_argmax_q", "baseline_all_positive", "baseline_majority_per_label"))
rows = [["META-CXR gốc (trang mã nguồn)", "0,87", "0,78", "0,73", "không nêu", "không nêu", "không nêu"]]
for label, v in (("Đề tài, argmax trên phân bố q", mdl), ("Baseline luôn dương tính", apos), ("Baseline lớp đa số theo từng nhãn", maj)):
    rows.append([label, f(v[0]), f(v[1]), f(v[2]), f(v[3]["negative"]), f(v[3]["positive"]), f(v[3]["uncertain"])])
T.table("Phân loại ba lớp P/N/U trên các ô được nhắc tới (tập kiểm tra)",
        ["Hệ thống", "Precision TB", "Recall TB", "F1 TB", "F1 âm tính", "F1 dương tính", "F1 không chắc"],
        rows, widths=[4.8, 1.8, 1.7, 1.6, 1.9, 2.1, 2.5], font_size=9,
        note=f"TB: trung bình theo ba lớp, precision không xác định tính bằng 0. Đề tài tính trên {n(pnu['mentioned_cells'])} ô được nhắc tới. Trang mã nguồn của bài báo gốc không nêu cách lấy trung bình.")
mq2, apq2, mjq2 = AGG("ft_masked_q"), baseline("ft_masked_q", "all_positive"), baseline("ft_masked_q", "majority_class")
T.para(
    f"Lớp không chắc chắn của đề tài có F1 {f(mdl[3]['uncertain'])} vì mô hình được huấn luyện với chính sách bỏ qua "
    "nhãn không chắc chắn, nên đầu ra của lớp này không bao giờ nhận tín hiệu học. Hệ quả là theo cách chấm ba lớp, "
    f"baseline lớp đa số đạt F1 trung bình {f(maj[2])}, {'cao hơn' if maj[2] > mdl[2] else 'thấp hơn'} mô hình "
    f"({f(mdl[2])}); con số này phản ánh chính sách huấn luyện chứ không phản ánh khả năng phân cực. Khi bỏ lớp không "
    f"chắc chắn cho khớp chính sách huấn luyện (Mục 2.3), F1 dương tính trung bình theo nhãn của mô hình là "
    f"{f(mq2['positive_macro_f1'])}, so với {f(apq2.get('positive_macro_f1'))} của baseline luôn dương tính và "
    f"{f(mjq2.get('positive_macro_f1'))} của baseline lớp đa số. Vì bài báo gốc không nêu cách xử lý lớp không chắc "
    "chắn và cách lấy trung bình, cả hai phép chấm này đều không so trực tiếp được với con số 0,73; chúng được trình "
    "bày để cho thấy các con số P/N/U cần baseline tầm thường đi kèm mới có ý nghĩa.")
rows = [["META-CXR gốc (trang mã nguồn)", "0,102", "0,173", "0,280", "0,291", "0,426 (chưa rõ co giãn)"]]
for k in ("armA", "armC"):
    r = tm[k]
    rows.append([SYS_VI[k], f(r["bleu_4"], 3), f(r["meteor"], 3), f(r["rouge_l"], 3), f(r["cider"], 3),
                 f"{f(r['bertscore_f1_roberta_large_rescaled'], 3)} co giãn; {f(r['bertscore_f1_roberta_large_raw'], 3)} thô"])
T.table("Sinh báo cáo: bài báo gốc và đề tài",
        ["Hệ thống", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "BERTScore"],
        rows, widths=[5.0, 1.8, 1.8, 1.8, 1.8, 4.2], font_size=9.5,
        note="Số của đề tài theo giao thức chuẩn trên tập kiểm tra. Bài báo gốc không nêu tập kiểm tra, bộ công cụ và mô hình BERTScore trên trang mã nguồn.")

T.section_heading("Các thay đổi và bằng chứng đo được")
T.para(
    "Không phải mọi thay đổi so với bài báo gốc đều là cải tiến có bằng chứng về chỉ số. Nhiều thay đổi là sửa lỗi tính "
    "đúng đắn, chẳng hạn cách xử lý nhãn trống hay điều kiện dừng sinh, mà giá trị của chúng nằm ở việc làm cho phép đo "
    "có nghĩa chứ không ở việc làm tăng một con số. Bảng dưới đây phân biệt rõ ba nhóm: có cải thiện đo được, không cải "
    "thiện, và sửa lỗi không có đối chứng chỉ số.")
T.table("Tổng hợp tác động của các thay đổi",
        ["Thay đổi", "Tác động đo được", "Mục"],
        [["Mở khóa phần trên bộ mã hóa", (f"AUROC macro {sf(ru['macro_auroc']['delta'])} {ci(ru['macro_auroc']['ci_lower'], ru['macro_auroc']['ci_upper'])}", True), "2.6.1"],
         ["Dùng cổng nhắc tới khi chấm", (f"AUROC macro {sf(rg['macro_auroc']['delta'])} {ci(rg['macro_auroc']['ci_lower'], rg['macro_auroc']['ci_upper'])}", True), "2.6.1"],
         ["Hạ số ca dương tối thiểu khi hiệu chuẩn", (f"F1 macro {sf(prows('minpos')['positive_macro_f1']['delta'])} {ci(prows('minpos')['positive_macro_f1']['ci_lower'], prows('minpos')['positive_macro_f1']['ci_upper'])}, đổi precision lấy recall", True), "2.6.1"],
         ["Sửa điều kiện dừng sinh", ("Token sau mã kết thúc từ 78% về 0%", True), "3.5"],
         ["Chọn ngưỡng theo bình nguyên", "Không thay đổi F1 có ý nghĩa", "2.6.1"],
         ["Mở khóa sâu hơn", "Không cải thiện khả năng phân biệt", "2.6.1"],
         ["Hàm mất mát phân cấp", "Cải thiện hiệu chuẩn giữa các nhãn, không cải thiện phân biệt, không có lợi ở Giai đoạn 2", "2.6.1, 3.6"],
         ["Hàm mất mát giải thích", "Không cải thiện, tăng 15,5% thời gian huấn luyện", "2.6.2"],
         ["Mục tiêu căn chỉnh Q-Former", "Ở mức ngẫu nhiên trong bốn lần đo", "2.6.2"],
         ["Huấn luyện thêm epoch", "Không cải thiện trên tập kiểm tra", "2.6.2"],
         ["Gợi ý văn bản và token phát hiện", "Không cải thiện sinh báo cáo", "3.6"],
         ["Che nhãn trống, cách chấm hiện diện", "Sửa lỗi tính đúng đắn; loại bỏ baseline tầm thường đạt F1 cao", "2.3"]],
        widths=[5.0, 8.6, 2.8], font_size=9.5)

# ======================================================================== CH5
T.chapter_heading("Thảo luận và hạn chế")
T.section_heading("Thảo luận")
T.para(
    "Kết quả Giai đoạn 1 cho thấy mô hình đạt khả năng phân biệt ở mức trung bình trên câu hỏi hiện diện bệnh lý và "
    "thấp hơn một đường cơ sở CNN chuyên phân loại trên cùng chính sách nhãn. Trong các lựa chọn huấn luyện đã thử, chỉ "
    "việc mở khóa phần trên của bộ mã hóa cải thiện khả năng phân biệt; các hàm mất mát bổ sung hoặc không tạo khác "
    "biệt, hoặc chỉ thay đổi hiệu chuẩn và điểm vận hành. Một bài học phương pháp quan trọng là cách chấm: cùng một "
    "tệp dự đoán cho F1 gần 0,9 hoặc khoảng 0,35 tùy cách diễn giải ô nhãn trống, và chỉ cách chấm có baseline tầm "
    "thường bằng 0 mới phản ánh kỹ năng thực.")
def _verdict(p, metric):
    r = p[metric]
    if r["at_zero_boundary"] or not r["ci_excludes_zero"]:
        return f"không khác biệt có ý nghĩa ({sf(r['delta'])})"
    return f"{'cao hơn' if r['delta'] > 0 else 'thấp hơn'} có ý nghĩa ({sf(r['delta'])})"
T.para(
    "Trên tập kiểm tra, so với MedGemma không tinh chỉnh, hệ thống A có CIDEr " + _verdict(pa, "cider")
    + ", METEOR " + _verdict(pa, "meteor") + " và BERTScore thô " + _verdict(pa, "bertscore_raw")
    + "; hệ thống C so với hệ thống A có CIDEr " + _verdict(pac, "cider") + " và BERTScore thô "
    + _verdict(pac, "bertscore_raw") + ". Kết quả này cần được đọc cùng hai điều kiện: hệ thống A chỉ hoàn thành "
    "0,86 epoch, và báo cáo sinh ra của cả ba hệ thống đều ngắn hơn đáng kể so với báo cáo tham chiếu.")
T.para(
    "Kết quả Giai đoạn 2 cho thấy mô hình ngôn ngữ thị giác không khai thác các kênh thông tin có cấu trúc từ Giai "
    "đoạn 1. Gợi ý văn bản, kể cả khi precision được nâng lên đáng kể, không cải thiện báo cáo; token học được mang "
    "giá trị liên tục cũng không, và các phép can thiệp phá hủy thông tin trong token không làm đổi đầu ra. Giải thích "
    "phù hợp nhất là tháp ảnh của MedGemma đã mang phần lớn thông tin mà các kênh này cung cấp, trong khi token mềm của "
    "Q-Former là một phép đọc cố định chưa được huấn luyện căn chỉnh. Đây là một kết quả âm có đối chứng chặt, và nó "
    "trả lời trực tiếp câu hỏi thiết kế ban đầu của đề tài.")
T.section_heading("Hạn chế")
T.para(
    "Thứ nhất, mỗi cấu hình chỉ được huấn luyện với một hạt giống, nên không có ước lượng biến thiên giữa các lần khởi "
    "tạo; khoảng tin cậy trong tài liệu phản ánh biến thiên do mẫu kiểm tra chứ không phải biến thiên do huấn luyện. "
    "Thứ hai, hệ thống A chỉ hoàn thành 0,86 epoch và không có bước chọn checkpoint. Thứ ba, đề tài không có độ đo lâm "
    "sàng đã kiểm chứng như F1 dựa trên CheXbert hay RadGraph, là các độ đo mà nhiều công trình sinh báo cáo gần đây "
    "dùng làm kết quả chính. Thứ tư, báo cáo sinh ra ngắn hơn đáng kể so với báo cáo tham chiếu, và các mô hình tinh "
    "chỉnh có xu hướng sụp về một số câu mẫu rất ngắn trên tập kiểm tra, làm giảm các độ đo dựa trên độ phủ; nguyên nhân "
    "huấn luyện của hiện tượng này chưa được xác định. Thứ năm, tập kiểm tra Giai đoạn 2 là tập con 2.772 nghiên cứu, và khoảng 6% báo cáo đích dùng khi "
    "huấn luyện là các đoạn một dòng chủ yếu mô tả kỹ thuật chụp. Thứ sáu, các thí nghiệm cắt bỏ trên kênh gợi ý dùng "
    "ngân sách huấn luyện 10.000 nghiên cứu, loại trừ được chế độ đó chứ không loại trừ một lần huấn luyện đầy đủ. Cuối "
    "cùng, số liệu của bài báo gốc được lấy từ trang mã nguồn của nhóm tác giả và cần được đối chiếu với bản toàn văn "
    "trên IEEE Access trước khi dùng trong bản nộp chính thức.")

T.references(REFS)

# ======================================================================== CLAIM CHECKS
# Every sentence above that states a conclusion is checked against the data. If the
# standard-protocol results contradict one, the build stops rather than printing it.
def sig_pos(r): return r["ci_excludes_zero"] and not r["at_zero_boundary"] and r["delta"] > 0
def sig_any(r): return r["ci_excludes_zero"] and not r["at_zero_boundary"]
M4 = ("meteor", "rouge_l", "cider", "bertscore_raw")
problems = []
if not sig_pos(ru["macro_auroc"]): problems.append("unfreeze macro AUROC is not a significant gain")
for k in ("depth", "hier_shallow", "hier_deep"):
    if sig_pos(prows(k)["macro_auroc"]): problems.append(f"{k} macro AUROC IS a significant gain")
for k in ("hier_shallow", "hier_deep"):
    if not sig_pos(prows(k)["micro_auroc"]): problems.append(f"{k} micro AUROC is not a significant gain")
if not sig_pos(rg["macro_auroc"]): problems.append("mention gate macro AUROC is not a significant gain")
if not sig_pos(prows("minpos")["positive_macro_f1"]): problems.append("min-positive F1 is not a significant gain")
if sig_any(prows("plateau")["positive_macro_f1"]): problems.append("plateau F1 IS significant, text says it is not")
for k in ("conditional", "marginal", "selective", "bal50", "rec35"):
    p = spair("cue_rules_100", f"{k} - none")
    if any(sig_pos(p[m]) for m in M4): problems.append(f"cue rule {k} shows a significant gain")
for comp in ("D - A", "D - B", "D - C", "A_prime - A", "B_prime - B"):
    p = spair("pilot_300", comp)
    if any(sig_pos(p[m]) for m in M4): problems.append(f"{comp} shows a significant gain")
for comp in ("D_zero - D", "D_shuffle - D", "D_permute - D"):
    p = spair("pilot_300", comp)
    if any(sig_any(p[m]) for m in M4): problems.append(f"{comp} changes output significantly")
for k in ("armA", "armC"):
    r = tm[k]
    if not (r["bleu_4"] < 0.102 and r["meteor"] < 0.142 and r["rouge_l"] < 0.268):
        problems.append(f"{k} is not below every published work on BLEU-4/METEOR/ROUGE-L")
if not sum(o["auroc"] for o in ours.values()) / len(ours) < 0.834:
    problems.append("Stage-1 mean AUROC is not below CheXclusion")
if not (so["none_fixed"]["repeated_5gram_3x"] <= so["none_old"]["repeated_5gram_3x"]):
    problems.append("stop fix did not reduce repetition under the standard scorer")
if not (LEN["armC"]["share_le10"] > LEN["armA"]["share_le10"] > LEN["zeroshot"]["share_le10"]):
    problems.append("short-output ordering C > A > zero-shot does not hold")
if abs(LEN["armC"]["ref_median_when_short"] - LEN["armC"]["ref_median_when_long"]) > 5:
    problems.append("short outputs DO track reference length")
if not (tm["armC"]["cider"] > tm["armA"]["cider"] and tm["armC"]["bleu_4"] < tm["armA"]["bleu_4"]
        and tm["armC"]["bertscore_f1_roberta_large_raw"] < tm["armA"]["bertscore_f1_roberta_large_raw"]):
    problems.append("arm C CIDEr-up / BLEU-BERTScore-down pattern does not hold")
if not LEN["armC_val_conditional"]["median_words"] > LEN["armC"]["median_words"]:
    problems.append("val is not longer than test for arm C")
_te, _tr = DS["splits"]["test"], DS["splits"]["train"]
if not (_te["anchor_view_counts"]["AP"] / _te["studies"] > _tr["anchor_view_counts"]["AP"] / _tr["studies"]
        and _te["findings_tokens_valid"]["median"] > _tr["findings_tokens_valid"]["median"]
        and _te["blank_fraction"] < _tr["blank_fraction"]
        and _te["studies"] / _te["subjects"] > _tr["studies"] / _tr["subjects"]):
    problems.append("split distribution-shift paragraph does not match data")
if DSX["leak_total"] != 0: problems.append("split leakage or duplicate images found")
if (DSX["val_studies"], DSX["test_studies"]) != (1808, 3269): problems.append("val/test study counts differ from text")
if DS["raw"]["metadata_images"] != 377110: problems.append("raw image count differs from text")
if sum(DS["splits"][k]["images"] for k in DS["splits"]) != DS["raw"]["metadata_images"]:
    problems.append("manifest image total differs from raw metadata")
if problems and os.environ.get("THESIS_LAYOUT_DRAFT") != "1":
    print("CLAIM CHECK FAILED -- the text contradicts the data:")
    for x in problems: print("  -", x)
    sys.exit(2)
if problems:
    print(f"LAYOUT DRAFT ONLY: {len(problems)} claim checks FAILED and were skipped -- this output must not be delivered")
else:
    print("all claim checks passed")
T.save(OUTPUT)
print("saved", OUTPUT)
