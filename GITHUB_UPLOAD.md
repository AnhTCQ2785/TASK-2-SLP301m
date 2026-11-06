# Hướng dẫn Upload lên GitHub

## 📁 Cấu trúc Repository

```
Truong/ (repository root)
├── README.md                          # Hướng dẫn sử dụng repository
├── DATA_CARD.md                       # Data Card + Manifest
├── TASK2_REPORT.md                    # Báo cáo chi tiết Task 2
├── TASK2_CONCLUSION.md                # Kết luận Task 2
├── EXPERIMENT_LOG.md                  # Nhật ký thực nghiệm
├── SLIDE_CONTENT.md                   # Nội dung slide
├── CONTRIBUTION.md                    # Bảng phân công
├── RESULTS_TABLE.csv                  # Bảng kết quả
├── GITHUB_UPLOAD.md                   # File này
├── COLAB_CELLS.md                     # Colab notebook cells
├── data/
│   ├── manifests/
│   │   ├── vivos_train.json          # Training manifest
│   │   └── vivos_test.json           # Test manifest
│   └── waves/                        # Audio files (có thể gitignore)
├── scripts/
│   ├── train_baseline.py             # Baseline training script
│   ├── train_v1.py                   # V1 training script
│   ├── train_v2.py                   # V2 training script
│   ├── train_v3.py                   # V3 training script
│   └── compare_versions.py          # Comparison script
├── results/
│   ├── training_histories.json
│   ├── v1_history.json
│   ├── v2_history.json
│   ├── v3_history.json
│   └── baseline_results.json
├── models/                           # (gitignore - quá lớn)
│   ├── whisper_vivos_seed42/
│   ├── whisper_vivos_v1_specaugment/
│   ├── whisper_vivos_v2_cosine/
│   └── whisper_vivos_v3_base/
└── checkpoints/                       # (gitignore - quá lớn)
```

## ✅ Files Cần Upload (Bắt buộc)

### 1. Documentation
- ✅ `README.md` - Hướng dẫn sử dụng repository
- ✅ `DATA_CARD.md` - Data Card + Manifest
- ✅ `TASK2_REPORT.md` - Báo cáo chi tiết
- ✅ `TASK2_CONCLUSION.md` - Kết luận
- ✅ `EXPERIMENT_LOG.md` - Nhật ký thực nghiệm
- ✅ `SLIDE_CONTENT.md` - Nội dung slide
- ✅ `CONTRIBUTION.md` - Bảng phân công
- ✅ `GITHUB_UPLOAD.md` - File này

### 2. Code
- ✅ `COLAB_CELLS.md` - Colab notebook cells
- ✅ `scripts/train_baseline.py`
- ✅ `scripts/train_v1.py`
- ✅ `scripts/train_v2.py`
- ✅ `scripts/train_v3.py`
- ✅ `scripts/compare_versions.py`

### 3. Data (Manifests)
- ✅ `data/manifests/vivos_train.json`
- ✅ `data/manifests/vivos_test.json`

### 4. Results
- ✅ `results/training_histories.json`
- ✅ `results/v1_history.json`
- ✅ `results/v2_history.json`
- ✅ `results/v3_history.json`
- ✅ `results/baseline_results.json`

### 5. Tables
- ✅ `RESULTS_TABLE.csv`

## ❌ Files KHÔNG Upload (Gitignore)

### 1. Audio Files (quá lớn)
- ❌ `data/waves/` - Toàn bộ audio files (nhiều GB)
- ❌ `data/raw/` - Dữ liệu thô VIVOS

### 2. Model Checkpoints (quá lớn)
- ❌ `models/` - All fine-tuned models (~1-2GB each)
- ❌ `checkpoints/` - Training checkpoints

### 3. Temporary Files
- ❌ `*.pyc`
- ❌ `__pycache__/`
- ❌ `.ipynb_checkpoints/`
- ❌ `*.ckpt`
- ❌ `*.pth` (trừ trong scripts)

## 📝 File .gitignore

Tạo file `.gitignore` với nội dung sau:

```gitignore
# Audio files (quá lớn)
data/waves/
data/raw/
*.wav
*.mp3
*.flac

# Model checkpoints (quá lớn)
models/
checkpoints/
*.pt
*.pth
*.ckpt
*.safetensors

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# Colab
.colab/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment
.env
venv/
env/
```

## 🚀 Lệnh Upload

### Bước 1: Tạo GitHub Repository

1. Đi đến https://github.com/new
2. Repository name: `vietnamese-asr-whisper` (hoặc tên bạn muốn)
3. Description: "Vietnamese ASR with Whisper Fine-tuning - Task 2"
4. Public/Private: Public (hoặc Private tùy yêu cầu)
5. Initialize: **KHÔNG** chọn "Add a README file" (đã có sẵn)
6. Click "Create repository"

### Bước 2: Setup Local Git

```bash
cd D:\Truong
git init
git add .
git commit -m "Initial commit: Task 2 Vietnamese ASR with Whisper"
```

### Bước 3: Kết nối với GitHub

```bash
git remote add origin https://github.com/[username]/vietnamese-asr-whisper.git
git branch -M main
git push -u origin main
```

Thay `[username]` bằng GitHub username của bạn.

### Bước 4: Thay thế remote URL (nếu đã có)

```bash
git remote set-url origin https://github.com/[username]/vietnamese-asr-whisper.git
```

## 🏷️ Tạo Tags cho Phiên Bản

```bash
# Tag các phiên bản
git tag v1.0 -m "Baseline implementation"
git tag v1.1 -m "Added V1 (SpecAugment)"
git tag v1.2 -m "Added V2 (Cosine LR)"
git tag v1.3 -m "Added V3 (Whisper Base)"
git tag v1.4 -m "Final comparison and documentation"

# Push tags lên GitHub
git push origin v1.0
git push origin v1.1
git push origin v1.2
git push origin v1.3
git push origin v1.4

# Hoặc push tất cả tags cùng lúc
git push origin --tags
```

## 📋 Checklist Upload

Trước khi upload, kiểm tra:

- [ ] File `.gitignore` đã tạo
- [ ] Không có audio files trong staging area
- [ ] Không có model checkpoints trong staging area
- [ ] Tất cả documentation files đã có
- [ ] Tất cả scripts đã có
- [ ] Manifest files đã có
- [ ] Results files đã có
- [ ] README.md đã cập nhật
- [ ] DATA_CARD.md đã cập nhật
- [ ] CONTRIBUTION.md đã điền tên thành viên

## 🔍 Kiểm tra trước khi Push

```bash
# Kiểm tra file sẽ được push
git status

# Kiểm tra file lớn
git ls-files | xargs ls -lh | grep -E "[0-9]+G"

# Kiểm tra số lượng file
git ls-files | wc -l
```

## 📤 Upload Audio Files (Optional)

Nếu cần upload audio files (không khuyến khích), dùng:

**Option 1: GitHub LFS (Large File Storage)**
```bash
# Cài đặt Git LFS
git lfs install

# Track audio files
git lfs track "*.wav"
git lfs track "*.mp3"

# Commit & push
git add .gitattributes
git commit -m "Track audio files with LFS"
git push
```

**Option 2: Upload lên Google Drive**
- Upload audio files lên Google Drive
- Đường dẫn trong README.md hoặc DATA_CARD.md

**Option 3: Upload lên Cloud Storage khác**
- AWS S3
- Google Cloud Storage
- Dropbox

## 📝 Sau khi Upload

1. **Kiểm tra trên GitHub**:
   - Truy cập repository URL
   - Kiểm tra tất cả files đã upload
   - Kiểm tra tags

2. **Cập nhật README.md**:
   - Thêm link đến GitHub repository
   - Thêm link đến release/tags

3. **Share link**:
   - Copy repository URL
   - Gửi cho giáo viên/nhóm

## 🔗 Links Cần Cập Nhật trong README.md

Sau khi upload, cập nhật các link trong `README.md`:

```markdown
## Download

### GitHub Repository
https://github.com/[username]/vietnamese-asr-whisper

### Releases/Tags
https://github.com/[username]/vietnamese-asr-whisper/releases
```

## ⚠️ Lưu ý Quan Trọng

1. **KHÔNG upload** model checkpoints (quá lớn, sẽ bị lỗi GitHub)
2. **KHÔNG upload** audio files trừ khi dùng LFS hoặc cloud storage
3. **Đảm bảo** file `.gitignore` đã đúng
4. **Commit rõ ràng** với message mô tả
5. **Đặt tags** cho từng phiên bản để dễ tracking
6. **Diền tên thật** vào CONTRIBUTION.md trước khi upload
