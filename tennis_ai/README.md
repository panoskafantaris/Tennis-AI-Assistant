conda create --name tennis-ai
conda activate tennis-ai
cd tennis_ai

# 1. Install PyTorch with CUDA 12.1 (for RTX 4050)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# 2. Install the rest
pip install -r requirements.txt

# 3. Download TrackNet weights (~50MB, MIT License)
python download_weights.py

# 4. Test on a YouTube tennis video
python main.py --source "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# 5. Test on just 200 frames first
python main.py --source tennis_match_2.mp4 --max-frames 200
python main.py --source "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --max-frames 200


# Two-pass with interpolation (~100% coverage, not real-time)
python main.py --source tennis_match_2.mp4 --two-pass --max-frames 200

# Old hybrid-only mode (now improved but still weaker)
python main.py --source tennis_match_2.mp4 --detector hybrid