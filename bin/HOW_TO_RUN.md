# How to Run (Step by Step)

1) Set up a virtual environment (recommended).
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`

2) Install dependencies.
   - `python -m pip install --upgrade pip`
   - `pip install -r requirements.txt`

3) Verify required files are present.
   - `models/yolo11m.pt` and `models/yolo11n-pose.pt` (model weights)
   - `test_video.mp4` (sample input video)

4) Run the main script (press `q` to quit).
   - `python final2.2.py`

Optional runs
5) Circular touch demo.
   - `python step_touch_circular.py`

6) Streamlit UI (open the URL it prints, upload a video, click Run Detection).
   - `streamlit run streamlit_app.py`

Notes
- To use a different video, replace `test_video.mp4` or update the path in `final2.2.py`.
- If Ultralytics asks for PyTorch separately, follow its prompt or install it from https://pytorch.org.
