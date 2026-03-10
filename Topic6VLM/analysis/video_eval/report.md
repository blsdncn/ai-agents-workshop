# Video Surveillance Evaluation

Scoring uses 2-second sample bins.
Truth events are snapped down to the start of their 2-second interval before building the binary presence timeline.

## sample-video-close

- Video: `/home/blaise/workspace/github.com/blsdncn/AI-Agents-Workshop/Topic6VLM/sample-video-close.mp4`
- Confusion matrix: TP=8, FP=5, TN=9, FN=7
- Precision=0.615, Recall=0.533, Accuracy=0.586, F1=0.571
- Truth events (snapped to bins): enter@00:00:04, exit@00:00:16, enter@00:00:24, exit@00:00:34, enter@00:00:42, exit@00:00:50
- Predicted events: enter@00:00:12, exit@00:00:18, enter@00:00:20, exit@00:00:22, enter@00:00:24, exit@00:00:32, enter@00:00:36, exit@00:00:38, enter@00:00:46, exit@00:00:52, enter@00:00:56
- Matched events: enter@00:00:24
- Missed truth events: enter@00:00:04, exit@00:00:16, exit@00:00:34, enter@00:00:42, exit@00:00:50
- Extra predicted events: enter@00:00:12, exit@00:00:18, enter@00:00:20, exit@00:00:22, exit@00:00:32, enter@00:00:36, exit@00:00:38, enter@00:00:46, exit@00:00:52, enter@00:00:56
- Representative grid: `/home/blaise/workspace/github.com/blsdncn/AI-Agents-Workshop/Topic6VLM/analysis/video_eval/sample-video-close-confusion-grid.png`
- Representative timestamps: fn=00:00:04, fp=00:00:16, tn=00:00:00, tp=00:00:12

## sample-video-far

- Video: `/home/blaise/workspace/github.com/blsdncn/AI-Agents-Workshop/Topic6VLM/sample-video-far.mp4`
- Confusion matrix: TP=1, FP=5, TN=6, FN=13
- Precision=0.167, Recall=0.071, Accuracy=0.280, F1=0.100
- Truth events (snapped to bins): enter@00:00:04, exit@00:00:18, enter@00:00:30, exit@00:00:44
- Predicted events: exit@00:00:04, enter@00:00:24, exit@00:00:30, enter@00:00:32, exit@00:00:34
- Matched events: none
- Missed truth events: enter@00:00:04, exit@00:00:18, enter@00:00:30, exit@00:00:44
- Extra predicted events: exit@00:00:04, enter@00:00:24, exit@00:00:30, enter@00:00:32, exit@00:00:34
- Representative grid: `/home/blaise/workspace/github.com/blsdncn/AI-Agents-Workshop/Topic6VLM/analysis/video_eval/sample-video-far-confusion-grid.png`
- Representative timestamps: fn=00:00:04, fp=00:00:00, tn=00:00:18, tp=00:00:32
