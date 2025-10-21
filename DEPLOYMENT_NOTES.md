# Deployment Notes - Memory-Safe Version

## 🎯 Purpose
This version is specifically optimized for Render's free tier (512MB RAM limit) to prevent 502 errors.

## ✅ Key Optimizations Applied

### 1. **Memory-Safe Video Processing**
- **Streaming file upload**: Reads uploads in 1MB chunks instead of loading entire file
- **Frame-by-frame processing**: No video buffering in memory
- **Periodic garbage collection**: Cleans up memory every 60 frames
- **Single-threaded processing**: Prevents thread explosion (`OMP_NUM_THREADS=1`)

### 2. **FFmpeg Transcoding**
- Automatically transcodes uploaded videos to 720p @ 24fps
- Uses H.264 with `veryfast` preset and `crf=26` for speed
- Removes audio to save memory and CPU
- Uses `faststart` for web streaming

### 3. **MediaPipe Lazy Loading**
- MediaPipe only imported when needed (not at startup)
- Graceful fallback if MediaPipe unavailable
- Lightweight landmark drawing (single-pixel dots instead of heavy overlays)
- Model complexity set to 1 (balanced mode)

### 4. **Dependencies Minimized**
- Removed unused packages (pandas, plotly, matplotlib, python-docx, openpyxl)
- Only essential packages: streamlit, opencv-python-headless, mediapipe, numpy
- Uses `opencv-python-headless` (no GUI dependencies that cause 502 errors)

### 5. **System Configuration**
- Python 3.9.18 (better stability than 3.8.1)
- FFmpeg installed via apt-get in build command
- All threading libraries limited to 1 thread
- Upload limit increased to 1000MB (file size, not RAM)

## 🔧 Configuration Files

### `requirements.txt`
```
streamlit>=1.28.0
opencv-python-headless>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
```

### `render.yaml`
- Plan: `free` (512MB RAM)
- Build: Installs ffmpeg + system libraries
- Environment variables for thread limiting
- Headless Streamlit mode

### `packages.txt`
- libgl1-mesa-glx (OpenCV dependency)
- libglib2.0-0 (GLib dependency)
- ffmpeg (video transcoding)

## 🎨 Features Preserved

### User Features
✅ Login system (admin/user roles)
✅ Demo video display
✅ Thai instructions
✅ Grey background with white text
✅ Candidate name and date inputs
✅ Video upload functionality

### Admin Features
✅ Full video analysis with MediaPipe
✅ Skeleton overlay rendering
✅ CSV timestamp export
✅ Download processed video

### User Features
✅ Video upload
✅ Video preview
✅ Queued for admin review

## 🚀 Deployment Steps

1. **Push to GitHub** (✅ Done)
2. **Go to Render Dashboard**
3. **Manual Deploy** → Deploy latest commit
4. **Wait 5-7 minutes** (FFmpeg installation takes time)
5. **Check build logs** for any errors
6. **Access app URL** - should work without 502!

## 📊 Expected Performance

- **Build time**: ~5-7 minutes (FFmpeg + MediaPipe installation)
- **Startup time**: ~10-15 seconds
- **Memory usage**: ~200-400MB during processing
- **Upload limit**: 1000MB file size
- **Processing speed**: ~24fps on free tier

## ⚠️ Known Limitations (Free Tier)

- Processing large videos (>5 minutes) may timeout
- Concurrent users limited (free tier has request limits)
- No persistent storage (uploaded videos lost on redeploy)
- Processing is slower than local machine

## 🔍 Troubleshooting

### If 502 persists:
1. Check Render logs for actual error
2. Verify FFmpeg installed correctly
3. Confirm mediapipe installation succeeded
4. Check if video file is corrupted

### If memory errors:
1. Reduce video resolution further (480p instead of 720p)
2. Increase frame skip (process every 2nd or 3rd frame)
3. Consider upgrading to Starter plan (512MB → 2GB)

## 📝 Next Steps (Optional Improvements)

1. **Add progress indicators** for upload/transcode stages
2. **Store videos** in cloud storage (S3, Google Cloud)
3. **Queue system** for batch processing
4. **Email notifications** when processing complete
5. **More advanced movement analysis** algorithms

---

**Last Updated**: October 16, 2025
**Status**: Deployed and optimized for Render free tier

