# AI People Reader - Motion Detection & Analysis

A sophisticated web application that analyzes human motion patterns and interpersonal skills through video analysis using AI and computer vision techniques.

## 🎯 Features

### Motion Detection & Analysis
- **Real-time skeleton tracking** using MediaPipe
- **14 motion types detection**: Effort Actions (Gliding, Floating, Punching, Dabbing, Flicking, Slashing, Wringing, Pressing) and Directional Motions (Advancing, Retreating, Enclosing, Spreading, Directing, Indirecting)
- **Video processing with skeleton overlay**
- **Detailed motion timeline analysis**

### Interpersonal Skills Assessment
- **4 key skill categories**:
  1. **Engaging & Connecting** - Can the speaker engage the audience?
  2. **Adaptability** - Can the speaker handle different types of audiences?
  3. **Confidence** - Can the speaker argue/insist on beliefs?
  4. **Authority** - Can the speaker get desired results from the audience?

### User Management System
- **Admin Access** (username: `admin`, password: `0108`)
  - Full video analysis capabilities
  - Access to complete motion detection and skills reports
  - Video management system
  - Download and remove user-uploaded videos

- **Regular User Access**
  - Upload videos for admin review
  - Videos remain on system until admin processes them

### Report Generation
- **Comprehensive Word documents** (.docx format)
- **Detailed motion analysis charts**
- **Interpersonal skills assessment with scores**
- **Movement combination analysis**
- **Timeline-based detection logs**

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ai-people-reader.git
   cd ai-people-reader
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   - Open your browser and go to `http://localhost:8501`

## 📋 Usage Instructions

### For Regular Users
1. **Login** with any username/password (except admin credentials)
2. **Upload your video** using the file uploader
3. **Wait for admin review** - your video will remain on the system
4. **Admin will analyze** and provide detailed reports

### For Admins
1. **Login** with username: `admin`, password: `0108`
2. **Upload videos** or **analyze user-submitted videos**
3. **Click "Complete Analysis"** to generate comprehensive reports
4. **Download Word documents** with detailed analysis
5. **Manage uploaded videos** - download and remove as needed

### Video Recording Guidelines
- **Orientation**: Record vertically (portrait mode)
- **Duration**: Answer questions in detail
- **Movement**: Move naturally while speaking
- **Quality**: Ensure good lighting and clear audio

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **OpenCV**: Video processing and computer vision
- **MediaPipe**: Pose detection and skeleton tracking
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive charts and visualizations
- **python-docx**: Word document generation
- **NumPy**: Numerical computations

### Architecture
- **Frontend**: Streamlit web interface with custom CSS styling
- **Backend**: Python-based motion detection and analysis engine
- **Data Processing**: Real-time video analysis with pose landmark detection
- **Report Generation**: Automated Word document creation with charts

### File Structure
```
ai-people-reader/
├── app.py                          # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
├── Complete_Effort_Action_Motion_Descriptions__All_8_.csv
├── Motion_Contrast_Chart_for_AI_Training.csv
├── Movement Combination Summary_Revised 151025.xlsx
└── Video Interview Simulation.mp4  # Demo video
```

## 🎨 Customization

### Styling
The app uses custom CSS for a dark theme with grey background and white text. You can modify the styling in the `main()` function of `app.py`.

### Motion Detection
Adjust motion detection sensitivity by modifying thresholds in the `MotionDetector` class:
- `FAST_THRESHOLD`: For quick movements
- `MEDIUM_THRESHOLD`: For moderate movements  
- `SLOW_THRESHOLD`: For subtle movements

### Skills Assessment
Customize interpersonal skills evaluation by modifying the `analyze_interpersonal_skills()` function parameters.

## 📊 Sample Analysis

The application includes a demo video ("Video Interview Simulation.mp4") that showcases:
- Motion detection with skeleton overlay
- Interpersonal skills assessment
- Detailed report generation
- Movement combination analysis

## 🔒 Security

- **Admin credentials**: Hardcoded for demo purposes (change in production)
- **Session management**: User roles and uploaded videos stored in session state
- **File handling**: Temporary files cleaned up after processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** for pose detection capabilities
- **Streamlit** for the web application framework
- **OpenCV** for computer vision processing
- **Plotly** for interactive visualizations

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation in the README
- Review the code comments for technical details

---

**Note**: This application is designed for educational and research purposes. Ensure proper consent and privacy considerations when analyzing video content.