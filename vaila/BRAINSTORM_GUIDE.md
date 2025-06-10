# 🎯 Brainstorm - AI Creative Assistant Guide

## Overview
The Brainstorm tool is an AI-powered creative assistant that transforms your voice recordings into music, art prompts, and creative ideas. Version 0.3.0 brings enhanced music generation, automatic MP3 conversion, and powerful batch processing capabilities.

## 🚀 Quick Start

1. **Run the tool**: `python brainstorm.py`
2. **Create a session**: Click "New Session" to organize your outputs
3. **Record audio**: Click "Record Audio" and speak your ideas
4. **Transcribe**: Convert your audio to text automatically
5. **Generate music**: Create sophisticated MIDI compositions from your words
6. **Export**: All files are saved in your session folder

## 📋 Key Features

### 🎤 Audio Recording & Processing
- **Automatic MP3 conversion**: Records in WAV and converts to MP3 if FFmpeg is installed
- **Flexible duration**: Record from 5 seconds to 10 minutes
- **Multi-format support**: Load existing WAV, MP3, M4A, FLAC files
- **Batch transcription**: Process multiple audio files at once

### 🎵 Enhanced Music Generation
- **Mood detection**: Analyzes text for emotional content (happy, sad, energetic, calm, mysterious, romantic)
- **Multi-track MIDI**: Generates melody, harmony, and bass lines
- **Dynamic composition**: Uses word length and position to create rhythm
- **Scale selection**: Chooses appropriate musical scales based on mood
- **Batch music generation**: Create music for multiple text files automatically

### 📝 Text Processing
- **Multi-language support**: Transcribes in Portuguese, English, and Spanish
- **Editable transcriptions**: Modify text before generating music
- **Text-to-speech**: Convert text back to audio with multiple TTS engines

### 🎨 Creative Outputs
- **Image prompts**: Generate detailed prompts for AI art generation
- **Creative ideas**: Get suggestions for projects based on your content
- **Session organization**: All outputs organized in timestamped folders

## 🛠️ Installation & Requirements

### Basic Requirements
```bash
pip install tkinter speech_recognition sounddevice soundfile midiutil
```

### Optional Tools (Recommended)
- **FFmpeg**: For MP3 conversion
  - Windows: Download from https://ffmpeg.org/
  - Linux: `sudo apt install ffmpeg`
  - Mac: `brew install ffmpeg`

- **FluidSynth**: For MIDI to MP3 conversion
  - Windows: Download from https://www.fluidsynth.org/
  - Linux: `sudo apt install fluidsynth`
  - Mac: `brew install fluidsynth`

## 📁 Session Structure

Each session creates an organized folder structure:

```
brainstorm_20250218_143022/
├── audio/          # Original recordings and MP3 files
│   ├── audio_143025.wav
│   └── audio_143025.mp3
├── text/           # Transcriptions and creative ideas
│   ├── transcription_143030.txt
│   └── creative_ideas_143045.txt
├── scripts/        # Python music generation scripts
│   ├── music_code_143050.py
│   └── brainstorm_music_143055.mid
└── images/         # Image prompts and descriptions
    └── image_prompts_143100.txt
```

## 🎹 Music Generation Details

### Mood Analysis
The tool analyzes your text for emotional keywords in multiple languages:
- **Happy**: feliz, alegre, joy, bright, festa
- **Sad**: triste, melancolia, saudade, dark
- **Energetic**: energia, rapido, dance, action
- **Calm**: paz, sereno, tranquil, quiet
- **Mysterious**: misterio, estranho, shadow
- **Romantic**: amor, coração, passion, together

### Musical Parameters
Based on detected mood:
- **Tempo**: 60-140 BPM
- **Scale**: Major/minor keys
- **Time signature**: 3/4, 4/4, or 5/4
- **Dynamics**: Volume variations

### Generation Process
1. Text analysis for mood and word count
2. Scale and tempo selection
3. Melody generation based on word rhythm
4. Harmony with appropriate chord progressions
5. Bass line following the harmonic structure

## 💡 Usage Tips

### For Best Results
1. **Clear speech**: Speak clearly for accurate transcription
2. **Descriptive text**: Use emotional and descriptive words
3. **Edit before generating**: Review and enhance transcriptions
4. **Batch processing**: Use batch features for multiple files

### Workflow Examples

#### Single Recording Workflow
1. Record your thoughts (30 seconds)
2. Transcribe automatically
3. Edit if needed
4. Generate music
5. Execute code to create MIDI
6. Convert to MP3

#### Batch Processing Workflow
1. Select folder with audio files
2. Choose "Batch Transcribe"
3. Optionally generate music for each
4. Review all outputs in session folder

## 🔧 Troubleshooting

### Common Issues

**No MP3 created after recording**
- Install FFmpeg and ensure it's in PATH

**MIDI to MP3 conversion fails**
- Install FluidSynth or TiMidity
- Use online converters as alternative

**Transcription errors**
- Check internet connection
- Try different language settings
- Speak more clearly

**Music generation fails**
- Install midiutil: `pip install midiutil`
- Check Python script for errors

## 📊 Advanced Features

### Batch Music Generation
Process existing transcriptions:
1. Click "Batch Music Generation"
2. Select folder with .txt files
3. Music generated for each file
4. MIDI files created automatically

### Custom Mood Keywords
Edit the `_generate_enhanced_music_code` method to add your own mood keywords and musical preferences.

### API Integration
Set API mode to "openai" for GPT-powered generation (requires API key).

## 🎯 Creative Use Cases

1. **Songwriting**: Record lyrics, generate backing tracks
2. **Podcasts**: Create intro/outro music from show descriptions
3. **Meditation**: Generate calming music from peaceful thoughts
4. **Education**: Turn lessons into memorable musical pieces
5. **Therapy**: Express emotions through music generation
6. **Content Creation**: Generate unique background music

## 📈 Future Enhancements

Planned features:
- Real-time music generation during recording
- More sophisticated harmony generation
- Integration with music production software
- Direct upload to music platforms
- Collaborative sessions

## 🤝 Contributing

To contribute to the Brainstorm tool:
1. Fork the repository
2. Create your feature branch
3. Test thoroughly
4. Submit a pull request

## 📄 License

GNU General Public License v3.0

---

Created by Paulo Roberto Pereira Santiago
Part of the vailá multimodal toolbox project 