"""
Project: vailá
Script: brainstorm.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 18 February 2025
Update Date: 18 February 2025
Version: 0.3.1

Description:
    Record voice audio, transcribe it to text, and use LLM to generate:
    - Python musical code (using MIDIUtil or Music21)
    - Artistic images via DALL-E/Stable Diffusion
    
    Enhanced workflow:
    1. Record and transcribe audio
    2. Edit transcribed text manually
    3. Generate music code or images via LLM
    4. Display and save results

    Usage:
        python brainstorm.py

Requirements:
    - Python 3.x
    - tkinter, speech_recognition, sounddevice, soundfile
    - openai (optional, for GPT integration)
    - midiutil, music21 (optional, for music generation)

License:
    GNU General Public License v3.0
"""

import os
import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog, ttk
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import datetime
import webbrowser
from pathlib import Path

# Optional imports with fallbacks
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from midiutil import MIDIFile
    MIDIUTIL_AVAILABLE = True
except ImportError:
    MIDIUTIL_AVAILABLE = False

# Optional imports for MP3 generation
try:
    import pygame.mixer
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import subprocess
    import tempfile
    SUBPROCESS_AVAILABLE = True
except ImportError:
    SUBPROCESS_AVAILABLE = False

# Optional imports for Text-to-Speech
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# Optional imports for audio conversion (fallbacks)
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Global variables
AUDIO_FILE = "audio.wav"
TRANSCRIBED_TEXT = ""
OUTPUT_DIR = "brainstorm_outputs"

class BrainstormApp:
    def __init__(self):
        self.root = tk.Tk()
        self.output_dir = OUTPUT_DIR
        self.transcription_file = None
        self.session_dir = None
        self.setup_ui()
        
    def show_message(self, msg_type, title, message):
        """Show message box always on top of the main window."""
        self.root.lift()
        self.root.attributes('-topmost', True)
        
        if msg_type == "info":
            result = messagebox.showinfo(title, message, parent=self.root)
        elif msg_type == "warning":
            result = messagebox.showwarning(title, message, parent=self.root)
        elif msg_type == "error":
            result = messagebox.showerror(title, message, parent=self.root)
        elif msg_type == "yesno":
            result = messagebox.askyesno(title, message, parent=self.root)
        
        self.root.focus_force()
        return result
        
    def show_workflow_guide(self):
        """Show a comprehensive workflow guide for users."""
        guide_text = """BRAINSTORM WORKFLOW GUIDE - v0.3.1

COMPLETE CREATIVE WORKFLOW:

1. SETUP SESSION
   • Click "Browse" to select base directory
   • Click "New Session" to create organized folders
   • All outputs will be saved in timestamped session

2. AUDIO CAPTURE
   • Record Audio: Captures voice (auto-converts to MP3)
   • Load Audio: Import existing audio files
   • Duration: Select 5-600 seconds recording time

3. TRANSCRIPTION
   • Transcribe: Convert single audio to text
   • Batch Transcribe: Process multiple audio files
   • Load Transcription: Import existing text files

4. MUSIC GENERATION
   • Generate Music Code: Creates Python/MIDI from text
   • Batch Music Generation: Process multiple texts
   • Execute Code: Run scripts to create MIDI files
   • Generate MP3: Convert MIDI to MP3 (needs tools)

5. CREATIVE OUTPUTS
   • Generate Image: Create AI art prompts
   • Creative Ideas: Get project suggestions
   • Text to Audio: Convert text back to speech

SESSION STRUCTURE:
   brainstorm_YYYYMMDD_HHMMSS/
   ├── audio/      (WAV, MP3 recordings)
   ├── text/       (transcriptions, ideas)
   ├── scripts/    (Python music code)
   └── images/     (prompts, descriptions)

MUSIC FEATURES:
   • Mood detection (happy, sad, energetic, calm, etc.)
   • Multi-track MIDI (melody, harmony, bass)
   • Text-based rhythm and dynamics
   • Automatic chord progressions
   • Scale selection based on mood

TIPS:
   • Install FFmpeg for MP3 conversion
   • Install MIDIUtil for music generation
   • Use clear speech for better transcription
   • Edit text before generating music
   • Batch process for efficiency

REQUIREMENTS:
   • FFmpeg (MP3 conversion)
   • FluidSynth/TiMidity (MIDI→MP3)
   • MIDIUtil (music generation)
   • Internet (transcription)

SHORTCUTS:
   • Record → Transcribe → Music → Export
   • Batch: Select folder → Auto-process all
   • Sessions keep everything organized!
"""
        
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Brainstorm Workflow Guide")
        guide_window.geometry("700x600")
        guide_window.attributes('-topmost', True)
        
        text_widget = scrolledtext.ScrolledText(guide_window, wrap=tk.WORD, width=80, height=35)
        text_widget.pack(padx=10, pady=10, fill="both", expand=True)
        text_widget.insert("1.0", guide_text)
        text_widget.config(state="disabled")
        
        tk.Button(guide_window, text="Close", command=guide_window.destroy,
                 bg="#f44336", fg="white", font=("Arial", 10)).pack(pady=10)
        
    def create_session_directory(self):
        """Create a new session directory with timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = f"{self.output_dir}/brainstorm_{timestamp}"
        
        Path(self.session_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.session_dir}/audio").mkdir(exist_ok=True)
        Path(f"{self.session_dir}/scripts").mkdir(exist_ok=True)
        Path(f"{self.session_dir}/images").mkdir(exist_ok=True)
        Path(f"{self.session_dir}/text").mkdir(exist_ok=True)
        
        return self.session_dir
        
    def setup_directories(self):
        """Legacy method - now creates session directory if needed."""
        if not self.session_dir:
            self.create_session_directory()
        
    def setup_ui(self):
        """Setup the user interface."""
        self.root.title("vailá Brainstorm - Voice to AI Creative Assistant")
        self.root.geometry("1000x800")
        self.root.attributes('-topmost', True)
        self.root.focus_force()
        
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        header = tk.Label(main_frame, text="Brainstorm: Voice to AI Creative Assistant", 
                         font=("Arial", 16, "bold"))
        header.pack(pady=10)
        
        # Help button
        help_frame = tk.Frame(main_frame)
        help_frame.pack()
        tk.Button(help_frame, text="Workflow Guide", command=self.show_workflow_guide,
                 bg="#607D8B", fg="white", font=("Arial", 10)).pack()
        
        # Output directory selection section
        output_frame = tk.LabelFrame(main_frame, text="Output Directory", font=("Arial", 12, "bold"))
        output_frame.pack(fill="x", pady=5)
        
        dir_selection_frame = tk.Frame(output_frame)
        dir_selection_frame.pack(fill="x", pady=5)
        
        tk.Label(dir_selection_frame, text="Base Directory:").pack(side="left")
        self.dir_var = tk.StringVar(value=self.output_dir)
        self.dir_label = tk.Label(dir_selection_frame, textvariable=self.dir_var, 
                                 relief="sunken", anchor="w", width=50)
        self.dir_label.pack(side="left", padx=5, fill="x", expand=True)
        
        tk.Button(dir_selection_frame, text="Browse", command=self.choose_output_directory,
                 bg="#607D8B", fg="white", font=("Arial", 9)).pack(side="right", padx=5)
        tk.Button(dir_selection_frame, text="New Session", command=self.create_new_session,
                 bg="#4CAF50", fg="white", font=("Arial", 9)).pack(side="right", padx=5)
        tk.Button(dir_selection_frame, text="Reset", command=self.reset_output_directory,
                 bg="#795548", fg="white", font=("Arial", 9)).pack(side="right")
        
        # Session directory display
        session_frame = tk.Frame(output_frame)
        session_frame.pack(fill="x", pady=2)
        
        tk.Label(session_frame, text="Current Session:").pack(side="left")
        self.session_var = tk.StringVar(value="No session created")
        self.session_label = tk.Label(session_frame, textvariable=self.session_var, 
                                    relief="sunken", anchor="w", width=50, fg="blue")
        self.session_label.pack(side="left", padx=5, fill="x", expand=True)
        
        # Audio section
        audio_frame = tk.LabelFrame(main_frame, text="1. Audio Recording", font=("Arial", 12, "bold"))
        audio_frame.pack(fill="x", pady=5)
        
        # Duration selection
        duration_frame = tk.Frame(audio_frame)
        duration_frame.pack(pady=2)
        
        tk.Label(duration_frame, text="Recording Duration (seconds):").pack(side="left")
        # Simplificar para usar apenas Entry, sem StringVar
        self.duration_entry = tk.Entry(duration_frame, width=8)
        self.duration_entry.insert(0, "10")  # Valor padrão inicial
        self.duration_entry.pack(side="left", padx=5)
        
        # Audio buttons (remover o botão Set Duration)
        audio_buttons = tk.Frame(audio_frame)
        audio_buttons.pack(pady=5)
        
        tk.Button(audio_buttons, text="Record Audio", command=self.record_audio,
                 bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(audio_buttons, text="Transcribe", command=self.transcribe_audio,
                 bg="#2196F3", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(audio_buttons, text="Load Audio", command=self.load_audio,
                 bg="#FF9800", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(audio_buttons, text="Load Transcription", command=self.load_transcription,
                 bg="#795548", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(audio_buttons, text="Batch Transcribe", command=self.batch_transcribe,
                 bg="#9C27B0", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        
        # Status label
        self.status_label = tk.Label(audio_frame, text="Ready to record...", fg="green")
        self.status_label.pack(pady=5)
        
        # Text editing section
        text_frame = tk.LabelFrame(main_frame, text="2. Text Editing & Prompt", font=("Arial", 12, "bold"))
        text_frame.pack(fill="both", expand=True, pady=5)
        
        text_header = tk.Frame(text_frame)
        text_header.pack(fill="x", pady=5)
        
        tk.Label(text_header, text="Edit your transcribed text or write a custom prompt:").pack(side="left", anchor="w")
        tk.Button(text_header, text="Text to Audio", command=self.text_to_audio,
                 bg="#FF9800", fg="white", font=("Arial", 9)).pack(side="right", padx=5)
        tk.Button(text_header, text="Save Text", command=self.save_transcription_edits,
                 bg="#4CAF50", fg="white", font=("Arial", 9)).pack(side="right", padx=5)
        
        self.text_display = scrolledtext.ScrolledText(text_frame, width=100, height=8, font=("Arial", 10))
        self.text_display.pack(fill="both", expand=True, pady=5)
        
        # Generation section
        gen_frame = tk.LabelFrame(main_frame, text="3. AI Generation", font=("Arial", 12, "bold"))
        gen_frame.pack(fill="x", pady=5)
        
        gen_buttons = tk.Frame(gen_frame)
        gen_buttons.pack(pady=5)
        
        tk.Button(gen_buttons, text="Generate Music Code", command=self.generate_music,
                 bg="#9C27B0", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(gen_buttons, text="Generate Image", command=self.generate_image,
                 bg="#E91E63", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(gen_buttons, text="Creative Ideas", command=self.generate_ideas,
                 bg="#FF5722", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(gen_buttons, text="Batch Music Generation", command=self.batch_generate_music,
                 bg="#00BCD4", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        
        # API Configuration
        api_frame = tk.Frame(gen_frame)
        api_frame.pack(pady=5)
        
        tk.Label(api_frame, text="AI Mode:").pack(side="left", padx=5)
        self.api_var = tk.StringVar(value="local")
        api_combo = ttk.Combobox(api_frame, textvariable=self.api_var, width=15, state="readonly")
        api_combo['values'] = ("local", "openai", "custom")
        api_combo.pack(side="left", padx=5)
        
        # Results section
        results_frame = tk.LabelFrame(main_frame, text="4. Generated Results", font=("Arial", 12, "bold"))
        results_frame.pack(fill="both", expand=True, pady=5)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill="both", expand=True, pady=5)
        
        # Music tab
        music_frame = tk.Frame(self.notebook)
        self.notebook.add(music_frame, text="Music Code")
        
        music_buttons = tk.Frame(music_frame)
        music_buttons.pack(fill="x", pady=5)
        tk.Button(music_buttons, text="Save Code", command=self.save_music_code,
                 bg="#2196F3", fg="white", font=("Arial", 9)).pack(side="left", padx=5)
        tk.Button(music_buttons, text="Execute Code", command=self.execute_music_code,
                 bg="#4CAF50", fg="white", font=("Arial", 9)).pack(side="left", padx=5)
        tk.Button(music_buttons, text="Generate MP3", command=self.generate_mp3,
                 bg="#9C27B0", fg="white", font=("Arial", 9)).pack(side="left", padx=5)
        
        self.music_display = scrolledtext.ScrolledText(music_frame, width=100, height=15, font=("Courier", 9))
        self.music_display.pack(fill="both", expand=True)
        
        # Image tab
        image_frame = tk.Frame(self.notebook)
        self.notebook.add(image_frame, text="Images")
        
        image_buttons = tk.Frame(image_frame)
        image_buttons.pack(fill="x", pady=5)
        tk.Button(image_buttons, text="Open Image URL", command=self.open_image_url).pack(side="left", padx=5)
        tk.Button(image_buttons, text="Save Image Info", command=self.save_image_info).pack(side="left", padx=5)
        
        self.image_display = scrolledtext.ScrolledText(image_frame, width=100, height=15, font=("Arial", 10))
        self.image_display.pack(fill="both", expand=True)
        
        # Ideas tab
        ideas_frame = tk.Frame(self.notebook)
        self.notebook.add(ideas_frame, text="Creative Ideas")
        
        ideas_buttons = tk.Frame(ideas_frame)
        ideas_buttons.pack(fill="x", pady=5)
        tk.Button(ideas_buttons, text="Save Ideas", command=self.save_ideas).pack(side="left", padx=5)
        
        self.ideas_display = scrolledtext.ScrolledText(ideas_frame, width=100, height=15, font=("Arial", 10))
        self.ideas_display.pack(fill="both", expand=True)

    def choose_output_directory(self):
        """Allow user to choose output directory."""
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.update()
        
        directory = filedialog.askdirectory(
            title="Select Base Output Directory",
            initialdir=self.output_dir if os.path.exists(self.output_dir) else os.path.expanduser("~"),
            parent=self.root
        )
        if directory:
            self.output_dir = directory
            self.dir_var.set(directory)
            self.session_dir = None
            self.session_var.set("No session created")
            self.status_label.config(text=f"Base directory set to: {directory}", fg="blue")
            self.show_message("info", "Directory Selected", "Please click 'New Session' to start working.")

    def reset_output_directory(self):
        """Reset output directory to default."""
        self.output_dir = OUTPUT_DIR
        self.dir_var.set(OUTPUT_DIR)
        self.session_dir = None
        self.session_var.set("No session created")
        self.status_label.config(text="Base directory reset to default", fg="blue")

    def create_new_session(self):
        """Create a new session directory."""
        if not self.output_dir:
            self.show_message("warning", "Warning", "Please select a base directory first.")
            return
            
        session_path = self.create_session_directory()
        session_name = os.path.basename(session_path)
        self.session_var.set(session_name)
        self.status_label.config(text=f"New session created: {session_name}", fg="green")
        self.show_message("info", "Session Created", f"New session created:\n{session_path}")
        self.transcription_file = None

    def record_audio(self):
        """Record audio for specified duration."""
        print("[DEBUG] Button clicked: Record Audio")
        
        if not self.session_dir:
            self.create_new_session()
            
        try:
            # Ler diretamente do Entry e validar
            try:
                duration = int(self.duration_entry.get())
                if duration < 1 or duration > 600:
                    raise ValueError("Duration must be between 1 and 600 seconds")
                print(f"[DEBUG] Using duration from entry: {duration} seconds")
            except ValueError:
                duration = 10
                self.status_label.config(text="Invalid duration. Using default (10s).", fg="red")
                print(f"[DEBUG] Invalid duration, using default: {duration} seconds")
            
            sample_rate = 44100
            self.status_label.config(text=f"Recording for {duration} seconds...", fg="red")
            self.root.update()
            
            self.show_message("info", "Recording", f"Click OK and start speaking!\nRecording for {duration} seconds...")

            print("[DEBUG] Starting audio recording...")
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()
            print("[DEBUG] Audio recording completed")
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            global AUDIO_FILE
            wav_file = f"{self.session_dir}/audio/audio_{timestamp}.wav"
            mp3_file = f"{self.session_dir}/audio/audio_{timestamp}.mp3"
            
            print(f"[DEBUG] Saving WAV file: {wav_file}")
            
            sf.write(wav_file, audio_data, sample_rate, subtype='PCM_16')
            AUDIO_FILE = wav_file
            
            print(f"[DEBUG] WAV file saved successfully. Size: {os.path.getsize(wav_file)} bytes")
            
            mp3_created = False
            print("[DEBUG] Attempting MP3 conversion...")
            if self._convert_wav_to_mp3(wav_file, mp3_file):
                mp3_created = True
                print(f"[DEBUG] MP3 conversion successful: {mp3_file}")
                self.status_label.config(text=f"Audio saved as MP3: {os.path.basename(mp3_file)}", fg="green")
            else:
                print("[DEBUG] MP3 conversion failed, keeping WAV")
                self.status_label.config(text=f"Audio saved as WAV: {os.path.basename(wav_file)}", fg="green")
            
            if mp3_created:
                self.show_message("info", "Success", f"Audio recorded and saved!\nWAV: {wav_file}\nMP3: {mp3_file}")
            else:
                self.show_message("info", "Success", f"Audio recorded and saved as WAV!\n{wav_file}\n\nNote: MP3 conversion requires FFmpeg")
            
            print(f"[DEBUG] Final AUDIO_FILE: {AUDIO_FILE}")
            
        except Exception as e:
            print(f"[ERROR] Recording failed: {str(e)}")
            self.status_label.config(text=f"Error recording: {str(e)}", fg="red")
            self.show_message("error", "Recording Error", str(e))

    def _convert_wav_to_mp3(self, wav_file, mp3_file):
        """Convert WAV to MP3 using FFmpeg if available."""
        try:
            import subprocess
            
            result = subprocess.run(["ffmpeg", "-version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False
            
            cmd = ["ffmpeg", "-i", wav_file, "-codec:a", "mp3", "-b:a", "192k", mp3_file, "-y"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            return result.returncode == 0 and os.path.exists(mp3_file)
            
        except Exception:
            return False

    def load_audio(self):
        print("[DEBUG] Button clicked: Load Audio")
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.update()
        
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.m4a *.flac"), ("All files", "*.*")],
            parent=self.root
        )
        if file_path:
            global AUDIO_FILE
            AUDIO_FILE = file_path
            self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}", fg="blue")

    def load_transcription(self):
        """Load an existing transcription file."""
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.update()
        
        file_path = filedialog.askopenfilename(
            title="Select Transcription File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            parent=self.root
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                self.transcription_file = file_path
                self.text_display.delete("1.0", tk.END)
                self.text_display.insert(tk.END, text)
                self.status_label.config(text=f"Loaded transcription: {os.path.basename(file_path)}", fg="blue")
                
            except Exception as e:
                self.show_message("error", "Error", f"Failed to load transcription: {str(e)}")

    def save_transcription_edits(self):
        """Save edited text back to transcription file."""
        if not self.session_dir:
            self.create_new_session()
            
        text = self.text_display.get("1.0", tk.END).strip()
        if not text:
            self.show_message("warning", "Warning", "No text to save.")
            return
            
        if not self.transcription_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.transcription_file = f"{self.session_dir}/text/transcription_{timestamp}.txt"
        
        try:
            with open(self.transcription_file, 'w', encoding='utf-8') as f:
                f.write(text)
            self.status_label.config(text=f"Text saved: {os.path.basename(self.transcription_file)}", fg="green")
            self.show_message("info", "Success", f"Text saved to:\n{self.transcription_file}")
        except Exception as e:
            self.show_message("error", "Error", f"Failed to save text: {str(e)}")

    def text_to_audio(self):
        """Convert text to audio using TTS."""
        if not self.session_dir:
            self.create_new_session()
            
        text = self.text_display.get("1.0", tk.END).strip()
        if not text:
            self.show_message("warning", "Warning", "No text to convert to audio.")
            return
            
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = f"{self.session_dir}/audio/tts_audio_{timestamp}.wav"
            
            tts_success = False
            method = ""
            
            if PYTTSX3_AVAILABLE and self._try_pyttsx3_tts(text, audio_file):
                tts_success = True
                method = "pyttsx3 (offline)"
            elif GTTS_AVAILABLE and self._try_gtts_tts(text, audio_file):
                tts_success = True
                method = "Google TTS (online)"
            elif self._try_windows_sapi_tts(text, audio_file):
                tts_success = True
                method = "Windows SAPI"
            
            if tts_success:
                self.status_label.config(text=f"TTS audio saved: {os.path.basename(audio_file)}", fg="green")
                self.show_message("info", "TTS Success", f"Audio generated successfully!\nMethod: {method}\nFile: {audio_file}")
            else:
                self.show_message("error", "TTS Failed", "Could not generate audio. Please install pyttsx3 or gtts:\npip install pyttsx3 gtts")
                
        except Exception as e:
            self.show_message("error", "TTS Error", f"Failed to generate audio: {str(e)}")

    def _try_pyttsx3_tts(self, text, audio_file):
        """Try pyttsx3 for TTS conversion."""
        try:
            engine = pyttsx3.init()
            
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate - 50)
            
            volume = engine.getProperty('volume')
            engine.setProperty('volume', 0.9)
            
            engine.save_to_file(text, audio_file)
            engine.runAndWait()
            
            return os.path.exists(audio_file)
            
        except Exception:
            return False

    def _try_gtts_tts(self, text, audio_file):
        """Try Google TTS for conversion."""
        try:
            portuguese_words = ["o", "a", "de", "para", "com", "em", "um", "uma", "que", "do", "da"]
            spanish_words = ["el", "la", "de", "para", "con", "en", "un", "una", "que", "del", "de la"]
            
            text_lower = text.lower()
            pt_count = sum(1 for word in portuguese_words if word in text_lower)
            es_count = sum(1 for word in spanish_words if word in text_lower)
            
            if pt_count > es_count and pt_count > 2:
                lang = 'pt'
            elif es_count > 2:
                lang = 'es'
            else:
                lang = 'en'
            
            tts = gTTS(text=text, lang=lang, slow=False)
            
            mp3_file = audio_file.replace('.wav', '.mp3')
            tts.save(mp3_file)
            
            if os.path.exists(mp3_file):
                try:
                    import subprocess
                    result = subprocess.run(["ffmpeg", "-i", mp3_file, audio_file, "-y"], 
                                          capture_output=True, timeout=10)
                    if result.returncode == 0:
                        os.remove(mp3_file)
                        return True
                    else:
                        return True
                except:
                    return True
            
            return False
            
        except Exception:
            return False

    def _try_windows_sapi_tts(self, text, audio_file):
        """Try Windows SAPI for TTS (Windows only)."""
        try:
            import subprocess
            import tempfile
            
            vbs_script = f"""
Dim objVoice, objFile
Set objVoice = CreateObject("SAPI.SpVoice")
Set objFile = CreateObject("SAPI.SpFileStream")

objFile.Open "{audio_file}", 3
Set objVoice.AudioOutputStream = objFile

objVoice.Speak "{text.replace('"', '""')}"

objFile.Close
Set objFile = Nothing
Set objVoice = Nothing
"""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.vbs', delete=False) as f:
                f.write(vbs_script)
                vbs_file = f.name
            
            try:
                result = subprocess.run(["cscript", "//NoLogo", vbs_file], 
                                      capture_output=True, timeout=30)
                os.unlink(vbs_file)
                return result.returncode == 0 and os.path.exists(audio_file)
            except:
                if os.path.exists(vbs_file):
                    os.unlink(vbs_file)
                return False
                
        except Exception:
            return False

    def transcribe_audio(self):
        """Transcribe the recorded audio and save to txt file."""
        print("[DEBUG] Button clicked: Transcribe Audio")
        
        if not os.path.exists(AUDIO_FILE):
            print(f"[ERROR] Audio file not found: {AUDIO_FILE}")
            self.show_message("error", "Error", "No audio file found. Please record audio first.")
            return
        
        print(f"[DEBUG] Transcribing audio file: {AUDIO_FILE}")
        print(f"[DEBUG] File size: {os.path.getsize(AUDIO_FILE)} bytes")
        
        try:
            r = sr.Recognizer()
            
            print("[DEBUG] Adjusting for ambient noise...")
            
            self.status_label.config(text="Processing audio file...", fg="orange")
            self.root.update()
            
            try:
                print("[DEBUG] Opening audio file...")
                with sr.AudioFile(AUDIO_FILE) as source:
                    print("[DEBUG] Audio file opened successfully")
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    print("[DEBUG] Recording audio from file...")
                    audio = r.record(source)
                    print("[DEBUG] Audio recorded from file successfully")
            except Exception as audio_error:
                print(f"[ERROR] Failed to read audio file: {str(audio_error)}")
                
                print("[DEBUG] Attempting audio conversion...")
                try:
                    if LIBROSA_AVAILABLE:
                        import librosa
                        audio_data, sr_rate = librosa.load(AUDIO_FILE, sr=22050)
                        
                        temp_wav = AUDIO_FILE.replace('.wav', '_temp.wav').replace('.mp3', '_temp.wav')
                        sf.write(temp_wav, audio_data, sr_rate, subtype='PCM_16')
                        
                        print(f"[DEBUG] Converted audio saved as: {temp_wav}")
                        
                        with sr.AudioFile(temp_wav) as source:
                            r.adjust_for_ambient_noise(source, duration=0.5)
                            audio = r.record(source)
                            
                        os.remove(temp_wav)
                    else:
                        raise ImportError("librosa not available")
                        
                except ImportError:
                    print("[WARNING] librosa not available, trying pydub...")
                    try:
                        if PYDUB_AVAILABLE:
                            from pydub import AudioSegment
                            
                            audio_segment = AudioSegment.from_file(AUDIO_FILE)
                            audio_segment = audio_segment.set_frame_rate(22050).set_channels(1)
                            
                            temp_wav = AUDIO_FILE.replace('.wav', '_temp.wav').replace('.mp3', '_temp.wav')
                            audio_segment.export(temp_wav, format="wav")
                            
                            print(f"[DEBUG] Converted audio with pydub: {temp_wav}")
                            
                            with sr.AudioFile(temp_wav) as source:
                                r.adjust_for_ambient_noise(source, duration=0.5)
                                audio = r.record(source)
                                
                            os.remove(temp_wav)
                        else:
                            raise ImportError("pydub not available")
                            
                    except ImportError:
                        print("[ERROR] No audio conversion libraries available")
                        print("[INFO] Install librosa or pydub for better audio format support:")
                        print("[INFO] pip install librosa")
                        print("[INFO] pip install pydub")
                        raise audio_error
            
            self.status_label.config(text="Transcribing...", fg="orange")
            self.root.update()
            
            languages = ["pt-BR", "en-US", "es-ES"]
            text = None
            detected_lang = None
            
            print("[DEBUG] Attempting transcription with multiple languages...")
            
            for lang in languages:
                try:
                    print(f"[DEBUG] Trying language: {lang}")
                    text = r.recognize_google(audio, language=lang)
                    detected_lang = lang
                    print(f"[DEBUG] Success with language: {lang}")
                    print(f"[DEBUG] Transcribed text: {text[:100]}...")
                    break
                except sr.UnknownValueError:
                    print(f"[DEBUG] Failed to understand audio in {lang}")
                    continue
                except sr.RequestError as e:
                    print(f"[ERROR] Google API error with {lang}: {str(e)}")
                    continue
            
            if text:
                global TRANSCRIBED_TEXT
                TRANSCRIBED_TEXT = text
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if not self.session_dir:
                    self.create_new_session()
                
                self.transcription_file = f"{self.session_dir}/text/transcription_{timestamp}.txt"
                
                print(f"[DEBUG] Saving transcription to: {self.transcription_file}")
                
                with open(self.transcription_file, 'w', encoding='utf-8') as f:
                    f.write(f"Original audio: {os.path.basename(AUDIO_FILE)}\n")
                    f.write(f"Detected language: {detected_lang}\n")
                    f.write(f"Transcription date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("-" * 50 + "\n\n")
                    f.write(text)
                
                self.text_display.delete("1.0", tk.END)
                self.text_display.insert(tk.END, text)
                
                self.status_label.config(text=f"Transcription completed! Language: {detected_lang}", fg="green")
                self.show_message("info", "Success", f"Audio transcribed successfully!\n\nLanguage: {detected_lang}\nText saved to:\n{self.transcription_file}")
                
                print("[DEBUG] Transcription completed successfully")
            else:
                error_msg = "Could not understand audio in any language (PT-BR, EN-US, ES-ES)"
                print(f"[ERROR] {error_msg}")
                raise sr.UnknownValueError(error_msg)
                
        except Exception as e:
            error_detail = str(e)
            print(f"[ERROR] Transcription failed: {error_detail}")
            self.status_label.config(text=f"Transcription error: {error_detail}", fg="red")
            self.show_message("error", "Transcription Error", 
                             f"Failed to transcribe audio:\n\n{error_detail}\n\n"
                             f"Suggestions:\n"
                             f"• Check if the audio file is valid\n"
                             f"• Try recording again\n"
                             f"• Check internet connection\n"
                             f"• Speak more clearly")

    def batch_transcribe(self):
        print("[DEBUG] Button clicked: Batch Transcribe")
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.update()
        
        directory = filedialog.askdirectory(
            title="Select Directory with Audio Files",
            parent=self.root
        )
        
        if not directory:
            return
            
        # Ensure session directory exists
        if not self.session_dir:
            self.create_new_session()
            
        try:
            # Find all audio files in the directory
            audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(Path(directory).glob(f"*{ext}"))
                audio_files.extend(Path(directory).glob(f"*{ext.upper()}"))
            
            if not audio_files:
                self.show_message("warning", "No Audio Files", "No audio files found in the selected directory.")
                return
            
            # Ask user for confirmation
            confirmed = self.show_message("yesno", "Batch Transcription", 
                                        f"Found {len(audio_files)} audio files.\n\nProceed with batch transcription?\n\nThis may take several minutes...")
            
            if not confirmed:
                return
            
            # Ask if user wants to generate music for each transcription
            generate_music = self.show_message("yesno", "Generate Music", 
                                             "Would you like to generate music code for each transcription?")
            
            # Create batch transcription directory
            batch_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_dir = f"{self.session_dir}/text/batch_transcription_{batch_timestamp}"
            Path(batch_dir).mkdir(exist_ok=True)
            
            # Create music directory if needed
            if generate_music:
                music_dir = f"{self.session_dir}/scripts/batch_music_{batch_timestamp}"
                Path(music_dir).mkdir(exist_ok=True)
            
            # Initialize recognizer
            r = sr.Recognizer()
            
            # Process each file
            successful = 0
            failed = 0
            results_summary = []
            
            for i, audio_file in enumerate(audio_files):
                try:
                    # Update status
                    progress = f"({i+1}/{len(audio_files)})"
                    filename = os.path.basename(audio_file)
                    self.status_label.config(text=f"Transcribing {progress}: {filename}", fg="orange")
                    self.root.update()
                    
                    # Transcribe the file
                    with sr.AudioFile(str(audio_file)) as source:
                        audio = r.record(source)
                    
                    # Try multiple languages
                    languages = ["pt-BR", "en-US", "es-ES"]
                    text = None
                    detected_lang = None
                    
                    for lang in languages:
                        try:
                            text = r.recognize_google(audio, language=lang)
                            detected_lang = lang
                            break
                        except sr.UnknownValueError:
                            continue
                    
                    if text:
                        # Save transcription
                        base_name = os.path.splitext(filename)[0]
                        output_file = f"{batch_dir}/{base_name}_transcription.txt"
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(f"Original file: {filename}\n")
                            f.write(f"Detected language: {detected_lang}\n")
                            f.write(f"Transcription date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write("-" * 50 + "\n\n")
                            f.write(text)
                        
                        successful += 1
                        results_summary.append(f"SUCCESS: {filename} -> {base_name}_transcription.txt")
                        
                        # Generate music if requested
                        if generate_music:
                            try:
                                # Generate music code for this transcription
                                music_code = self._generate_enhanced_music_code(text, filename)
                                music_file = f"{music_dir}/{base_name}_music.py"
                                
                                with open(music_file, 'w', encoding='utf-8') as f:
                                    f.write(music_code)
                                
                                results_summary.append(f"   Music code generated: {base_name}_music.py")
                                
                                # Try to execute the music code to generate MIDI
                                try:
                                    import subprocess
                                    result = subprocess.run(["python", music_file], 
                                                          capture_output=True, text=True, 
                                                          cwd=music_dir,
                                                          timeout=30)
                                    
                                    if result.returncode == 0:
                                        results_summary.append(f"   MIDI file generated successfully")
                                    else:
                                        results_summary.append(f"   WARNING: MIDI generation failed: {result.stderr[:100]}")
                                except Exception as midi_error:
                                    results_summary.append(f"   WARNING: Could not execute music code: {str(midi_error)[:100]}")
                                    
                            except Exception as music_error:
                                results_summary.append(f"   ERROR: Music generation failed: {str(music_error)[:100]}")
                    else:
                        failed += 1
                        results_summary.append(f"FAILED: {filename} -> Could not understand audio")
                        
                except Exception as e:
                    failed += 1
                    results_summary.append(f"ERROR: {filename} -> Error: {str(e)}")
            
            # Create summary report
            summary_content = f"""Batch Transcription Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source Directory: {directory}
Output Directory: {batch_dir}

Summary:
- Total files processed: {len(audio_files)}
- Successful transcriptions: {successful}
- Failed transcriptions: {failed}

Results:
{chr(10).join(results_summary)}

Files saved in: {batch_dir}
"""
            
            # Save summary
            summary_file = f"{batch_dir}/batch_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            # Update status and show results
            self.status_label.config(text=f"Batch completed: {successful} success, {failed} failed", fg="green")
            
            # Show summary in text display
            self.text_display.delete("1.0", tk.END)
            self.text_display.insert(tk.END, summary_content)
            
            self.show_message("info", "Batch Transcription Complete", 
                            f"Batch transcription completed!\n\n"
                            f"Successful: {successful}\n"
                            f"Failed: {failed}\n\n"
                            f"Files saved in:\n{batch_dir}")
            
        except Exception as e:
            self.status_label.config(text=f"Batch transcription error: {str(e)}", fg="red")
            self.show_message("error", "Batch Transcription Error", f"Failed to process batch transcription:\n{str(e)}")

    def generate_music(self):
        print("[DEBUG] Button clicked: Generate Music")
        # Try to get text from transcription file first, then from interface
        text = None
        if self.transcription_file and os.path.exists(self.transcription_file):
            try:
                with open(self.transcription_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            except Exception as e:
                self.show_message("error", "Error", f"Failed to read transcription file: {str(e)}")
                return
        
        if not text:
            text = self.text_display.get("1.0", tk.END).strip()
            
        if not text:
            self.show_message("warning", "Warning", "Please transcribe audio or enter some text first.")
            return
            
        if self.api_var.get() == "openai" and OPENAI_AVAILABLE:
            self._generate_music_openai(text)
        else:
            self._generate_music_local(text)

    def _generate_music_openai(self, text):
        """Generate music using OpenAI API."""
        try:
            # This would require API key setup
            prompt = f"""
            Create a Python script that generates music based on this description: "{text}"
            
            Use the midiutil library to create a MIDI file. Include:
            1. A simple melody that reflects the mood of the text
            2. Basic chord progression
            3. Appropriate tempo and time signature
            4. Save as a .mid file
            
            Make the code executable and well-commented.
            """
            
            # Mock response for now (replace with actual OpenAI call)
            code = self._generate_mock_music_code(text)
            self.music_display.delete("1.0", tk.END)
            self.music_display.insert(tk.END, code)
            self.show_message("info", "Success", "Music code generated! (Mock version)")
            
        except Exception as e:
            self.show_message("error", "Error", f"Failed to generate music: {str(e)}")

    def _generate_music_local(self, text):
        """Generate music code locally (mock version)."""
        code = self._generate_enhanced_music_code(text, "interactive_session")
        self.music_display.delete("1.0", tk.END)
        self.music_display.insert(tk.END, code)
        self.show_message("info", "Success", "Enhanced music code generated!")

    def _generate_mock_music_code(self, text):
        """Generate a mock music code based on text analysis."""
        # Simple mood analysis
        happy_words = ["happy", "joy", "bright", "feliz", "alegre"]
        sad_words = ["sad", "melancholy", "dark", "triste", "melancolia"]
        energetic_words = ["energy", "fast", "dance", "energia", "rapido"]
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in happy_words):
            tempo = 120
            scale = "C major"
            mood = "happy"
        elif any(word in text_lower for word in sad_words):
            tempo = 60
            scale = "A minor"
            mood = "melancholic"
        elif any(word in text_lower for word in energetic_words):
            tempo = 140
            scale = "E major"
            mood = "energetic"
        else:
            tempo = 100
            scale = "G major"
            mood = "neutral"
            
        return f'''# Generated music code based on: "{text}"
# Detected mood: {mood}
# Scale: {scale}, Tempo: {tempo} BPM

from midiutil import MIDIFile
import datetime
import os

# Create MIDI file
track = 0
channel = 0
time = 0
tempo = {tempo}
volume = 100

# Create MIDIFile object
midi = MIDIFile(1)
midi.addTempo(track, time, tempo)

# Define scale notes for {scale}
if "{scale}" == "C major":
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
elif "{scale}" == "A minor":
    notes = [57, 59, 60, 62, 64, 65, 67, 69]  # A minor scale
elif "{scale}" == "E major":
    notes = [64, 66, 68, 69, 71, 73, 75, 76]  # E major scale
else:
    notes = [67, 69, 71, 72, 74, 76, 78, 79]  # G major scale

# Generate melody based on mood: {mood}
duration = 1
for i, note in enumerate(notes):
    midi.addNote(track, channel, note, time + i * duration, duration, volume)

# Add harmony (simple chord progression)
chord_time = 0
chord_duration = 4
chords = [
    [notes[0], notes[2], notes[4]],  # I chord
    [notes[3], notes[5], notes[7]],  # IV chord
    [notes[4], notes[6], notes[1]],  # V chord
    [notes[0], notes[2], notes[4]]   # I chord
]

for chord in chords:
    for note in chord:
        midi.addNote(track, channel, note, chord_time, chord_duration, volume - 20)
    chord_time += chord_duration

# Save the file in current working directory (will be scripts folder)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"brainstorm_music_{{timestamp}}.mid"
with open(filename, "wb") as output_file:
    midi.writeFile(output_file)

print(f"MIDI file saved as: {{filename}}")
print(f"Mood: {mood}, Scale: {scale}, Tempo: {tempo} BPM")
'''

    def _generate_enhanced_music_code(self, text, source_filename=""):
        """Generate enhanced music code with more sophisticated analysis."""
        import random
        
        # Enhanced mood and language analysis
        mood_keywords = {
            "happy": ["happy", "joy", "bright", "feliz", "alegre", "love", "amor", "smile", "festa", "celebration"],
            "sad": ["sad", "melancholy", "dark", "triste", "melancolia", "cry", "choro", "saudade", "lonely"],
            "energetic": ["energy", "fast", "dance", "energia", "rapido", "run", "correr", "move", "action", "festa"],
            "calm": ["calm", "peace", "tranquil", "calma", "paz", "sereno", "quiet", "silencio", "relax"],
            "mysterious": ["mystery", "strange", "dark", "misterio", "estranho", "unknown", "shadow", "sombra"],
            "romantic": ["love", "amor", "heart", "coração", "romance", "passion", "paixão", "together", "juntos"]
        }
        
        # Analyze text
        text_lower = text.lower()
        word_count = len(text.split())
        
        # Count mood matches
        mood_scores = {}
        for mood, keywords in mood_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            mood_scores[mood] = score
        
        # Determine primary mood
        primary_mood = max(mood_scores, key=mood_scores.get)
        if mood_scores[primary_mood] == 0:
            primary_mood = "neutral"
        
        # Musical parameters based on mood
        mood_params = {
            "happy": {"tempo": 120, "scale": "C major", "time_sig": 4, "dynamics": 100},
            "sad": {"tempo": 60, "scale": "A minor", "time_sig": 3, "dynamics": 70},
            "energetic": {"tempo": 140, "scale": "E major", "time_sig": 4, "dynamics": 110},
            "calm": {"tempo": 70, "scale": "F major", "time_sig": 4, "dynamics": 60},
            "mysterious": {"tempo": 90, "scale": "D minor", "time_sig": 5, "dynamics": 80},
            "romantic": {"tempo": 80, "scale": "G major", "time_sig": 3, "dynamics": 85},
            "neutral": {"tempo": 100, "scale": "G major", "time_sig": 4, "dynamics": 90}
        }
        
        params = mood_params.get(primary_mood, mood_params["neutral"])
        
        # Generate melodic pattern based on text length and complexity
        if word_count < 10:
            melody_complexity = "simple"
            note_count = 8
        elif word_count < 30:
            melody_complexity = "moderate"
            note_count = 16
        else:
            melody_complexity = "complex"
            note_count = 32
            
        return f'''# Enhanced music generation for: "{source_filename}"
# Text preview: "{text[:100]}..."
# Analysis: {word_count} words, primary mood: {primary_mood}
# Musical parameters: {params}

from midiutil import MIDIFile
import random
import math
import datetime
import os

# Musical configuration
TEMPO = {params["tempo"]}
TIME_SIGNATURE = {params["time_sig"]}
SCALE = "{params["scale"]}"
BASE_VOLUME = {params["dynamics"]}
MELODY_COMPLEXITY = "{melody_complexity}"

# Create MIDI file with multiple tracks
midi = MIDIFile(3)  # 3 tracks: melody, harmony, bass

# Set tempo and time signature
for track in range(3):
    midi.addTempo(track, 0, TEMPO)
    midi.addTimeSignature(track, 0, TIME_SIGNATURE, 4, 24)

# Define scale notes
scales = {{
    "C major": [60, 62, 64, 65, 67, 69, 71, 72],
    "A minor": [57, 59, 60, 62, 64, 65, 67, 69],
    "E major": [64, 66, 68, 69, 71, 73, 75, 76],
    "F major": [65, 67, 69, 70, 72, 74, 76, 77],
    "D minor": [62, 64, 65, 67, 69, 70, 72, 74],
    "G major": [67, 69, 71, 72, 74, 76, 78, 79]
}}

base_notes = scales.get(SCALE, scales["C major"])

# Extend scale for more range
extended_notes = []
for octave in [-12, 0, 12]:
    extended_notes.extend([note + octave for note in base_notes])

# Track 1: Melody based on text rhythm
print("Generating melody...")
melody_time = 0
text_words = "{text}".split()[:{note_count}]

for i, word in enumerate(text_words):
    # Use word length to determine note duration
    duration = min(len(word) / 5.0, 2.0)
    
    # Use word position for pitch selection
    if "{primary_mood}" in ["happy", "energetic"]:
        # Upward tendency for happy/energetic moods
        note_index = (i + len(word)) % len(base_notes)
        note = base_notes[note_index] + random.choice([0, 12])
    elif "{primary_mood}" in ["sad", "mysterious"]:
        # Downward tendency for sad/mysterious moods
        note_index = (len(base_notes) - 1 - i) % len(base_notes)
        note = base_notes[note_index] + random.choice([-12, 0])
    else:
        # Random walk for neutral/other moods
        note_index = random.randint(0, len(base_notes) - 1)
        note = base_notes[note_index]
    
    # Add note with dynamics variation
    volume = BASE_VOLUME + random.randint(-10, 10)
    midi.addNote(0, 0, note, melody_time, duration, volume)
    
    # Add occasional rest
    if random.random() < 0.1:
        melody_time += duration + 0.5
    else:
        melody_time += duration

# Track 2: Harmony (chord progression)
print("Generating harmony...")
chord_progressions = {{
    "happy": [[0, 2, 4], [3, 5, 7], [4, 6, 1], [0, 2, 4]],  # I-IV-V-I
    "sad": [[0, 2, 4], [5, 7, 2], [3, 5, 7], [0, 2, 4]],   # i-vi-iv-i
    "energetic": [[0, 2, 4], [4, 6, 1], [5, 7, 2], [3, 5, 7]], # I-V-vi-IV
    "calm": [[0, 2, 4], [2, 4, 6], [3, 5, 7], [0, 2, 4]],  # I-iii-IV-I
    "mysterious": [[0, 2, 4], [1, 3, 5], [4, 6, 1], [0, 2, 4]], # i-ii°-V-i
    "romantic": [[0, 2, 4], [3, 5, 7], [1, 3, 5], [4, 6, 1]],  # I-IV-ii-V
    "neutral": [[0, 2, 4], [3, 5, 7], [4, 6, 1], [0, 2, 4]]    # I-IV-V-I
}}

progression = chord_progressions.get("{primary_mood}", chord_progressions["neutral"])
chord_time = 0
chord_duration = 4  # Each chord lasts 4 beats

for chord_indices in progression * 4:  # Repeat progression 4 times
    for note_index in chord_indices:
        if note_index < len(base_notes):
            note = base_notes[note_index]
            midi.addNote(1, 1, note, chord_time, chord_duration, BASE_VOLUME - 20)
    chord_time += chord_duration

# Track 3: Bass line
print("Generating bass line...")
bass_time = 0
bass_pattern = [0, 0, 4, 4, 5, 5, 0, 0]  # Simple bass pattern

for i in range(int(melody_time / 2)):
    note_index = bass_pattern[i % len(bass_pattern)]
    bass_note = base_notes[note_index] - 24  # Two octaves lower
    
    # Rhythm variation based on mood
    if "{primary_mood}" in ["energetic", "happy"]:
        duration = 0.5  # Faster bass for upbeat moods
    elif "{primary_mood}" in ["calm", "romantic"]:
        duration = 2.0  # Slower bass for calm moods
    else:
        duration = 1.0
    
    midi.addNote(2, 2, bass_note, bass_time, duration, BASE_VOLUME - 10)
    bass_time += duration

# Add metadata
midi.addText(0, 0, "Generated from: {source_filename}")
midi.addText(0, 0, "Mood: {primary_mood}")
midi.addText(0, 0, "Text: {text[:50]}...")

# Save MIDI file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"brainstorm_{{os.path.splitext('{source_filename}')[0]}}_{{timestamp}}.mid"
if not filename.startswith("brainstorm_"):
    filename = f"brainstorm_music_{{timestamp}}.mid"

with open(filename, "wb") as output_file:
    midi.writeFile(output_file)

print(f"MIDI file created: {{filename}}")
print(f"Mood: {primary_mood}, Scale: {params['scale']}, Tempo: {params['tempo']} BPM")
print(f"Complexity: {melody_complexity} ({{word_count}} words)")
print("Tracks: Melody, Harmony, Bass")
'''

    def generate_image(self):
        print("[DEBUG] Button clicked: Generate Image")
        # Try to get text from transcription file first, then from interface
        text = None
        if self.transcription_file and os.path.exists(self.transcription_file):
            try:
                with open(self.transcription_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            except Exception as e:
                self.show_message("error", "Error", f"Failed to read transcription file: {str(e)}")
                return
        
        if not text:
            text = self.text_display.get("1.0", tk.END).strip()
            
        if not text:
            self.show_message("warning", "Warning", "Please transcribe audio or enter some text first.")
            return
            
        # Generate image prompt
        image_prompt = f"""
Image Generation Prompt:

Original text: "{text}"

Artistic interpretation:
- Style: Digital art, neural networks aesthetic
- Colors: Vibrant, technology-inspired palette
- Composition: Abstract representation of brain waves and musical notes
- Elements: Flowing data streams, geometric patterns, synaptic connections
- Mood: Futuristic, creative, inspiring

Detailed prompt for AI image generation:
"Create a digital artwork that visualizes the concept of '{text}' through the lens of neuroscience and artificial intelligence. Include flowing neural networks, musical notation floating in space, and vibrant colors that represent creativity and innovation. Style should be modern, abstract, and technology-focused."

Alternative prompts:
1. Minimalist: "Simple, elegant visualization of '{text}' with clean lines and soft colors"
2. Realistic: "Photorealistic representation of '{text}' in a laboratory setting"
3. Artistic: "Impressionist painting style interpretation of '{text}' with bold brushstrokes"

Tips for best results:
- Use specific style keywords (photorealistic, digital art, watercolor, etc.)
- Include lighting preferences (soft, dramatic, natural)
- Specify composition (portrait, landscape, close-up)
- Add mood descriptors (peaceful, energetic, mysterious)
"""
        
        self.image_display.delete("1.0", tk.END)
        self.image_display.insert(tk.END, image_prompt)
        self.show_message("info", "Success", "Image prompts generated!")

    def generate_ideas(self):
        print("[DEBUG] Button clicked: Generate Ideas")
        # Try to get text from transcription file first, then from interface
        text = None
        if self.transcription_file and os.path.exists(self.transcription_file):
            try:
                with open(self.transcription_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            except Exception as e:
                self.show_message("error", "Error", f"Failed to read transcription file: {str(e)}")
                return
        
        if not text:
            text = self.text_display.get("1.0", tk.END).strip()
            
        if not text:
            self.show_message("warning", "Warning", "Please transcribe audio or enter some text first.")
            return
            
        ideas = f"""
Creative Ideas Generator
Based on: "{text}"

Musical Applications:
1. Create a ambient soundscape for meditation
2. Compose a short jingle for presentations
3. Generate background music for videos
4. Create rhythmic patterns for dance
5. Develop a musical signature/theme

Visual Arts:
1. Create album cover artwork
2. Design presentation backgrounds
3. Generate social media graphics
4. Create animated visualizations
5. Design logo concepts

Performance Ideas:
1. Spoken word performance with music
2. Interactive sound installation
3. Dance choreography inspiration
4. Theater scene background
5. Podcast intro/outro

Research Applications:
1. Study music's effect on brain waves
2. Analyze speech patterns and emotions
3. Explore AI creativity boundaries
4. Investigate sound-color synesthesia
5. Test human-AI collaboration

Educational Uses:
1. Language learning with music
2. Memory enhancement techniques
3. Creative writing prompts
4. Art therapy exercises
5. STEM demonstration tools

Innovation Projects:
1. Voice-controlled music creation
2. Emotion-responsive soundtracks
3. Collaborative AI composition
4. Real-time music visualization
5. Therapeutic sound design

Digital Applications:
1. Mobile app for instant composition
2. Web-based collaborative platform
3. VR/AR music creation environment
4. Social sharing platform
5. Educational game development
"""
        
        self.ideas_display.delete("1.0", tk.END)
        self.ideas_display.insert(tk.END, ideas)
        self.show_message("info", "Success", "Creative ideas generated!")

    def save_music_code(self):
        print("[DEBUG] Button clicked: Save Music Code")
        # Ensure session directory exists
        if not self.session_dir:
            self.create_new_session()
            
        code = self.music_display.get("1.0", tk.END).strip()
        if not code:
            self.show_message("warning", "Warning", "No music code to save.")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.session_dir}/scripts/music_code_{timestamp}.py"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            self.show_message("info", "Success", f"Music code saved as: {filename}")
        except Exception as e:
            self.show_message("error", "Error", f"Failed to save: {str(e)}")

    def execute_music_code(self):
        print("[DEBUG] Button clicked: Execute Music Code")
        # Ensure session directory exists
        if not self.session_dir:
            self.create_new_session()
            
        code = self.music_display.get("1.0", tk.END).strip()
        if not code:
            self.show_message("warning", "Warning", "No music code to execute.")
            return
            
        try:
            # Save code temporarily and execute in session scripts directory
            temp_file = f"{self.session_dir}/scripts/temp_music.py"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            import subprocess
            # Execute with working directory set to scripts folder so MIDI saves there
            result = subprocess.run(["python", temp_file], 
                                  capture_output=True, text=True, 
                                  cwd=f"{self.session_dir}/scripts")
            
            if result.returncode == 0:
                self.show_message("info", "Success", f"Code executed successfully!\n\nOutput:\n{result.stdout}")
            else:
                self.show_message("error", "Execution Error", f"Error:\n{result.stderr}")
                
        except Exception as e:
            self.show_message("error", "Error", f"Failed to execute: {str(e)}")

    def generate_mp3(self):
        print("[DEBUG] Button clicked: Generate MP3")
        if not self.session_dir:
            self.show_message("error", "Error", "No session created. Please create a session first.")
            return
            
        try:
            # Check if there are any MIDI files in the session scripts directory
            scripts_dir = f"{self.session_dir}/scripts"
            if not os.path.exists(scripts_dir):
                self.show_message("error", "Error", "No scripts directory found. Please generate and execute music code first.")
                return
            
            midi_files = [f for f in os.listdir(scripts_dir) if f.endswith('.mid')]
            
            if not midi_files:
                self.show_message("error", "Error", "No MIDI files found. Please execute music code first to generate MIDI files.")
                return
            
            # Get the most recent MIDI file
            midi_files.sort(reverse=True)
            midi_file = f"{scripts_dir}/{midi_files[0]}"
            
            # Try different MP3 conversion methods
            mp3_created = False
            conversion_log = []
            
            # Method 1: Try FluidSynth + FFmpeg (most common on Windows)
            mp3_file = f"{scripts_dir}/{midi_files[0].replace('.mid', '.mp3')}"
            
            # Try FluidSynth conversion
            if self._try_fluidsynth_conversion(midi_file, mp3_file):
                mp3_created = True
                conversion_log.append("SUCCESS: FluidSynth conversion successful!")
            else:
                conversion_log.append("FAILED: FluidSynth not available or failed")
                
                # Method 2: Try TiMidity conversion
                if self._try_timidity_conversion(midi_file, mp3_file):
                    mp3_created = True
                    conversion_log.append("SUCCESS: TiMidity conversion successful!")
                else:
                    conversion_log.append("FAILED: TiMidity not available or failed")
                    
                    # Method 3: Try pygame MIDI playback (basic)
                    if self._try_pygame_conversion(midi_file, mp3_file):
                        mp3_created = True
                        conversion_log.append("SUCCESS: Basic pygame conversion successful!")
                    else:
                        conversion_log.append("FAILED: Pygame conversion failed")
            
            # Create report
            if mp3_created:
                result_msg = f"""
MP3 Generation SUCCESSFUL!

MIDI file: {midi_files[0]}
MP3 file: {os.path.basename(mp3_file)}
Location: {mp3_file}

Conversion log:
{chr(10).join(conversion_log)}

Your MP3 file is ready to play!
"""
                self.show_message("info", "MP3 Generated!", f"MP3 file created successfully:\n{mp3_file}")
            else:
                result_msg = f"""
MP3 Generation - Manual Steps Required

MIDI file: {midi_files[0]}
Location: {midi_file}

Automatic conversion failed. Please try these manual methods:

1. **Online Converters (Easiest):**
   - Upload your MIDI file to: zamzar.com, convertio.co, or online-audio-converter.com

2. **Desktop Software:**
   - Audacity (Free): Import MIDI → Export as MP3
   - FL Studio, GarageBand, or other DAW software

3. **Install FluidSynth (Advanced):**
   - Download from: https://www.fluidsynth.org/
   - Install FFmpeg: https://ffmpeg.org/
   - Run: fluidsynth -F output.wav soundfont.sf2 {midi_files[0]}
   - Then: ffmpeg -i output.wav output.mp3

Conversion log:
{chr(10).join(conversion_log)}

The MIDI file is ready for manual conversion!
"""
                self.show_message("warning", "Manual Conversion Required", f"Automatic MP3 conversion failed.\nPlease see instructions in the display.")
            
            # Show in music display
            self.music_display.delete("1.0", tk.END)
            self.music_display.insert(tk.END, result_msg)
            
            # Save report
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"{self.session_dir}/text/mp3_generation_report_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(result_msg)
            
        except Exception as e:
            self.show_message("error", "Error", f"Failed to generate MP3: {str(e)}")

    def _try_fluidsynth_conversion(self, midi_file, mp3_file):
        """Try to convert MIDI to MP3 using FluidSynth + FFmpeg."""
        try:
            import subprocess
            
            # Check if fluidsynth is available
            result = subprocess.run(["fluidsynth", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False
            
            # Check if ffmpeg is available
            result = subprocess.run(["ffmpeg", "-version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False
            
            # Try to find a soundfont file (very basic search)
            soundfont_paths = [
                "C:/Windows/System32/FluidR3_GM.sf2",
                "C:/Program Files/FluidSynth/sf2/FluidR3_GM.sf2",
                "/usr/share/soundfonts/FluidR3_GM.sf2",
                "/usr/share/sounds/sf2/FluidR3_GM.sf2"
            ]
            
            soundfont = None
            for sf in soundfont_paths:
                if os.path.exists(sf):
                    soundfont = sf
                    break
                    
            if not soundfont:
                return False
            
            # Convert MIDI to WAV using FluidSynth
            wav_file = midi_file.replace('.mid', '.wav')
            cmd1 = ["fluidsynth", "-ni", soundfont, midi_file, "-F", wav_file, "-r", "44100"]
            result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=30)
            
            if result1.returncode != 0 or not os.path.exists(wav_file):
                return False
            
            # Convert WAV to MP3 using FFmpeg
            cmd2 = ["ffmpeg", "-i", wav_file, "-codec:a", "mp3", "-b:a", "192k", mp3_file, "-y"]
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
            
            # Clean up WAV file
            if os.path.exists(wav_file):
                os.remove(wav_file)
            
            return result2.returncode == 0 and os.path.exists(mp3_file)
            
        except Exception:
            return False

    def _try_timidity_conversion(self, midi_file, mp3_file):
        """Try to convert MIDI to MP3 using TiMidity + FFmpeg."""
        try:
            import subprocess
            
            # Check if timidity is available
            result = subprocess.run(["timidity", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False
            
            # Check if ffmpeg is available
            result = subprocess.run(["ffmpeg", "-version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False
            
            # Convert MIDI to WAV using TiMidity
            wav_file = midi_file.replace('.mid', '.wav')
            cmd1 = ["timidity", midi_file, "-Ow", "-o", wav_file]
            result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=30)
            
            if result1.returncode != 0 or not os.path.exists(wav_file):
                return False
            
            # Convert WAV to MP3 using FFmpeg
            cmd2 = ["ffmpeg", "-i", wav_file, "-codec:a", "mp3", "-b:a", "192k", mp3_file, "-y"]
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
            
            # Clean up WAV file
            if os.path.exists(wav_file):
                os.remove(wav_file)
            
            return result2.returncode == 0 and os.path.exists(mp3_file)
            
        except Exception:
            return False

    def _try_pygame_conversion(self, midi_file, mp3_file):
        """Try basic pygame-based conversion (limited functionality)."""
        try:
            if not PYGAME_AVAILABLE:
                return False
                
            # This is a very basic approach and won't work well for MIDI
            # Pygame can play MIDI but not easily convert to MP3
            # This is mainly a placeholder for future implementation
            return False
            
        except Exception:
            return False

    def save_image_info(self):
        """Save image generation information."""
        # Ensure session directory exists
        if not self.session_dir:
            self.create_new_session()
            
        content = self.image_display.get("1.0", tk.END).strip()
        if not content:
            self.show_message("warning", "Warning", "No image information to save.")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.session_dir}/images/image_prompts_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            self.show_message("info", "Success", f"Image information saved as: {filename}")
        except Exception as e:
            self.show_message("error", "Error", f"Failed to save: {str(e)}")

    def save_ideas(self):
        """Save the generated ideas."""
        # Ensure session directory exists
        if not self.session_dir:
            self.create_new_session()
            
        ideas = self.ideas_display.get("1.0", tk.END).strip()
        if not ideas:
            self.show_message("warning", "Warning", "No ideas to save.")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.session_dir}/text/creative_ideas_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(ideas)
            self.show_message("info", "Success", f"Ideas saved as: {filename}")
        except Exception as e:
            self.show_message("error", "Error", f"Failed to save: {str(e)}")

    def open_image_url(self):
        """Open image URL if available."""
        content = self.image_display.get("1.0", tk.END)
        # This would extract URLs from the content if they exist
        self.show_message("info", "Info", "Image URL opening functionality would be implemented with actual API integration.")

    def batch_generate_music(self):
        """Generate music for multiple existing transcription files."""
        # Bring window to front before showing dialog
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.update()
        
        directory = filedialog.askdirectory(
            title="Select Directory with Transcription Files (.txt)",
            parent=self.root
        )
        
        if not directory:
            return
            
        # Ensure session directory exists
        if not self.session_dir:
            self.create_new_session()
            
        try:
            # Find all text files in the directory
            text_files = list(Path(directory).glob("*.txt"))
            
            if not text_files:
                self.show_message("warning", "No Text Files", "No .txt files found in the selected directory.")
                return
            
            # Ask user for confirmation
            confirmed = self.show_message("yesno", "Batch Music Generation", 
                                        f"Found {len(text_files)} text files.\n\n"
                                        f"Generate music for each transcription?\n\n"
                                        f"This will create Python music scripts and MIDI files.")
            
            if not confirmed:
                return
            
            # Create batch music directory
            batch_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            music_dir = f"{self.session_dir}/scripts/batch_music_{batch_timestamp}"
            Path(music_dir).mkdir(parents=True, exist_ok=True)
            
            # Process each file
            successful = 0
            failed = 0
            results_summary = []
            
            for i, text_file in enumerate(text_files):
                try:
                    # Update status
                    progress = f"({i+1}/{len(text_files)})"
                    filename = os.path.basename(text_file)
                    self.status_label.config(text=f"Generating music {progress}: {filename}", fg="orange")
                    self.root.update()
                    
                    # Read transcription
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    if text:
                        # Generate music code
                        base_name = os.path.splitext(filename)[0]
                        music_code = self._generate_enhanced_music_code(text, filename)
                        music_file = f"{music_dir}/{base_name}_music.py"
                        
                        with open(music_file, 'w', encoding='utf-8') as f:
                            f.write(music_code)
                        
                        successful += 1
                        results_summary.append(f"SUCCESS: {filename} -> {base_name}_music.py")
                        
                        # Try to execute the music code to generate MIDI
                        try:
                            import subprocess
                            result = subprocess.run(["python", music_file], 
                                                  capture_output=True, text=True, 
                                                  cwd=music_dir,
                                                  timeout=30)
                            
                            if result.returncode == 0:
                                results_summary.append(f"   MIDI file generated successfully")
                                
                                # Try to convert MIDI to MP3
                                midi_files = [f for f in os.listdir(music_dir) 
                                            if f.endswith('.mid') and base_name in f]
                                if midi_files:
                                    midi_path = f"{music_dir}/{midi_files[0]}"
                                    mp3_path = midi_path.replace('.mid', '.mp3')
                                    if self._try_convert_midi_to_mp3(midi_path, mp3_path):
                                        results_summary.append(f"   MP3 file created successfully")
                            else:
                                results_summary.append(f"   WARNING: MIDI generation failed")
                        except Exception as exec_error:
                            results_summary.append(f"   WARNING: Could not execute music code")
                    else:
                        failed += 1
                        results_summary.append(f"FAILED: {filename} -> Empty file")
                        
                except Exception as e:
                    failed += 1
                    results_summary.append(f"ERROR: {filename} -> Error: {str(e)[:100]}")
            
            # Create summary report
            summary_content = f"""Batch Music Generation Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source Directory: {directory}
Output Directory: {music_dir}

Summary:
- Total files processed: {len(text_files)}
- Successful generations: {successful}
- Failed generations: {failed}

Results:
{chr(10).join(results_summary)}

Music files saved in: {music_dir}

Next steps:
1. Review the generated Python music scripts
2. Execute them to create MIDI files
3. Convert MIDI to MP3 using external tools if needed
4. Use the generated music in your creative projects!
"""
            
            # Save summary
            summary_file = f"{music_dir}/batch_music_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            # Update status and show results
            self.status_label.config(text=f"Batch completed: {successful} success, {failed} failed", fg="green")
            
            # Show summary in music display
            self.music_display.delete("1.0", tk.END)
            self.music_display.insert(tk.END, summary_content)
            
            self.show_message("info", "Batch Music Generation Complete", 
                            f"Batch music generation completed!\n\n"
                            f"Successful: {successful}\n"
                            f"Failed: {failed}\n\n"
                            f"Files saved in:\n{music_dir}")
            
        except Exception as e:
            self.status_label.config(text=f"Batch music generation error: {str(e)}", fg="red")
            self.show_message("error", "Batch Music Error", f"Failed to process batch music generation:\n{str(e)}")

    def _try_convert_midi_to_mp3(self, midi_path, mp3_path):
        """Try to convert MIDI to MP3 using available tools."""
        # Try FluidSynth first
        if self._try_fluidsynth_conversion(midi_path, mp3_path):
            return True
        # Try TiMidity
        elif self._try_timidity_conversion(midi_path, mp3_path):
            return True
        return False

    def run(self):
        """Start the application."""
        self.root.mainloop()

    # Método para validar a duração
    def validate_duration(self):
        """Validate and set the recording duration from the input field."""
        try:
            duration = int(self.duration_entry.get())
            if duration < 1 or duration > 600:
                raise ValueError("Duration must be between 1 and 600 seconds.")
            self.recording_duration = duration
            self.status_label.config(text=f"Duration set to: {duration} seconds", fg="blue")
            print(f"[DEBUG] Recording duration set to: {duration} seconds")
        except ValueError as e:
            self.show_message("error", "Invalid Duration", str(e))
            self.recording_duration = 10  # Fallback to default
            self.status_label.config(text="Invalid duration. Using default (10s).", fg="red")

def run_brainstorm():
    """Main function to run the Brainstorm module."""
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting Enhanced Brainstorm...")
    
    app = BrainstormApp()
    app.run()

if __name__ == "__main__":
    run_brainstorm()

