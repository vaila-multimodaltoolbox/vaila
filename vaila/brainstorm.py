"""
Project: vailá
Script: brainstorm.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 18 February 2025
Update Date: 3 June 2025
Version: 0.2.0

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

# Global variables
AUDIO_FILE = "audio.wav"
TRANSCRIBED_TEXT = ""
OUTPUT_DIR = "brainstorm_outputs"

class BrainstormApp:
    def __init__(self):
        self.root = tk.Tk()
        self.output_dir = OUTPUT_DIR  # Add instance variable for output directory
        self.transcription_file = None  # Store path to current transcription file
        self.session_dir = None  # Current session directory
        self.setup_ui()
        # Don't create directories automatically - only when user selects location
        
    def show_message(self, msg_type, title, message):
        """Show message box always on top of the main window."""
        self.root.lift()  # Bring main window to front first
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
        
    def create_session_directory(self):
        """Create a new session directory with timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = f"{self.output_dir}/brainstorm_{timestamp}"
        
        # Create session directory and subdirectories
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
        self.root.title("vailá Brainstorm - Voice to Creative AI")
        self.root.geometry("1000x800")
        
        # Keep window always on top and focus
        self.root.attributes('-topmost', True)
        self.root.focus_force()
        
        # Main frame with scrollbar
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        header = tk.Label(main_frame, text="Brainstorm: Voice to AI Creative Assistant", 
                         font=("Arial", 16, "bold"))
        header.pack(pady=10)
        
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
        
        # Session directory display
        session_frame = tk.Frame(output_frame)
        session_frame.pack(fill="x", pady=2)
        
        tk.Label(session_frame, text="Current Session:").pack(side="left")
        self.session_var = tk.StringVar(value="No session created")
        self.session_label = tk.Label(session_frame, textvariable=self.session_var, 
                                    relief="sunken", anchor="w", width=50, fg="blue")
        self.session_label.pack(side="left", padx=5, fill="x", expand=True)
        
        tk.Button(dir_selection_frame, text="Browse", command=self.choose_output_directory,
                 bg="#607D8B", fg="white", font=("Arial", 9)).pack(side="right", padx=5)
        tk.Button(dir_selection_frame, text="New Session", command=self.create_new_session,
                 bg="#4CAF50", fg="white", font=("Arial", 9)).pack(side="right", padx=5)
        tk.Button(dir_selection_frame, text="Reset", command=self.reset_output_directory,
                 bg="#795548", fg="white", font=("Arial", 9)).pack(side="right")
        
        # Audio section
        audio_frame = tk.LabelFrame(main_frame, text="1. Audio Recording", font=("Arial", 12, "bold"))
        audio_frame.pack(fill="x", pady=5)
        
        audio_buttons = tk.Frame(audio_frame)
        audio_buttons.pack(pady=5)
        
        tk.Button(audio_buttons, text="Record Audio", command=self.record_audio,
                 bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        
        # Duration selection
        duration_frame = tk.Frame(audio_frame)
        duration_frame.pack(pady=2)
        
        tk.Label(duration_frame, text="Recording Duration (seconds):").pack(side="left")
        self.duration_var = tk.StringVar(value="10")
        duration_combo = ttk.Combobox(duration_frame, textvariable=self.duration_var, width=8, state="readonly")
        duration_combo['values'] = ("5", "10", "15", "30", "60", "120", "300", "600")
        duration_combo.pack(side="left", padx=5)
        tk.Button(audio_buttons, text="Transcribe", command=self.transcribe_audio,
                 bg="#2196F3", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(audio_buttons, text="Load Audio", command=self.load_audio,
                 bg="#FF9800", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(audio_buttons, text="Load Transcription", command=self.load_transcription,
                 bg="#795548", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        
        # Status label
        self.status_label = tk.Label(audio_frame, text="Ready to record...", fg="green")
        self.status_label.pack(pady=5)
        
        # Text editing section
        text_frame = tk.LabelFrame(main_frame, text="2. Text Editing & Prompt", font=("Arial", 12, "bold"))
        text_frame.pack(fill="both", expand=True, pady=5)
        
        text_header = tk.Frame(text_frame)
        text_header.pack(fill="x", pady=5)
        
        tk.Label(text_header, text="Edit your transcribed text or write a custom prompt:").pack(side="left", anchor="w")
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
        # Bring window to front before showing dialog
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
            self.session_dir = None  # Reset session directory
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
        
        # Reset transcription file for new session
        self.transcription_file = None

    def record_audio(self):
        """Record audio for specified duration."""
        # Ensure session directory exists
        if not self.session_dir:
            self.create_new_session()
            
        try:
            # Get duration from interface
            try:
                duration = int(self.duration_var.get())
            except:
                duration = 10  # fallback
                
            sample_rate = 44100
            self.status_label.config(text=f"Recording for {duration} seconds...", fg="red")
            self.root.update()
            
            # Show countdown dialog
            self.show_message("info", "Recording", f"Click OK and start speaking!\nRecording for {duration} seconds...")

            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()
            
            # Create timestamped filename using session directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            global AUDIO_FILE
            AUDIO_FILE = f"{self.session_dir}/audio/audio_{timestamp}.wav"
            
            sf.write(AUDIO_FILE, audio_data, sample_rate)
            
            self.status_label.config(text=f"Audio saved: {AUDIO_FILE}", fg="green")
            self.show_message("info", "Success", f"Audio recorded and saved as '{AUDIO_FILE}'")
            
        except Exception as e:
            self.status_label.config(text=f"Error recording: {str(e)}", fg="red")
            self.show_message("error", "Recording Error", str(e))

    def load_audio(self):
        """Load an existing audio file."""
        # Bring window to front before showing dialog
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
        # Bring window to front before showing dialog
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
        # Ensure session directory exists
        if not self.session_dir:
            self.create_new_session()
            
        text = self.text_display.get("1.0", tk.END).strip()
        if not text:
            self.show_message("warning", "Warning", "No text to save.")
            return
            
        if not self.transcription_file:
            # Create new transcription file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.transcription_file = f"{self.session_dir}/text/transcription_{timestamp}.txt"
        
        try:
            with open(self.transcription_file, 'w', encoding='utf-8') as f:
                f.write(text)
            self.status_label.config(text=f"Text saved: {os.path.basename(self.transcription_file)}", fg="green")
            self.show_message("info", "Success", f"Text saved to:\n{self.transcription_file}")
        except Exception as e:
            self.show_message("error", "Error", f"Failed to save text: {str(e)}")

    def transcribe_audio(self):
        """Transcribe the recorded audio and save to txt file."""
        if not os.path.exists(AUDIO_FILE):
            self.show_message("error", "Error", "No audio file found. Please record audio first.")
            return
            
        try:
            r = sr.Recognizer()
            self.status_label.config(text="Transcribing...", fg="orange")
            self.root.update()
            
            with sr.AudioFile(AUDIO_FILE) as source:
                audio = r.record(source)
            
            # Try multiple languages
            languages = ["pt-BR", "en-US", "es-ES"]
            text = None
            
            for lang in languages:
                try:
                    text = r.recognize_google(audio, language=lang)
                    break
                except sr.UnknownValueError:
                    continue
            
            if text:
                global TRANSCRIBED_TEXT
                TRANSCRIBED_TEXT = text
                
                # Save transcription to txt file
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Ensure session directory exists
                if not self.session_dir:
                    self.create_new_session()
                
                self.transcription_file = f"{self.session_dir}/text/transcription_{timestamp}.txt"
                
                with open(self.transcription_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                # Display in interface
                self.text_display.delete("1.0", tk.END)
                self.text_display.insert(tk.END, text)
                
                self.status_label.config(text=f"Transcription saved: {self.transcription_file}", fg="green")
                self.show_message("info", "Success", f"Audio transcribed and saved as:\n{self.transcription_file}")
            else:
                raise sr.UnknownValueError("Could not understand audio in any language")
                
        except Exception as e:
            self.status_label.config(text=f"Transcription error: {str(e)}", fg="red")
            self.show_message("error", "Transcription Error", str(e))

    def generate_music(self):
        """Generate Python music code based on the transcription file."""
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
        code = self._generate_mock_music_code(text)
        self.music_display.delete("1.0", tk.END)
        self.music_display.insert(tk.END, code)
        self.show_message("info", "Success", "Music code generated! (Local mock version)")

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

    def generate_image(self):
        """Generate image description/prompt based on transcription file."""
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
🎨 Image Generation Prompt:

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

💡 Tips for best results:
- Use specific style keywords (photorealistic, digital art, watercolor, etc.)
- Include lighting preferences (soft, dramatic, natural)
- Specify composition (portrait, landscape, close-up)
- Add mood descriptors (peaceful, energetic, mysterious)
"""
        
        self.image_display.delete("1.0", tk.END)
        self.image_display.insert(tk.END, image_prompt)
        self.show_message("info", "Success", "Image prompts generated!")

    def generate_ideas(self):
        """Generate creative ideas based on transcription file."""
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
💡 Creative Ideas Generator
Based on: "{text}"

🎵 Musical Applications:
1. Create a ambient soundscape for meditation
2. Compose a short jingle for presentations
3. Generate background music for videos
4. Create rhythmic patterns for dance
5. Develop a musical signature/theme

🎨 Visual Arts:
1. Create album cover artwork
2. Design presentation backgrounds
3. Generate social media graphics
4. Create animated visualizations
5. Design logo concepts

🎭 Performance Ideas:
1. Spoken word performance with music
2. Interactive sound installation
3. Dance choreography inspiration
4. Theater scene background
5. Podcast intro/outro

🔬 Research Applications:
1. Study music's effect on brain waves
2. Analyze speech patterns and emotions
3. Explore AI creativity boundaries
4. Investigate sound-color synesthesia
5. Test human-AI collaboration

🎓 Educational Uses:
1. Language learning with music
2. Memory enhancement techniques
3. Creative writing prompts
4. Art therapy exercises
5. STEM demonstration tools

🚀 Innovation Projects:
1. Voice-controlled music creation
2. Emotion-responsive soundtracks
3. Collaborative AI composition
4. Real-time music visualization
5. Therapeutic sound design

📱 Digital Applications:
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
        """Save the generated music code."""
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
        """Execute the generated music code."""
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
        """Generate MP3 from MIDI file if available."""
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
                conversion_log.append("✅ FluidSynth conversion successful!")
            else:
                conversion_log.append("❌ FluidSynth not available or failed")
                
                # Method 2: Try TiMidity conversion
                if self._try_timidity_conversion(midi_file, mp3_file):
                    mp3_created = True
                    conversion_log.append("✅ TiMidity conversion successful!")
                else:
                    conversion_log.append("❌ TiMidity not available or failed")
                    
                    # Method 3: Try pygame MIDI playback (basic)
                    if self._try_pygame_conversion(midi_file, mp3_file):
                        mp3_created = True
                        conversion_log.append("✅ Basic pygame conversion successful!")
                    else:
                        conversion_log.append("❌ Pygame conversion failed")
            
            # Create report
            if mp3_created:
                result_msg = f"""
🎵 MP3 Generation SUCCESSFUL! 🎉

MIDI file: {midi_files[0]}
MP3 file: {os.path.basename(mp3_file)}
Location: {mp3_file}

Conversion log:
{chr(10).join(conversion_log)}

✅ Your MP3 file is ready to play!
"""
                self.show_message("info", "MP3 Generated!", f"MP3 file created successfully:\n{mp3_file}")
            else:
                result_msg = f"""
🎵 MP3 Generation - Manual Steps Required

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

    def run(self):
        """Start the application."""
        self.root.mainloop()

def run_brainstorm():
    """Main function to run the Brainstorm module."""
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting Enhanced Brainstorm...")
    
    app = BrainstormApp()
    app.run()

if __name__ == "__main__":
    run_brainstorm()

