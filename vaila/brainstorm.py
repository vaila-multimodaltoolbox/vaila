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

# Global variables
AUDIO_FILE = "audio.wav"
TRANSCRIBED_TEXT = ""
OUTPUT_DIR = "brainstorm_outputs"

class BrainstormApp:
    def __init__(self):
        self.root = tk.Tk()
        self.setup_ui()
        self.setup_directories()
        
    def setup_directories(self):
        """Create output directories if they don't exist."""
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        Path(f"{OUTPUT_DIR}/audio").mkdir(exist_ok=True)
        Path(f"{OUTPUT_DIR}/scripts").mkdir(exist_ok=True)
        Path(f"{OUTPUT_DIR}/images").mkdir(exist_ok=True)
        
    def setup_ui(self):
        """Setup the user interface."""
        self.root.title("vailá Brainstorm - Voice to Creative AI")
        self.root.geometry("1000x800")
        
        # Main frame with scrollbar
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        header = tk.Label(main_frame, text="Brainstorm: Voice to AI Creative Assistant", 
                         font=("Arial", 16, "bold"))
        header.pack(pady=10)
        
        # Audio section
        audio_frame = tk.LabelFrame(main_frame, text="1. Audio Recording", font=("Arial", 12, "bold"))
        audio_frame.pack(fill="x", pady=5)
        
        audio_buttons = tk.Frame(audio_frame)
        audio_buttons.pack(pady=5)
        
        tk.Button(audio_buttons, text="Record Audio", command=self.record_audio,
                 bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(audio_buttons, text="Transcribe", command=self.transcribe_audio,
                 bg="#2196F3", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(audio_buttons, text="Load Audio", command=self.load_audio,
                 bg="#FF9800", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        
        # Status label
        self.status_label = tk.Label(audio_frame, text="Ready to record...", fg="green")
        self.status_label.pack(pady=5)
        
        # Text editing section
        text_frame = tk.LabelFrame(main_frame, text="2. Text Editing & Prompt", font=("Arial", 12, "bold"))
        text_frame.pack(fill="both", expand=True, pady=5)
        
        tk.Label(text_frame, text="Edit your transcribed text or write a custom prompt:").pack(anchor="w")
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
        
        tk.Label(api_frame, text="API Mode:").pack(side="left")
        self.api_var = tk.StringVar(value="local")
        tk.Radiobutton(api_frame, text="Local (Mock)", variable=self.api_var, value="local").pack(side="left")
        tk.Radiobutton(api_frame, text="OpenAI", variable=self.api_var, value="openai").pack(side="left")
        tk.Radiobutton(api_frame, text="Custom", variable=self.api_var, value="custom").pack(side="left")
        
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
        tk.Button(music_buttons, text="Save Code", command=self.save_music_code).pack(side="left", padx=5)
        tk.Button(music_buttons, text="Execute Code", command=self.execute_music_code).pack(side="left", padx=5)
        
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

    def record_audio(self, duration=10):
        """Record audio for specified duration."""
        try:
            sample_rate = 44100
            self.status_label.config(text=f"Recording for {duration} seconds...", fg="red")
            self.root.update()
            
            # Show countdown dialog
            messagebox.showinfo("Recording", f"Click OK and start speaking!\nRecording for {duration} seconds...")

            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()
            
            # Create timestamped filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            global AUDIO_FILE
            AUDIO_FILE = f"{OUTPUT_DIR}/audio/audio_{timestamp}.wav"
            
            sf.write(AUDIO_FILE, audio_data, sample_rate)
            
            self.status_label.config(text=f"Audio saved: {AUDIO_FILE}", fg="green")
            messagebox.showinfo("Success", f"Audio recorded and saved as '{AUDIO_FILE}'")
            
        except Exception as e:
            self.status_label.config(text=f"Error recording: {str(e)}", fg="red")
            messagebox.showerror("Recording Error", str(e))

    def load_audio(self):
        """Load an existing audio file."""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.m4a *.flac"), ("All files", "*.*")]
        )
        if file_path:
            global AUDIO_FILE
            AUDIO_FILE = file_path
            self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}", fg="blue")

    def transcribe_audio(self):
        """Transcribe the recorded audio."""
        if not os.path.exists(AUDIO_FILE):
            messagebox.showerror("Error", "No audio file found. Please record audio first.")
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
                self.text_display.delete("1.0", tk.END)
                self.text_display.insert(tk.END, text)
                self.status_label.config(text="Transcription successful!", fg="green")
                messagebox.showinfo("Success", "Audio transcribed successfully!")
            else:
                raise sr.UnknownValueError("Could not understand audio in any language")
                
        except Exception as e:
            self.status_label.config(text=f"Transcription error: {str(e)}", fg="red")
            messagebox.showerror("Transcription Error", str(e))

    def generate_music(self):
        """Generate Python music code based on the text."""
        text = self.text_display.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text first.")
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
            messagebox.showinfo("Success", "Music code generated! (Mock version)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate music: {str(e)}")

    def _generate_music_local(self, text):
        """Generate music code locally (mock version)."""
        code = self._generate_mock_music_code(text)
        self.music_display.delete("1.0", tk.END)
        self.music_display.insert(tk.END, code)
        messagebox.showinfo("Success", "Music code generated! (Local mock version)")

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

# Save the file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"brainstorm_music_{timestamp}.mid"
with open(filename, "wb") as output_file:
    midi.writeFile(output_file)

print(f"MIDI file saved as: {{filename}}")
print(f"Mood: {mood}, Scale: {scale}, Tempo: {tempo} BPM")
'''

    def generate_image(self):
        """Generate image description/prompt."""
        text = self.text_display.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text first.")
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
        messagebox.showinfo("Success", "Image prompts generated!")

    def generate_ideas(self):
        """Generate creative ideas based on the text."""
        text = self.text_display.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text first.")
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
        messagebox.showinfo("Success", "Creative ideas generated!")

    def save_music_code(self):
        """Save the generated music code."""
        code = self.music_display.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "No music code to save.")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/scripts/music_code_{timestamp}.py"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            messagebox.showinfo("Success", f"Music code saved as: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def execute_music_code(self):
        """Execute the generated music code."""
        code = self.music_display.get("1.0", tk.END).strip()
        if not code:
            messagebox.showwarning("Warning", "No music code to execute.")
            return
            
        try:
            # Save code temporarily and execute
            temp_file = f"{OUTPUT_DIR}/scripts/temp_music.py"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            import subprocess
            result = subprocess.run(["python", temp_file], capture_output=True, text=True)
            
            if result.returncode == 0:
                messagebox.showinfo("Success", f"Code executed successfully!\n\nOutput:\n{result.stdout}")
            else:
                messagebox.showerror("Execution Error", f"Error:\n{result.stderr}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to execute: {str(e)}")

    def save_image_info(self):
        """Save image generation information."""
        content = self.image_display.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "No image information to save.")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/images/image_prompts_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            messagebox.showinfo("Success", f"Image information saved as: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def save_ideas(self):
        """Save the generated ideas."""
        ideas = self.ideas_display.get("1.0", tk.END).strip()
        if not ideas:
            messagebox.showwarning("Warning", "No ideas to save.")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_DIR}/creative_ideas_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(ideas)
            messagebox.showinfo("Success", f"Ideas saved as: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def open_image_url(self):
        """Open image URL if available."""
        content = self.image_display.get("1.0", tk.END)
        # This would extract URLs from the content if they exist
        messagebox.showinfo("Info", "Image URL opening functionality would be implemented with actual API integration.")

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

