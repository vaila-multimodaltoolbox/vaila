import os

def main():
    original_file = "/home/preto/Preto/vaila/vaila/interp_smooth_split.py"
    new_class_file = "/home/preto/Preto/vaila/new_tkinter_class.py"
    
    with open(original_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    with open(new_class_file, "r", encoding="utf-8") as f:
        new_class_code = f.read()

    start_idx = -1
    end_idx = -1
    
    # Find class InterpolationConfigDialog:
    for i, line in enumerate(lines):
        if line.startswith("class InterpolationConfigDialog:"):
            start_idx = i
            break
            
    # Find def winter_residual_analysis
    for i in range(start_idx, len(lines)):
        if line.startswith("def winter_residual_analysis"):
            end_idx = i
            break
            
    # As a fallback if exact string isn't found
    if end_idx == -1:
        for i, line in enumerate(lines):
            if "def winter_residual_analysis" in line:
                end_idx = i
                break

    if start_idx != -1 and end_idx != -1:
        new_lines = lines[:start_idx] + [new_class_code + "\n\n"] + lines[end_idx:]
        with open(original_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"Successfully replaced from line {start_idx} to {end_idx}.")
    else:
        print(f"Failed to find bounds. Start: {start_idx}, End: {end_idx}")

if __name__ == "__main__":
    main()
