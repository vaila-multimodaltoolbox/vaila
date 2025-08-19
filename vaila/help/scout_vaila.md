# vailá Scout - Sports Scouting Tool

## Overview

vailá Scout is an integrated GUI application for annotating sports events on a virtual soccer field and generating quick analyses (e.g., heatmaps). It's designed to fit the vailá project style and doesn't require external field images - the field is drawn to scale using standard FIFA dimensions (105m x 68m).

## Features

### Core Functionality
- **Real-time annotation**: Click on the field to mark player actions
- **Timer control**: Built-in timer with start/pause/reset functionality
- **Team management**: Support for home and away teams with player rosters
- **Action tracking**: Customizable action types with visual symbols
- **Result tracking**: Success/fail/neutral outcomes for each action
- **Heatmap generation**: Visual analysis of player movement patterns

### Data Management
- **CSV import/export**: Load and save event data in CSV format
- **TOML configuration**: Flexible configuration system for teams, actions, and field settings
- **Skout conversion**: Convert data from Skout.exe ASCII export format

### User Interface
- **Responsive design**: Adapts to different screen sizes
- **Keyboard shortcuts**: Quick access to common functions
- **Visual feedback**: Color-coded events and player markers
- **Roster management**: Easy player selection and team switching

## Installation

### Requirements
- Python 3.x
- tkinter (GUI)
- matplotlib
- seaborn
- pandas
- rich (for console prints)
- toml (for writing config) and tomllib/tomli (for reading config)

### Running the Application

#### Option 1: From vailá GUI
Click the "Scout" button in the main vailá interface.

#### Option 2: Command Line
```bash
python vaila.py
```

#### Option 3: Direct Module
```bash
python -m vaila.scout_vaila
```

#### Option 4: Direct Script
```bash
cd vaila
python scout_vaila.py
```

## Usage Guide

### Getting Started

1. **Launch the application** using any of the methods above
2. **Load or create a configuration** (TOML file) with your teams and actions
3. **Select your team** (use 'T' key to toggle between home/away)
4. **Choose an action** from the dropdown menu
5. **Set the result** (success/fail/neutral)
6. **Click on the field** to mark player actions
7. **Use the timer** to track game time

### Basic Workflow

1. **Configure Teams**: Set up home and away teams with player rosters
2. **Define Actions**: Create custom action types (pass, shot, tackle, etc.)
3. **Annotate Events**: Click on the field to mark player actions
4. **Track Time**: Use the built-in timer or manual time entry
5. **Save Data**: Export your annotations to CSV format
6. **Analyze**: Generate heatmaps and other visualizations

### Field Interaction

- **Left Click**: Mark action with "success" result
- **Right Click**: Mark action with "fail" result  
- **Middle Click**: Mark action with "neutral" result
- **Ctrl + Click**: Remove events near clicked position
- **Shift + Click**: Mark action with "fail" result (touchpad alternative)
- **Alt + Click**: Mark action with "neutral" result (touchpad alternative)

## Configuration

### TOML Configuration File

The application uses TOML configuration files to store:
- Team information (names, colors, players)
- Action definitions (names, codes, symbols, colors)
- Field settings (dimensions, units)
- Visual preferences (marker sizes, colors)

### Default Configuration

When first launched, the application creates a default configuration with:
- FIFA standard field (105m x 68m)
- Two teams (HOME/AWAY) with sample players
- Common soccer actions (pass, shot, tackle, etc.)
- Standard visual settings

### Creating Custom Configurations

1. Use "Create Template" to generate a base configuration
2. Edit the TOML file manually or use the configuration tools
3. Load your custom configuration
4. Save changes as needed

## Data Format

### CSV Export Format

Events are exported in CSV format with the following columns:
```
timestamp_s, team, player_name, player, action, action_code, result, pos_x_m, pos_y_m
```

**Column Descriptions:**
- `timestamp_s`: Time in seconds from game start
- `team`: Team name (HOME/AWAY or custom)
- `player_name`: Player's full name
- `player`: Player number
- `action`: Action name (pass, shot, etc.)
- `action_code`: Numeric action code
- `result`: success/fail/neutral
- `pos_x_m`: X position in meters (0-105)
- `pos_y_m`: Y position in meters (0-68)

### Coordinate System

- **Origin**: Bottom-left corner (0,0)
- **X-axis**: Left to right (0-105m)
- **Y-axis**: Bottom to top (0-68m)
- **Units**: Meters (FIFA standard)

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+S` | Save CSV |
| `Ctrl+O` | Load CSV |
| `Ctrl+K` | Clear events |
| `H` | Show heatmap |
| `R` | Reset timer |
| `Space` | Start/Pause clock |
| `?` | Open help |
| `T` | Toggle current team (home/away) |
| `Ctrl+L` | Load config |
| `Ctrl+Shift+S` | Save config |
| `Ctrl+T` | Rename teams |
| `Digits 0–9` | Enter action code; Enter apply; Backspace edit; Esc clear |
| `Mouse` | Left=success, Right=fail, Middle=neutral |
| `Ctrl+Right Click` | Remove events near clicked position |

## Advanced Features

### Heatmap Generation

1. Click "Heatmap" button or press 'H'
2. Select team and/or player filters
3. View density plot of player actions
4. Analyze movement patterns and hot zones

### Skout Data Conversion

Convert data from Skout.exe ASCII export format:
1. Go to **Tools** → **Convert Skout to vailá**
2. Select your Skout .txt file
3. Choose output directory
4. Enter team name
5. Get both CSV and TOML configuration files

### Player Management

- **Roster View**: See all players for both teams
- **Quick Selection**: Click on player names to select them
- **Auto-numbering**: Automatic player assignment options
- **Name Mapping**: Link player numbers to full names

### Timer Features

- **Live Timer**: Real-time game clock
- **Manual Entry**: Set specific timestamps
- **Pause/Resume**: Control timing during breaks
- **Reset**: Start over with clean timing

## Troubleshooting

### Common Issues

**Application won't start:**
- Check Python version (3.x required)
- Verify all dependencies are installed
- Ensure tkinter is available

**Configuration not loading:**
- Check TOML file syntax
- Verify file permissions
- Use "Create Template" to generate valid config

**Field not displaying:**
- Check matplotlib installation
- Verify display settings
- Try resizing the window

**Data not saving:**
- Check write permissions in target directory
- Verify CSV format compatibility
- Ensure sufficient disk space

### Performance Tips

- Close other applications when working with large datasets
- Use appropriate field dimensions for your analysis
- Limit the number of simultaneous events for better performance
- Save work frequently to avoid data loss

## File Structure

```
vaila/
├── scout_vaila.py          # Main application
├── help/
│   ├── scout_vaila.md      # This documentation
│   ├── scout_vaila.html    # HTML version
│   ├── scout_vaila_pt.md   # Portuguese documentation
│   └── scout_vaila_pt.html # Portuguese HTML version
└── models/
    └── vaila_scout_config.toml  # Default configuration
```

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

## Support

For issues, questions, or contributions:
- **Email**: paulosantiago@usp.br
- **GitHub**: https://github.com/vaila-multimodaltoolbox/vaila
- **Documentation**: See help files in the vaila/help directory

---

**Version**: 0.1.4  
**Last Updated**: August 19, 2025  
**Author**: Paulo Roberto Pereira Santiago


