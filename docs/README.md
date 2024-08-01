<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vailá - Versatile Anarcho-Integrated Multimodal Toolbox</title>
    <link rel="stylesheet" href="style.css">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
</head>
<body>
    <div class="container mt-5">
        <div class="text-center">
            <img src="images/vaila.png" alt="vailá Logo" class="logo-img mb-4">
        </div>
        <h1 class="text-center">vailá - Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox</h1>
        <p>This toolbox provides various functionalities for analyzing multimodal data. Below is a brief explanation of each feature available in this toolbox along with instructions on what to do for each command.</p>
        
        <h2>Overview of the Toolbox:</h2>
        <p>vailá is a toolbox designed to enhance biomechanics analysis by leveraging multiple motion capture systems. "vailá" is an expression that blends the sound of the French word "voilà" with the direct encouragement in Portuguese "vai lá," meaning "go there and do it!" This toolbox empowers you to explore, experiment, and create without the constraints of expensive commercial software. vailá and use your imagination!</p>
        <div class="text-center">
            <img src="images/gui.png" alt="GUI Image" class="center-img mb-4">
        </div>

        <h2>vailá manifest!</h2>
        <p>If you have new ideas or suggestions, please send them to us.
        Join us in the liberation from paid software with the "vailá: Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox."</p>

        <p>In front of you stands a versatile and anarcho-integrated tool, designed to challenge the boundaries of commercial systems.
        This software, not a mere substitute, is a symbol of innovation and freedom, now available and accessible.
        However, this brave visitation of an old problem is alive and determined to eliminate these venal and
        virulent barriers that protect the monopoly of expensive software, ensuring the dissemination of knowledge and accessibility.
        We have left the box open with vailá to insert your ideas and processing in a liberated manner.
        The only verdict is versatility; a vendetta against exorbitant costs, held as a vow, not in vain, for the value and veracity of which shall one day
        vindicate the vigilant and the virtuous in the field of motion analysis.
        Surely, this torrent of technology tends to be very innovative, so let me simply add that it is a great honor to have you with us
        and you may call this tool vailá.</p>

        <p>― The vailá idea!</p>

        <p>"vailá" é uma expressão que mistura a sonoridade da palavra francesa "voilà" com o incentivo direto em português "vai lá".
        É uma chamada à ação, um convite à iniciativa e à liberdade de explorar, experimentar e criar sem as limitações impostas por softwares comerciais caros.
        "vailá" significa "vai lá e faça!", encorajando todos a aproveitar o poder das ferramentas versáteis e integradas do "vailá: Análise versátil da libertação anarquista integrada na caixa de ferramentas multimodal" para realizar análises com dados de múltiplos sistemas.</p>

        <p>"vailá" is an expression that blends the sound of the French word "voilà" with the direct encouragement in Portuguese BR "vai lá."
        It is a call to action, an invitation to initiative and freedom to explore, experiment, and create without the constraints imposed by expensive commercial software.
        "vailá" means "go there and do it!", encouraging everyone to harness the power of the "vailá: Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox" to perform analysis with data from multiple systems.</p>

        <h2>Available Multimodal Analysis:</h2>
        
        <h3>1. IMU Analysis</h3>
        <p><strong>Description:</strong> Analyze Inertial Measurement Unit (IMU) data stored in CSV or C3D format. It processes and interprets motion data from wearable IMU sensors.</p>
        <p><strong>Instructions:</strong> Use the GUI or CLI to select your IMU CSV or C3D files. The analysis will process these files to extract and visualize the motion data.</p>
        
        <h3>2. Kinematic Cluster Analysis</h3>
        <p><strong>Description:</strong> Analyze cluster marker data stored in CSV format. It helps in interpreting the motion data collected from marker-based motion capture systems.</p>
        <p><strong>Instructions:</strong> Use the GUI or CLI to select your cluster CSV files. You will be prompted to enter the sample rate and the configuration for the trunk and pelvis. Optionally, provide anatomical position data.</p>
        
        <h3>3. Kinematic Mocap Full Body Analysis</h3>
        <p><strong>Description:</strong> Analyze full-body motion capture data in C3D format. It processes the data captured by motion capture systems that track full-body movements.</p>
        <p><strong>Instructions:</strong> Use the GUI or CLI to select your C3D files. The analysis will convert these files and process the motion capture data.</p>
        
        <h3>4. Markerless 2D with video</h3>
        <p><strong>Description:</strong> Analyze 2D video data without using markers. It processes the motion data from 2D video recordings to extract relevant motion parameters.</p>
        <p><strong>Instructions:</strong> Use the GUI or CLI to select your 2D video files. The analysis will process these videos to extract motion data.</p>
        
        <h3>5. Markerless 3D with multiple videos</h3>
        <p><strong>Description:</strong> Process 3D video data without markers. It analyzes 3D video recordings to extract motion data and parameters.</p>
        <p><strong>Instructions:</strong> Use the GUI or CLI to select your 3D video files. The analysis will process these videos to extract 3D motion data.</p>
        
        <h2>Available Tools:</h2>
        
        <h3>6. Edit CSV</h3>
        <p><strong>Description:</strong> Organize and rearrange data files within a specified directory. This tool allows you to reorder columns, cut and select rows, and modify the global reference system of the data. It also allows for unit conversion and data reshaping, ensuring alignment with the desired coordinate system and consistent data formatting.</p>
        <p><strong>Instructions:</strong> Use the GUI or CLI to select your CSV files. The tool will guide you through the process of editing the files as needed, including reordering columns, cutting and selecting rows, modifying the laboratory coordinate system, and converting units.</p>
        
        <h3>7. Convert C3D data to CSV</h3>
        <p><strong>Description:</strong> Convert motion capture data files from C3D format to CSV format. This makes it easier to analyze and manipulate the data using standard data analysis tools.</p>
        <p><strong>Instructions:</strong> Use the GUI or CLI to select your C3D files. The tool will convert these files to CSV format.</p>
        
        <h3>8. Metadata info</h3>
        <p><strong>Description:</strong> Extract metadata information from video files. This is useful for synchronizing video data with other motion capture data.</p>
        <p><strong>Instructions:</strong> Use the GUI or CLI to select your video files. The tool will extract and display metadata information for each video file.</p>
        
        <h3>9. Cut videos based on list</h3>
        <p><strong>Description:</strong> Cut video files based on a list of specified time intervals. This is useful for segmenting videos into relevant portions for analysis.</p>
        <p><strong>Instructions:</strong> Use the GUI or CLI to select your video files and the list of time intervals. The tool will process these files to cut the videos accordingly.</p>
        
        <h3>10. Draw a black box around videos</h3>
        <p><strong>Description:</strong> Overlay a black box around video frames to highlight specific areas of interest. This can help in focusing on particular regions during analysis.</p>
        <p><strong>Instructions:</strong> Use the GUI or CLI to select your video files. The tool will overlay a black box around the specified area in the videos.</p>
        
        <h3>11. Compress videos to HEVC (H.265)</h3>
        <p><strong>Description:</strong> Compress video files using the HEVC (H.265) codec. This helps in reducing the file size while maintaining video quality.</p>
        <p><strong>Instructions:</strong> Use the GUI or CLI to select your video files. The tool will compress these videos using the HEVC codec.</p>
        
        <h3>12. Compress videos to H.264</h3>
        <p><strong>Description:</strong> Compress video files using the H.264 codec. This helps in reducing the file size while maintaining video quality.</p>
        <p><strong>Instructions:</strong> Use the GUI or CLI to select your video files. The tool will compress these videos using the H.264 codec.</p>
        
        <h2>Additional Commands:</h2>
        <p><strong>h</strong> - Display this help message.</p>
        <p><strong>exit</strong> - Exit the program.</p>
        
        <p>To use this toolbox, simply select the desired option by typing the corresponding number and pressing Enter. You can also type 'h' to view this help message or 'exit' to quit the program.</p>
    </div>
</body>
<footer>
    <div class="container mt-5">
        <p class="text-center">© 2024 vailá - Multimodal Toolbox. All rights reserved.</p>
    </div>
</footer>
</html>
